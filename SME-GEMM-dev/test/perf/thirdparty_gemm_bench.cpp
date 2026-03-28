#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(BENCH_BACKEND_OPENBLAS) || defined(BENCH_BACKEND_ARMPL)
#include "cblas.h"
#elif defined(BENCH_BACKEND_LIBXSMM)
#include "libxsmm.h"
#else
#error "one backend macro must be defined"
#endif

namespace
{

struct Options
{
    std::vector<int> sizes;
    int batch = 8;
    int warmup = 10;
    int iters = 1000;
    bool verify_results = false;
    std::string csv_output;
    std::string layout = "tn";
    std::string dtype = "fp64";
};

std::vector<int> parse_sizes(const std::string &text)
{
    std::vector<int> result;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        if (!item.empty())
        {
            result.push_back(std::stoi(item));
        }
    }
    if (result.empty())
    {
        throw std::invalid_argument("sizes must not be empty");
    }
    return result;
}

Options parse_args(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto need_value = [&](const char *name) -> std::string {
            if (i + 1 >= argc)
            {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--sizes")
        {
            options.sizes = parse_sizes(need_value("--sizes"));
        }
        else if (arg == "--batch")
        {
            options.batch = std::stoi(need_value("--batch"));
        }
        else if (arg == "--warmup")
        {
            options.warmup = std::stoi(need_value("--warmup"));
        }
        else if (arg == "--iters")
        {
            options.iters = std::stoi(need_value("--iters"));
        }
        else if (arg == "--csv-output")
        {
            options.csv_output = need_value("--csv-output");
        }
        else if (arg == "--layout")
        {
            options.layout = need_value("--layout");
        }
        else if (arg == "--dtype")
        {
            options.dtype = need_value("--dtype");
        }
        else if (arg == "--verify-results")
        {
            options.verify_results = true;
        }
        else
        {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (options.sizes.empty())
    {
        options.sizes = parse_sizes("2,4,6,8,10,12,14,16,18,20");
    }
    if (options.csv_output.empty())
    {
        throw std::invalid_argument("--csv-output is required");
    }
    if (options.batch <= 0 || options.warmup < 0 || options.iters <= 0)
    {
        throw std::invalid_argument("batch and iters must be > 0, warmup must be >= 0");
    }
    if (options.layout != "nn" && options.layout != "nt" && options.layout != "tn" && options.layout != "tt")
    {
        throw std::invalid_argument("--layout must be one of nn, nt, tn, tt");
    }
    if (options.dtype != "fp64" && options.dtype != "fp32")
    {
        throw std::invalid_argument("--dtype must be one of fp64, fp32");
    }
    return options;
}

template <typename T>
constexpr const char *dtype_name()
{
    if constexpr (std::is_same_v<T, double>)
    {
        return "fp64";
    }
    else
    {
        return "fp32";
    }
}

template <typename T>
void fill_inputs(std::vector<T> &a, std::vector<T> &b)
{
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        a[i] = static_cast<T>(0.25 + static_cast<double>((i % 17) + 1) * 0.03125);
    }
    for (std::size_t i = 0; i < b.size(); ++i)
    {
        b[i] = static_cast<T>(0.50 + static_cast<double>((i % 13) + 1) * 0.015625);
    }
}

template <typename T>
double checksum(const std::vector<T> &c)
{
    return std::accumulate(
        c.begin(), c.end(), 0.0, [](double acc, T value) { return acc + static_cast<double>(value); });
}

template <typename T>
double max_abs_diff(const std::vector<T> &lhs, const std::vector<T> &rhs)
{
    if (lhs.size() != rhs.size())
    {
        throw std::invalid_argument("max_abs_diff requires vectors of equal length");
    }

    double max_diff = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i)
    {
        max_diff = std::max(max_diff, std::abs(static_cast<double>(lhs[i]) - static_cast<double>(rhs[i])));
    }
    return max_diff;
}

char transa_for_layout(const std::string &layout) { return static_cast<char>(std::toupper(layout.at(0))); }

char transb_for_layout(const std::string &layout) { return static_cast<char>(std::toupper(layout.at(1))); }

std::string layout_note_name(const std::string &layout)
{
    std::string result = layout;
    for (char &ch : result)
    {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return result;
}

template <typename T>
T load_a(const T *a_batch, int i, int inner, int size, char transa)
{
    if (transa == 'N')
    {
        return a_batch[static_cast<std::size_t>(i) * size + inner];
    }
    return a_batch[static_cast<std::size_t>(inner) * size + i];
}

template <typename T>
T load_b(const T *b_batch, int inner, int j, int size, char transb)
{
    if (transb == 'N')
    {
        return b_batch[static_cast<std::size_t>(inner) * size + j];
    }
    return b_batch[static_cast<std::size_t>(j) * size + inner];
}

template <typename T>
void compute_reference(const std::vector<T> &a,
                       const std::vector<T> &b,
                       std::vector<T> &c_ref,
                       int size,
                       int batch,
                       char transa,
                       char transb)
{
    std::fill(c_ref.begin(), c_ref.end(), static_cast<T>(0));
    for (int batch_idx = 0; batch_idx < batch; ++batch_idx)
    {
        const T *a_batch = a.data() + static_cast<std::size_t>(batch_idx) * size * size;
        const T *b_batch = b.data() + static_cast<std::size_t>(batch_idx) * size * size;
        T *c_batch = c_ref.data() + static_cast<std::size_t>(batch_idx) * size * size;
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                T sum = static_cast<T>(0);
                for (int inner = 0; inner < size; ++inner)
                {
                    sum += load_a(a_batch, i, inner, size, transa) * load_b(b_batch, inner, j, size, transb);
                }
                c_batch[static_cast<std::size_t>(i) * size + j] = sum;
            }
        }
    }
}

const char *backend_name()
{
#if defined(BENCH_BACKEND_OPENBLAS)
    return "openblas";
#elif defined(BENCH_BACKEND_LIBXSMM)
    return "libxsmm";
#elif defined(BENCH_BACKEND_ARMPL)
    return "armpl";
#else
    return "unknown";
#endif
}

std::string backend_note(const std::string &layout, const std::string &dtype, bool verify_results, double verify_diff)
{
    std::ostringstream note;
    note << "threads=1;batched_reference_loop=1;dtype=" << dtype
         << ";layout=rowmajor;trans=" << layout_note_name(layout) << ";verify=" << (verify_results ? "on" : "off");
#if defined(BENCH_BACKEND_OPENBLAS)
    note << ";thread_api=openblas_set_num_threads";
#elif defined(BENCH_BACKEND_ARMPL)
    note << ";armpl_variant=lp64";
#endif
    if (verify_results)
    {
        note << ";max_abs_diff=" << std::scientific << std::setprecision(3) << verify_diff;
    }
    return note.str();
}

template <typename T>
void kernel(int size, const std::string &layout, const T *a, const T *b, T *c)
{
    const char transa = transa_for_layout(layout);
    const char transb = transb_for_layout(layout);
#if defined(BENCH_BACKEND_OPENBLAS) || defined(BENCH_BACKEND_ARMPL)
    const CBLAS_TRANSPOSE cblas_transa = transa == 'N' ? CblasNoTrans : CblasTrans;
    const CBLAS_TRANSPOSE cblas_transb = transb == 'N' ? CblasNoTrans : CblasTrans;
    if constexpr (std::is_same_v<T, double>)
    {
        cblas_dgemm(CblasRowMajor, cblas_transa, cblas_transb, size, size, size, 1.0, a, size, b, size, 0.0, c, size);
    }
    else
    {
        cblas_sgemm(CblasRowMajor, cblas_transa, cblas_transb, size, size, size, 1.0f, a, size, b, size, 0.0f, c, size);
    }
#elif defined(BENCH_BACKEND_LIBXSMM)
    const libxsmm_blasint s = static_cast<libxsmm_blasint>(size);
    const T alpha = static_cast<T>(1);
    const T beta = static_cast<T>(0);
    libxsmm_gemm(&transb, &transa, s, s, s, &alpha, b, &s, a, &s, &beta, c, &s);
#endif
}

void maybe_init_backend()
{
#if defined(BENCH_BACKEND_OPENBLAS)
    openblas_set_num_threads(1);
#endif
#if defined(BENCH_BACKEND_LIBXSMM)
    libxsmm_init();
#endif
}

void maybe_finalize_backend()
{
#if defined(BENCH_BACKEND_LIBXSMM)
    libxsmm_finalize();
#endif
}

template <typename T>
void run_case(
    std::ofstream &csv, int size, int batch, int warmup, int iters, bool verify_results, const std::string &layout)
{
    std::vector<T> a(size * size * batch);
    std::vector<T> b(size * size * batch);
    std::vector<T> c(size * size * batch, static_cast<T>(0));
    fill_inputs(a, b);

    for (int i = 0; i < warmup; ++i)
    {
        for (int j = 0; j < batch; ++j)
        {
            kernel(size,
                   layout,
                   a.data() + static_cast<std::size_t>(j) * size * size,
                   b.data() + static_cast<std::size_t>(j) * size * size,
                   c.data() + static_cast<std::size_t>(j) * size * size);
        }
    }

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
    {
        for (int j = 0; j < batch; ++j)
        {
            kernel(size,
                   layout,
                   a.data() + static_cast<std::size_t>(j) * size * size,
                   b.data() + static_cast<std::size_t>(j) * size * size,
                   c.data() + static_cast<std::size_t>(j) * size * size);
        }
    }
    const auto stop = std::chrono::steady_clock::now();

    const double elapsed_ns =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());
    const double avg_ns = elapsed_ns / static_cast<double>(iters * batch);
    const double gflops =
        (2.0 * static_cast<double>(size) * static_cast<double>(size) * static_cast<double>(size)) / avg_ns;
    double verify_diff = 0.0;
    if (verify_results)
    {
        std::vector<T> c_ref(size * size * batch, static_cast<T>(0));
        compute_reference(a, b, c_ref, size, batch, transa_for_layout(layout), transb_for_layout(layout));
        verify_diff = max_abs_diff(c, c_ref);
        const double tolerance = std::is_same_v<T, double> ? 1e-9 : 5e-4;
        if (verify_diff > tolerance)
        {
            throw std::runtime_error(std::string("verification failed for backend ") + backend_name());
        }
    }

    csv << backend_name() << ',' << size << ',' << size << ',' << size << ',' << batch << ',' << warmup << ',' << iters
        << ',' << std::fixed << std::setprecision(3) << avg_ns << ',' << std::fixed << std::setprecision(6) << gflops
        << ',' << std::fixed << std::setprecision(6) << checksum(c) << ',' << "ok,"
        << backend_note(layout, dtype_name<T>(), verify_results, verify_diff) << '\n';
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options options = parse_args(argc, argv);
        std::ofstream csv(options.csv_output);
        if (!csv)
        {
            throw std::runtime_error("failed to open csv output: " + options.csv_output);
        }
        csv << "backend,m,n,k,batch,warmup,iters,avg_ns,gflops,checksum,status,note\n";
        maybe_init_backend();
        for (int size : options.sizes)
        {
            if (options.dtype == "fp64")
            {
                run_case<double>(
                    csv, size, options.batch, options.warmup, options.iters, options.verify_results, options.layout);
            }
            else
            {
                run_case<float>(
                    csv, size, options.batch, options.warmup, options.iters, options.verify_results, options.layout);
            }
        }
        maybe_finalize_backend();
        return 0;
    } catch (const std::exception &e)
    {
        std::cerr << backend_name() << "_gemm_bench failed: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
