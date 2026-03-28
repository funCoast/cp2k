#include "interface.h"

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

namespace
{

#if defined(__clang__)
#define SMELT_BENCH_NOINLINE __attribute__((noinline, optnone))
#else
#define SMELT_BENCH_NOINLINE __attribute__((noinline))
#endif

struct Options
{
    std::vector<int> sizes;
    int batch = 8;
    int warmup = 10;
    int iters = 1000;
    bool verify_results = false;
    std::string csv_output;
    std::string strategy = "AUTO";
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
        else if (arg == "--strategy")
        {
            options.strategy = need_value("--strategy");
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

SMELT::Strategy parse_strategy(const std::string &text)
{
    if (text == "AUTO")
    {
        return SMELT::Strategy::AUTO;
    }
    if (text == "COSTMODEL")
    {
        return SMELT::Strategy::COSTMODEL;
    }
    if (text == "SCALAR")
    {
        return SMELT::Strategy::SCALAR;
    }
    if (text == "SVE")
    {
        return SMELT::Strategy::SVE;
    }
    if (text == "FUSE_SVE")
    {
        return SMELT::Strategy::FUSE_SVE;
    }
    if (text == "MOPA")
    {
        return SMELT::Strategy::MOPA;
    }
    if (text == "FUSE_MOPA")
    {
        return SMELT::Strategy::FUSE_MOPA;
    }
    if (text == "STRATEGY1")
    {
        return SMELT::Strategy::STRATEGY1;
    }
    if (text == "SME2")
    {
        return SMELT::Strategy::SME2;
    }
    if (text == "FUSE_SME2")
    {
        return SMELT::Strategy::FUSE_SME2;
    }
    throw std::invalid_argument("unsupported strategy: " + text);
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
T load_a(const T *a_batch, int i, int inner, int m, int k, char transa)
{
    if (transa == 'N')
    {
        return a_batch[static_cast<std::size_t>(i) * k + inner];
    }
    return a_batch[static_cast<std::size_t>(inner) * m + i];
}

template <typename T>
T load_b(const T *b_batch, int inner, int j, int k, int n, char transb)
{
    if (transb == 'N')
    {
        return b_batch[static_cast<std::size_t>(inner) * n + j];
    }
    return b_batch[static_cast<std::size_t>(j) * k + inner];
}

template <typename T>
void compute_reference(const std::vector<T> &a,
                       const std::vector<T> &b,
                       std::vector<T> &c_ref,
                       int m,
                       int n,
                       int k,
                       int batch,
                       char transa,
                       char transb)
{
    std::fill(c_ref.begin(), c_ref.end(), static_cast<T>(0));
    for (int batch_idx = 0; batch_idx < batch; ++batch_idx)
    {
        const T *a_batch = a.data() + static_cast<std::size_t>(batch_idx) * m * k;
        const T *b_batch = b.data() + static_cast<std::size_t>(batch_idx) * k * n;
        T *c_batch = c_ref.data() + static_cast<std::size_t>(batch_idx) * m * n;
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                T sum = static_cast<T>(0);
                for (int inner = 0; inner < k; ++inner)
                {
                    sum += load_a(a_batch, i, inner, m, k, transa) * load_b(b_batch, inner, j, k, n, transb);
                }
                c_batch[static_cast<std::size_t>(i) * n + j] = sum;
            }
        }
    }
}

void enter_manual_sme()
{
#if defined(__aarch64__)
    asm volatile("smstart");
#endif
}

void exit_manual_sme()
{
#if defined(__aarch64__)
    asm volatile("smstop");
#endif
}

template <typename T>
SMELT_BENCH_NOINLINE void run_manual_kernel_loop(
    typename std::conditional_t<std::is_same_v<T, double>, SMELT::DgemmBatchKernelPtr, SMELT::SgemmBatchKernelPtr>
        kernel,
    int m,
    int n,
    int k,
    int batch,
    int iters,
    const T *const *a_ptrs,
    const T *const *b_ptrs,
    T *const *c_ptrs)
{
    enter_manual_sme();
    for (int i = 0; i < iters; ++i)
    {
        kernel(batch, a_ptrs, b_ptrs, c_ptrs, m, n, k);
    }
    exit_manual_sme();
}

template <typename T>
void run_case(std::ofstream &csv,
              int size,
              int batch,
              int warmup,
              int iters,
              bool verify_results,
              SMELT::Strategy strategy,
              const std::string &strategy_name,
              const std::string &layout)
{
    const int m = size;
    const int n = size;
    const int k = size;
    const char transa = transa_for_layout(layout);
    const char transb = transb_for_layout(layout);

    std::vector<T> a(m * k * batch);
    std::vector<T> b(k * n * batch);
    std::vector<T> c(m * n * batch, static_cast<T>(0));
    std::vector<const T *> a_ptrs(batch);
    std::vector<const T *> b_ptrs(batch);
    std::vector<T *> c_ptrs(batch);
    for (int i = 0; i < batch; ++i)
    {
        a_ptrs[i] = a.data() + static_cast<std::size_t>(i) * m * k;
        b_ptrs[i] = b.data() + static_cast<std::size_t>(i) * k * n;
        c_ptrs[i] = c.data() + static_cast<std::size_t>(i) * m * n;
    }
    fill_inputs(a, b);

    SMELT::clear_jit_cache();
    SMELT::set_cache_enabled(true);
    SMELT::set_strategy(strategy);
    SMELT::set_auto_context_switch(true);

    if constexpr (std::is_same_v<T, double>)
    {
        auto kernel = SMELT::get_dgemm_batch_kernel_ptr(transa, transb, m, n, k, strategy);

        SMELT::set_auto_context_switch(false);
        run_manual_kernel_loop<T>(kernel, m, n, k, batch, warmup, a_ptrs.data(), b_ptrs.data(), c_ptrs.data());

        const auto start = std::chrono::steady_clock::now();
        run_manual_kernel_loop<T>(kernel, m, n, k, batch, iters, a_ptrs.data(), b_ptrs.data(), c_ptrs.data());
        const auto stop = std::chrono::steady_clock::now();

        const double elapsed_ns =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());
        const double avg_ns = elapsed_ns / static_cast<double>(iters * batch);
        const double gflops = (2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k)) / avg_ns;
        double verify_diff = 0.0;
        if (verify_results)
        {
            std::vector<T> c_ref(m * n * batch, static_cast<T>(0));
            compute_reference(a, b, c_ref, m, n, k, batch, transa, transb);
            verify_diff = max_abs_diff(c, c_ref);
            const double tolerance = std::is_same_v<T, double> ? 1e-9 : 5e-4;
            if (verify_diff > tolerance)
            {
                throw std::runtime_error("verification failed for SMELT row-major benchmark");
            }
        }

        csv << "smelt," << m << ',' << n << ',' << k << ',' << batch << ',' << warmup << ',' << iters << ','
            << std::fixed << std::setprecision(3) << avg_ns << ',' << std::fixed << std::setprecision(6) << gflops
            << ',' << std::fixed << std::setprecision(6) << checksum(c) << ',' << "ok,"
            << "strategy=" << strategy_name << ";dtype=" << dtype_name<T>()
            << ";layout=rowmajor;trans=" << layout_note_name(layout)
            << ";auto_context_switch=off;manual_context_scope=outer_loop;interface_test_style=1;benchmark_opt=-O1"
            << ";verify=" << (verify_results ? "on" : "off");
        if (verify_results)
        {
            csv << ";max_abs_diff=" << std::scientific << std::setprecision(3) << verify_diff;
        }
        csv << '\n';
        return;
    }
    else
    {
        auto kernel = SMELT::get_sgemm_batch_kernel_ptr(transa, transb, m, n, k, strategy);
        kernel(batch, a_ptrs.data(), b_ptrs.data(), c_ptrs.data(), m, n, k);

        SMELT::set_auto_context_switch(false);
        run_manual_kernel_loop<T>(kernel, m, n, k, batch, warmup, a_ptrs.data(), b_ptrs.data(), c_ptrs.data());

        const auto start = std::chrono::steady_clock::now();
        run_manual_kernel_loop<T>(kernel, m, n, k, batch, iters, a_ptrs.data(), b_ptrs.data(), c_ptrs.data());
        const auto stop = std::chrono::steady_clock::now();

        const double elapsed_ns =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());
        const double avg_ns = elapsed_ns / static_cast<double>(iters * batch);
        const double gflops = (2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k)) / avg_ns;
        double verify_diff = 0.0;
        if (verify_results)
        {
            std::vector<T> c_ref(m * n * batch, static_cast<T>(0));
            compute_reference(a, b, c_ref, m, n, k, batch, transa, transb);
            verify_diff = max_abs_diff(c, c_ref);
            const double tolerance = std::is_same_v<T, double> ? 1e-9 : 5e-4;
            if (verify_diff > tolerance)
            {
                throw std::runtime_error("verification failed for SMELT row-major benchmark");
            }
        }

        csv << "smelt," << m << ',' << n << ',' << k << ',' << batch << ',' << warmup << ',' << iters << ','
            << std::fixed << std::setprecision(3) << avg_ns << ',' << std::fixed << std::setprecision(6) << gflops
            << ',' << std::fixed << std::setprecision(6) << checksum(c) << ',' << "ok,"
            << "strategy=" << strategy_name << ";dtype=" << dtype_name<T>()
            << ";layout=rowmajor;trans=" << layout_note_name(layout)
            << ";auto_context_switch=off;manual_context_scope=outer_loop;interface_test_style=1;benchmark_opt=-O1"
            << ";verify=" << (verify_results ? "on" : "off");
        if (verify_results)
        {
            csv << ";max_abs_diff=" << std::scientific << std::setprecision(3) << verify_diff;
        }
        csv << '\n';
        return;
    }
}

#undef SMELT_BENCH_NOINLINE

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options options = parse_args(argc, argv);
        const SMELT::Strategy strategy = parse_strategy(options.strategy);

        std::ofstream csv(options.csv_output);
        if (!csv)
        {
            throw std::runtime_error("failed to open csv output: " + options.csv_output);
        }
        csv << "backend,m,n,k,batch,warmup,iters,avg_ns,gflops,checksum,status,note\n";
        for (int size : options.sizes)
        {
            if (options.dtype == "fp64")
            {
                run_case<double>(csv,
                                 size,
                                 options.batch,
                                 options.warmup,
                                 options.iters,
                                 options.verify_results,
                                 strategy,
                                 options.strategy,
                                 options.layout);
            }
            else
            {
                run_case<float>(csv,
                                size,
                                options.batch,
                                options.warmup,
                                options.iters,
                                options.verify_results,
                                strategy,
                                options.strategy,
                                options.layout);
            }
        }
        return 0;
    } catch (const std::exception &e)
    {
        std::cerr << "smelt_gemm_bench failed: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
