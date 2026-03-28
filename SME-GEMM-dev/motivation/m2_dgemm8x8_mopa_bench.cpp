#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void enable_sme_context_unchange(char *);
extern "C" void disable_sme_context_unchange(char *);

namespace
{

constexpr int kM = 16;
constexpr int kN = 16;

#define MOPA_16X16_4ZA_4MOPA                                                                                           \
    "fmopa za0.d, p0/m, p0/m, z0.d, z2.d\n\t"                                                                          \
    "fmopa za1.d, p0/m, p0/m, z0.d, z3.d\n\t"                                                                          \
    "fmopa za2.d, p0/m, p0/m, z1.d, z2.d\n\t"                                                                          \
    "fmopa za3.d, p0/m, p0/m, z1.d, z3.d\n\t"

#define MOVA_STORE_ROW(ZA, IDX, PTR)                                                                                   \
    "mov w12, #" #IDX "\n\t"                                                                                           \
    "mova z16.d, p7/m, za" #ZA "v.d[w12, 0]\n\t"                                                                       \
    "st1d z16.d, p7, [" #PTR "]\n\t"                                                                                   \
    "add " #PTR ", " #PTR ", #128\n\t"

#define MOVA_STORE_TILE(ZA, PTR)                                                                                       \
    MOVA_STORE_ROW(ZA, 0, PTR)                                                                                         \
    MOVA_STORE_ROW(ZA, 1, PTR)                                                                                         \
    MOVA_STORE_ROW(ZA, 2, PTR)                                                                                         \
    MOVA_STORE_ROW(ZA, 3, PTR)                                                                                         \
    MOVA_STORE_ROW(ZA, 4, PTR) MOVA_STORE_ROW(ZA, 5, PTR) MOVA_STORE_ROW(ZA, 6, PTR) MOVA_STORE_ROW(ZA, 7, PTR)

struct Config
{
    std::vector<int> k_values{8, 16, 32, 64, 128, 256, 512};
    int iters = 20000;
    int warmup = 2000;
    int repeats = 5;
    std::string csv_path;
};

struct ResultRow
{
    int K = 0;
    double full_ns = 0.0;
    double compute_ns = 0.0;
    double mova_only_ns = 0.0;
    double gflops_full = 0.0;
    double gflops_compute = 0.0;
    double mova_ratio = 0.0;
};

double now_ns()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::nano>(clock::now().time_since_epoch()).count();
}

std::vector<int> parse_k_list(const std::string &text)
{
    std::vector<int> ks;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        if (token.empty())
        {
            continue;
        }
        const int k = std::atoi(token.c_str());
        if (k > 0)
        {
            ks.push_back(k);
        }
    }
    if (ks.empty())
    {
        throw std::runtime_error("k-list is empty");
    }
    return ks;
}

void init_inputs(std::vector<double> &a_col_major, std::vector<double> &b_row_major, int K)
{
    a_col_major.resize(kM * K);
    b_row_major.resize(K * kN);

    for (int k = 0; k < K; ++k)
    {
        for (int i = 0; i < kM; ++i)
        {
            a_col_major[i + k * kM] = static_cast<double>((i + 3 * k) % 17) / 17.0;
        }
        for (int j = 0; j < kN; ++j)
        {
            b_row_major[k * kN + j] = static_cast<double>((j + 5 * k) % 19) / 19.0;
        }
    }
}
__attribute__((noinline)) __attribute__((target("+sve+sme+sme-f64f64"))) void
run_compute_only_once(const double *a_col_major, const double *b_row_major, int K)
{
    asm volatile("mov x10, %[a]\n\t"
                 "mov x11, %[b]\n\t"
                 "mov w12, %w[k]\n\t"
                 "ptrue p0.d\n\t"
                 "cbz w12, 2f\n\t"
                 "1:\n\t"
                 "ld1d z0.d, p0/z, [x10]\n\t"
                 "ld1d z1.d, p0/z, [x10, #1, mul vl]\n\t"
                 "ld1d z2.d, p0/z, [x11]\n\t"
                 "ld1d z3.d, p0/z, [x11, #1, mul vl]\n\t" MOPA_16X16_4ZA_4MOPA "add x10, x10, #128\n\t"
                 "add x11, x11, #128\n\t"
                 "subs w12, w12, #1\n\t"
                 "b.ne 1b\n\t"
                 "2:\n\t"
                 :
                 : [a] "r"(a_col_major), [b] "r"(b_row_major), [k] "r"(K)
                 : "x10", "x11", "w12", "p0", "z0", "z1", "z2", "z3", "za", "memory");
}
__attribute__((noinline)) __attribute__((target("+sve+sme"))) void run_mova_only_once(double *c_row_major)
{
    asm volatile("mov x20, %[c]\n\t"
                 "add x21, x20, #64\n\t"
                 "add x22, x20, #1024\n\t"
                 "add x23, x22, #64\n\t"
                 "ptrue p7.d\n\t" MOVA_STORE_TILE(0, x20) MOVA_STORE_TILE(1, x21) MOVA_STORE_TILE(2, x22)
                     MOVA_STORE_TILE(3, x23)
                 :
                 : [c] "r"(c_row_major)
                 : "x20", "x21", "x22", "x23", "w12", "p7", "z16", "za", "memory");
}

__attribute__((noinline)) __attribute__((target("+sve+sme+sme-f64f64"))) void
run_full_once(const double *a_col_major, const double *b_row_major, double *c_row_major, int K)
{
    asm volatile(
        "mov x10, %[a]\n\t"
        "mov x11, %[b]\n\t"
        "mov x20, %[c]\n\t"
        "add x21, x20, #64\n\t"
        "add x22, x20, #1024\n\t"
        "add x23, x22, #64\n\t"
        "mov w12, %w[k]\n\t"
        "ptrue p0.d\n\t"
        "cbz w12, 2f\n\t"
        "1:\n\t"
        "ld1d z0.d, p0/z, [x10]\n\t"
        "ld1d z1.d, p0/z, [x10, #1, mul vl]\n\t"
        "ld1d z2.d, p0/z, [x11]\n\t"
        "ld1d z3.d, p0/z, [x11, #1, mul vl]\n\t" MOPA_16X16_4ZA_4MOPA "add x10, x10, #128\n\t"
        "add x11, x11, #128\n\t"
        "subs w12, w12, #1\n\t"
        "b.ne 1b\n\t"
        "2:\n\t"
        "ptrue p7.d\n\t" MOVA_STORE_TILE(0, x20) MOVA_STORE_TILE(1, x21) MOVA_STORE_TILE(2, x22) MOVA_STORE_TILE(3, x23)
        :
        : [a] "r"(a_col_major), [b] "r"(b_row_major), [c] "r"(c_row_major), [k] "r"(K)
        : "x10", "x11", "x20", "x21", "x22", "x23", "w12", "p0", "p7", "z0", "z1", "z2", "z3", "z16", "za", "memory");
}

template <typename Fn>
double bench_median_ns(const Config &cfg, Fn &&fn)
{
    double samples[32] = {};
    const int reps = std::clamp(cfg.repeats, 1, 32);

    fn(cfg.warmup);
    for (int i = 0; i < reps; ++i)
    {
        const double st = now_ns();
        fn(cfg.iters);
        const double ed = now_ns();
        samples[i] = ed - st;
    }

    std::sort(samples, samples + reps);
    return samples[reps / 2];
}

ResultRow measure_one_k(const Config &cfg, int K)
{
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> c(kM * kN, 0.0);

    init_inputs(a, b, K);

    char sme_ctx[512] = {0};
    enable_sme_context_unchange(sme_ctx);

    const double full_ns = bench_median_ns(cfg, [&](int loop_count) {
        for (int it = 0; it < loop_count; ++it)
        {
            run_full_once(a.data(), b.data(), c.data(), K);
        }
    });

    const double compute_ns = bench_median_ns(cfg, [&](int loop_count) {
        for (int it = 0; it < loop_count; ++it)
        {
            run_compute_only_once(a.data(), b.data(), K);
        }
    });

    const double mova_only_ns = bench_median_ns(cfg, [&](int loop_count) {
        for (int it = 0; it < loop_count; ++it)
        {
            run_mova_only_once(c.data());
        }
    });

    disable_sme_context_unchange(sme_ctx);

    const double full_sec = full_ns * 1e-9;
    const double compute_sec = compute_ns * 1e-9;
    const double flop_total = static_cast<double>(cfg.iters) * 2.0 * static_cast<double>(kM) * static_cast<double>(kN) *
                              static_cast<double>(K);

    ResultRow row;
    row.K = K;
    row.full_ns = full_ns;
    row.compute_ns = compute_ns;
    row.mova_only_ns = mova_only_ns;
    row.gflops_full = full_sec > 0.0 ? (flop_total / full_sec / 1e9) : 0.0;
    row.gflops_compute = compute_sec > 0.0 ? (flop_total / compute_sec / 1e9) : 0.0;
    row.mova_ratio = full_ns > 0.0 ? (std::max(0.0, full_ns - compute_ns) / full_ns) : 0.0;
    return row;
}

void emit_csv(const std::string &path, const std::vector<ResultRow> &rows, const Config &cfg)
{
    if (path.empty())
    {
        return;
    }

    std::ofstream out(path, std::ios::trunc);
    if (!out)
    {
        throw std::runtime_error("failed to open csv output: " + path);
    }

    out << "M,N,K,iters,repeats,full_ns,compute_ns,mova_only_ns,gflops_full,gflops_compute,mova_ratio\n";
    for (const auto &r : rows)
    {
        out << kM << "," << kN << "," << r.K << "," << cfg.iters << "," << cfg.repeats << "," << r.full_ns << ","
            << r.compute_ns << "," << r.mova_only_ns << "," << r.gflops_full << "," << r.gflops_compute << ","
            << r.mova_ratio << "\n";
    }
}

void print_usage(const char *argv0)
{
    std::cout << "Usage: " << argv0 << " [--k-list 8,16,32,...] [--iters N] [--warmup N] [--repeats N] [--csv path]\n";
}
__attribute__((noinline)) __attribute__((target("+nosve+nosme"))) Config parse_args(int argc, char **argv)
{
    Config cfg;

    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--k-list") == 0 && i + 1 < argc)
        {
            cfg.k_values = parse_k_list(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
        {
            cfg.iters = std::max(1, std::atoi(argv[++i]));
            continue;
        }
        if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
        {
            cfg.warmup = std::max(1, std::atoi(argv[++i]));
            continue;
        }
        if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc)
        {
            cfg.repeats = std::max(1, std::atoi(argv[++i]));
            continue;
        }
        if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc)
        {
            cfg.csv_path = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0)
        {
            print_usage(argv[0]);
            std::exit(0);
        }

        throw std::runtime_error(std::string("unknown arg: ") + argv[i]);
    }

    return cfg;
}

} // namespace

__attribute__((noinline)) __attribute__((target("+nosve+nosme"))) int main(int argc, char **argv)
{
    try
    {
        const Config cfg = parse_args(argc, argv);

        std::vector<ResultRow> rows;
        rows.reserve(cfg.k_values.size());

        std::cout << "M=16 N=16 layout: A col-major, B row-major, C row-major\n";
        std::cout << std::left << std::setw(8) << "K" << std::setw(16) << "GFLOPS(full)" << std::setw(18)
                  << "GFLOPS(compute)" << std::setw(16) << "mova_ratio" << "\n";

        for (int K : cfg.k_values)
        {
            const ResultRow r = measure_one_k(cfg, K);
            rows.push_back(r);

            std::cout << std::left << std::setw(8) << r.K << std::setw(16) << r.gflops_full << std::setw(18)
                      << r.gflops_compute << std::setw(16) << r.mova_ratio << "\n";
        }

        emit_csv(cfg.csv_path, rows, cfg);
        if (!cfg.csv_path.empty())
        {
            std::cout << "csv=" << cfg.csv_path << "\n";
        }
    } catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
