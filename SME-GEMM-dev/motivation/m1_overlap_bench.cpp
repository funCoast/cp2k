#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

extern "C" void enable_sme_context_unchange(char *);
extern "C" void disable_sme_context_unchange(char *);

namespace
{

#define ASM_MOPA                                                                                                       \
    "fmopa za0.d, p0/m, p0/m, z0.d, z1.d\n\t"                                                                          \
    "fmopa za1.d, p0/m, p0/m, z0.d, z1.d\n\t"                                                                          \
    "fmopa za2.d, p0/m, p0/m, z0.d, z1.d\n\t"                                                                          \
    "fmopa za3.d, p0/m, p0/m, z0.d, z1.d\n\t"

#ifdef __APPLE__
#define ASM_SVE_MLA                                                                                                    \
    "fmla z2.d, p0/m, z3.d, z4.d\n\t"                                                                                  \
    "fmla z5.d, p0/m, z3.d, z4.d\n\t"                                                                                  \
    "//fmla z6.d, p0/m, z3.d, z4.d\n\t"                                                                                \
    "//fmla z7.d, p0/m, z3.d, z4.d\n\t"                                                                                \
    "//fmla z8.d, p0/m, z3.d, z4.d\n\t"                                                                                \
    "//fmla z9.d, p0/m, z3.d, z4.d\n\t"                                                                                \
    "//fmla z10.d, p0/m, z3.d, z4.d\n\t"                                                                               \
    "//fmla z11.d, p0/m, z3.d, z4.d\n\t"                                                                               \
    "//fmla z12.d, p0/m, z3.d, z4.d\n\t"
constexpr int kMlaCount = 1;
constexpr const char *arch = "M4 Pro";
#else
#define ASM_SVE_MLA                                                                                                    \
    "fmla z2.d, p0/m, z3.d, z4.d\n\t"                                                                                  \
    "fmla z5.d, p0/m, z3.d, z4.d\n\t"                                                                                  \
    "fmla z6.d, p0/m, z3.d, z4.d\n\t"                                                                                  \
    "fmla z7.d, p0/m, z3.d, z4.d\n\t"                                                                                  \
    "fmla z8.d, p0/m, z3.d, z4.d\n\t"                                                                                  \
    "fmla z9.d, p0/m, z3.d, z4.d\n\t"                                                                                  \
    "fmla z10.d, p0/m, z3.d, z4.d\n\t"                                                                                 \
    "fmla z11.d, p0/m, z3.d, z4.d\n\t"                                                                                 \
    "fmla z12.d, p0/m, z3.d, z4.d\n\t"

constexpr int kMlaCount = 9;
constexpr const char *arch = "72F";
#endif

#define ASM_ZA_MOVA                                                                                                    \
    "mova z2.d, p7/m, za7v.d[w12, 0]\n\t"                                                                              \
    "mova z3.d, p7/m, za7v.d[w12, 0]\n\t"                                                                              \
    "mova z4.d, p7/m, za7v.d[w12, 0]\n\t"                                                                              \
    "mova z5.d, p7/m, za7v.d[w12, 0]\n\t"

#define REP2(x) x x
#define REP4(x) REP2(x) REP2(x)

constexpr int kMovaCount = 4;
constexpr int kMopaCount = 4;

enum class TargetKind
{
    SveMla,
    ZaMova,
};

enum class RunMode
{
    MopaOnly,
    TargetOnly,
    Mixed,
};

struct Config
{
    TargetKind target = TargetKind::SveMla;
    int iters = 300000;
    int warmup = 6000;
    int repeats = 5;
    std::string csv_path;
    bool append_csv = false;
};

double now_ns()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::nano>(clock::now().time_since_epoch()).count();
}

const char *target_name(TargetKind target) { return target == TargetKind::SveMla ? "sve_mla" : "za_mova"; }

const char *mode_name(RunMode mode)
{
    switch (mode)
    {
    case RunMode::MopaOnly:
        return "mopa_only";
    case RunMode::TargetOnly:
        return "companion_only";
    case RunMode::Mixed:
        return "mixed";
    }
    return "unknown";
}

void run_kernel(const Config &cfg, RunMode mode, int iters)
{
    if (mode == RunMode::MopaOnly)
    {
        for (int it = 0; it < iters; ++it)
        {
            asm volatile("ptrue p0.d\n\t" ASM_MOPA : : : "p0", "z0", "z1", "za", "memory");
        }
        return;
    }

    if (mode == RunMode::TargetOnly)
    {
        if (cfg.target == TargetKind::SveMla)
        {
            for (int it = 0; it < iters; ++it)
            {
                asm volatile("ptrue p0.d\n\t" ASM_SVE_MLA
                             :
                             :
                             : "p0", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "memory");
            }
            return;
        }
        else if (cfg.target == TargetKind::ZaMova)
        {
            for (int it = 0; it < iters; ++it)
            {
                asm volatile("mov w12, #0\n\t"
                             "ptrue p7.d\n\t" ASM_ZA_MOVA
                             :
                             :
                             : "w12", "p7", "z2", "za", "memory");
            }
        }
        return;
    }

    if (cfg.target == TargetKind::SveMla)
    {
        for (int it = 0; it < iters; ++it)
        {
            asm volatile("ptrue p0.d\n\t" ASM_SVE_MLA ASM_MOPA
                         :
                         :
                         : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "za", "memory");
        }
        return;
    }
    else if (cfg.target == TargetKind::ZaMova)
    {
        for (int it = 0; it < iters; ++it)
        {
            asm volatile("mov w12, #0\n\t"
                         "ptrue p7.d\n\t" ASM_ZA_MOVA ASM_MOPA
                         :
                         :
                         : "w12", "p0", "p7", "z0", "z1", "z2", "za", "memory");
        }
    }
}

double bench_once_ns(const Config &cfg, RunMode mode)
{
    char sme_ctx[512] = {0};
    enable_sme_context_unchange(sme_ctx);
    run_kernel(cfg, mode, cfg.warmup);

    const double st = now_ns();
    run_kernel(cfg, mode, cfg.iters);
    const double ed = now_ns();

    disable_sme_context_unchange(sme_ctx);
    return ed - st;
}

double bench_median_ns(const Config &cfg, RunMode mode)
{
    double best[32] = {};
    const int reps = std::clamp(cfg.repeats, 1, 32);
    for (int i = 0; i < reps; ++i)
    {
        best[i] = bench_once_ns(cfg, mode);
    }
    std::sort(best, best + reps);
    return best[reps / 2];
}

bool parse_target(const char *arg, TargetKind &target)
{
    if (std::strcmp(arg, "sve_mla") == 0)
    {
        target = TargetKind::SveMla;
        return true;
    }
    if (std::strcmp(arg, "za_mova") == 0)
    {
        target = TargetKind::ZaMova;
        return true;
    }
    return false;
}

void print_usage(const char *argv0)
{
    std::cout << "Usage: " << argv0 << " [--target sve_mla|za_mova] [--companion sve_mla|za_mova]"
              << " [--iters N] [--warmup N] [--repeats N]"
              << " [--csv path] [--append]\n";
}

bool parse_args(int argc, char **argv, Config &cfg)
{
    for (int i = 1; i < argc; ++i)
    {
        if ((std::strcmp(argv[i], "--target") == 0 || std::strcmp(argv[i], "--companion") == 0) && i + 1 < argc)
        {
            if (!parse_target(argv[++i], cfg.target))
            {
                std::cerr << "Invalid --target/--companion value\n";
                return false;
            }
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
        if (std::strcmp(argv[i], "--append") == 0)
        {
            cfg.append_csv = true;
            continue;
        }
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0)
        {
            print_usage(argv[0]);
            std::exit(0);
        }

        std::cerr << "Unknown arg: " << argv[i] << "\n";
        return false;
    }
    return true;
}

void emit_csv(const Config &cfg, double t_mopa_only, double t_target_only, double t_mixed)
{
    if (cfg.csv_path.empty())
    {
        return;
    }

    const bool file_exists = std::ifstream(cfg.csv_path).good();
    const bool write_header = !(cfg.append_csv && file_exists);

    std::ofstream out(cfg.csv_path, cfg.append_csv ? std::ios::app : std::ios::trunc);
    if (!out)
    {
        std::cerr << "Failed to open csv output: " << cfg.csv_path << "\n";
        return;
    }

    if (write_header)
    {
        out << "companion,mopa_count,companion_count,iters,repeats,mode,time_ns,ns_per_iter\n";
    }

    const auto write_row = [&](const char *mode, double ns_total) {
        out << target_name(cfg.target) << "," << kMopaCount << ","
            << (cfg.target == TargetKind::ZaMova ? kMovaCount : kMlaCount) << "," << cfg.iters << "," << cfg.repeats
            << "," << mode << "," << ns_total << "," << (ns_total / cfg.iters) << "\n";
    };

    write_row(mode_name(RunMode::MopaOnly), t_mopa_only);
    write_row(mode_name(RunMode::TargetOnly), t_target_only);
    write_row(mode_name(RunMode::Mixed), t_mixed);
}

} // namespace

int main(int argc, char **argv)
{
    Config cfg;
    if (!parse_args(argc, argv, cfg))
    {
        print_usage(argv[0]);
        return 1;
    }

    const double t_mopa_only = bench_median_ns(cfg, RunMode::MopaOnly);
    const double t_target_only = bench_median_ns(cfg, RunMode::TargetOnly);
    const double t_mixed = bench_median_ns(cfg, RunMode::Mixed);

    std::cout << "companion=" << target_name(cfg.target) << "\n";
    std::cout << "mopa_count=" << kMopaCount << "\n";
    std::cout << "companion_count=" << (cfg.target == TargetKind::ZaMova ? kMovaCount : kMlaCount) << "\n";
    std::cout << "iters=" << cfg.iters << "\n";
    std::cout << "repeats=" << cfg.repeats << "\n";
    std::cout << "mopa_only_ns=" << t_mopa_only << "\n";
    std::cout << "companion_only_ns=" << t_target_only << "\n";
    std::cout << "mixed_ns=" << t_mixed << "\n";

    emit_csv(cfg, t_mopa_only, t_target_only, t_mixed);
    return 0;
}

#undef MIXED_MOVA_SEQ
#undef MIXED_SVE_SEQ
#undef TARGET_MOVA_SEQ
#undef TARGET_SVE_SEQ
#undef MOPA_SEQ
#undef REP4
#undef REP2
#undef ASM_ZA_MOVA
#undef ASM_SVE_MLA
#undef ASM_MOPA
