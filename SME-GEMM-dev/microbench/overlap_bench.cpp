#include <chrono>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <arm_sve.h>
#include <arm_sme.h>
#if defined(__linux__) && defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

extern "C" void enable_sme(char *);
extern "C" void disable_sme(char *);
extern "C" void enable_sme_context_unchange(char *);
extern "C" void disable_sme_context_unchange(char *);

alignas(64) static double g_mem_buf[64] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0,
                                           14.0, 15.0, 16.0, 1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5,  10.5,
                                           11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,
                                           7.5,  8.5,  9.5,  10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 2.0,  3.0,  4.0,  5.0,
                                           6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0};

static inline double now_ns()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::nano>(clock::now().time_since_epoch()).count();
}

static bool host_supports_sme2()
{
    // Allow environment variable override for systems where hwcap2 is unavailable
    const char *forced_sme2 = std::getenv("FORCE_SME2");
    if (forced_sme2 != nullptr && std::atoi(forced_sme2) != 0)
    {
        return true;
    }

#if defined(__APPLE__)
    return true;
#else
    return false;
#endif
}

__attribute__((noinline)) static void sve_1(int iters)
{
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p0.d\n\t"
                     "fmla z0.d, p0/m, z1.d, z2.d\n\t"
                     :
                     :
                     : "p0", "z0", "z1", "z2", "memory");
    }
}

__attribute__((noinline)) static void sve_2(int iters)
{
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p0.d\n\t"
                     "fmla z0.d, p0/m, z1.d, z2.d\n\t"
                     "fmla z3.d, p0/m, z4.d, z5.d\n\t"
                     :
                     :
                     : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "memory");
    }
}

__attribute__((noinline)) static void sve_4(int iters)
{
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p0.d\n\t"
                     "fmla z0.d, p0/m, z1.d, z2.d\n\t"
                     "fmla z3.d, p0/m, z4.d, z5.d\n\t"
                     "fmla z6.d, p0/m, z7.d, z8.d\n\t"
                     "fmla z9.d, p0/m, z10.d, z11.d\n\t"
                     :
                     :
                     : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "memory");
    }
}

__attribute__((noinline)) static void sve_8(int iters)
{
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p0.d\n\t"
                     "fmla z0.d, p0/m, z1.d, z2.d\n\t"
                     "fmla z3.d, p0/m, z4.d, z5.d\n\t"
                     "fmla z6.d, p0/m, z7.d, z8.d\n\t"
                     "fmla z9.d, p0/m, z10.d, z11.d\n\t"
                     "fmla z12.d, p0/m, z13.d, z14.d\n\t"
                     "fmla z15.d, p0/m, z16.d, z17.d\n\t"
                     "fmla z18.d, p0/m, z19.d, z20.d\n\t"
                     "fmla z21.d, p0/m, z22.d, z23.d\n\t"
                     :
                     :
                     : "p0",
                       "z0",
                       "z1",
                       "z2",
                       "z3",
                       "z4",
                       "z5",
                       "z6",
                       "z7",
                       "z8",
                       "z9",
                       "z10",
                       "z11",
                       "z12",
                       "z13",
                       "z14",
                       "z15",
                       "z16",
                       "z17",
                       "z18",
                       "z19",
                       "z20",
                       "z21",
                       "z22",
                       "z23",
                       "memory");
    }
}

__attribute__((noinline)) static void mopa_1(int iters)
{
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p0.d\n\t"
                     "fmopa za0.d, p0/m, p0/m, z0.d, z1.d\n\t"
                     :
                     :
                     : "p0", "z0", "z1", "za", "memory");
    }
}

__attribute__((noinline)) static void mopa_2(int iters)
{
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p0.d\n\t"
                     "fmopa za0.d, p0/m, p0/m, z0.d, z1.d\n\t"
                     "fmopa za0.d, p0/m, p0/m, z2.d, z3.d\n\t"
                     :
                     :
                     : "p0", "z0", "z1", "z2", "z3", "za", "memory");
    }
}

__attribute__((noinline)) static void mopa_4(int iters)
{
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p0.d\n\t"
                     "fmopa za0.d, p0/m, p0/m, z0.d, z1.d\n\t"
                     "fmopa za0.d, p0/m, p0/m, z2.d, z3.d\n\t"
                     "fmopa za0.d, p0/m, p0/m, z4.d, z5.d\n\t"
                     "fmopa za0.d, p0/m, p0/m, z6.d, z7.d\n\t"
                     :
                     :
                     : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "za", "memory");
    }
}

__attribute__((noinline)) static void mix_mopa1_sve1(int iters)
{
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p0.d\n\t"
                     "fmopa za0.d, p0/m, p0/m, z0.d, z1.d\n\t"
                     "fmla z2.d, p0/m, z3.d, z4.d\n\t"
                     :
                     :
                     : "p0", "z0", "z1", "z2", "z3", "z4", "za", "memory");
    }
}

__attribute__((noinline)) static void mix_mopa1_sve2(int iters)
{
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p0.d\n\t"
                     "fmopa za0.d, p0/m, p0/m, z0.d, z1.d\n\t"
                     "fmla z2.d, p0/m, z3.d, z4.d\n\t"
                     "fmla z5.d, p0/m, z6.d, z7.d\n\t"
                     :
                     :
                     : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "za", "memory");
    }
}

__attribute__((noinline)) static void mix_mopa1_sve4(int iters)
{
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p0.d\n\t"
                     "fmopa za0.d, p0/m, p0/m, z0.d, z1.d\n\t"
                     "fmla z2.d, p0/m, z3.d, z4.d\n\t"
                     "fmla z5.d, p0/m, z6.d, z7.d\n\t"
                     "fmla z8.d, p0/m, z9.d, z10.d\n\t"
                     "fmla z11.d, p0/m, z12.d, z13.d\n\t"
                     :
                     :
                     : "p0",
                       "z0",
                       "z1",
                       "z2",
                       "z3",
                       "z4",
                       "z5",
                       "z6",
                       "z7",
                       "z8",
                       "z9",
                       "z10",
                       "z11",
                       "z12",
                       "z13",
                       "za",
                       "memory");
    }
}

__attribute__((noinline)) static void mix_mopa1_sve8(int iters)
{
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p0.d\n\t"
                     "fmopa za0.d, p0/m, p0/m, z0.d, z1.d\n\t"
                     "fmla z2.d, p0/m, z3.d, z4.d\n\t"
                     "fmla z5.d, p0/m, z6.d, z7.d\n\t"
                     "fmla z8.d, p0/m, z9.d, z10.d\n\t"
                     "fmla z11.d, p0/m, z12.d, z13.d\n\t"
                     "fmla z14.d, p0/m, z15.d, z16.d\n\t"
                     "fmla z17.d, p0/m, z18.d, z19.d\n\t"
                     "fmla z20.d, p0/m, z21.d, z22.d\n\t"
                     "fmla z23.d, p0/m, z24.d, z25.d\n\t"
                     :
                     :
                     : "p0",
                       "z0",
                       "z1",
                       "z2",
                       "z3",
                       "z4",
                       "z5",
                       "z6",
                       "z7",
                       "z8",
                       "z9",
                       "z10",
                       "z11",
                       "z12",
                       "z13",
                       "z14",
                       "z15",
                       "z16",
                       "z17",
                       "z18",
                       "z19",
                       "z20",
                       "z21",
                       "z22",
                       "z23",
                       "z24",
                       "z25",
                       "za",
                       "memory");
    }
}

#if defined(__ARM_FEATURE_SME2)
__attribute__((noinline)) static void sme2_mla1(int iters)
{
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p0.d\n\t"
                     "fmla za.d[w8, 0, vgx2], {z0.d - z1.d}, {z2.d - z3.d}\n\t"
                     :
                     :
                     : "p0", "z0", "z1", "z2", "z3", "za", "memory");
    }
}

__attribute__((noinline)) static void mix_mopa1_sme2_mla1(int iters)
{
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p0.d\n\t"
                     "fmopa za0.d, p0/m, p0/m, z0.d, z1.d\n\t"
                     "fmla za.d[w9, 0, vgx2], {z2.d - z3.d}, {z4.d - z5.d}\n\t"
                     :
                     :
                     : "p0", "z0", "z1", "z2", "z3", "z4", "z5", "za", "memory");
    }
}
#endif

__attribute__((noinline)) static void mova_write_1(int iters)
{
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p7.d\n\t"
                     "mov w12, #0\n\t"
                     "mova za0h.d[w12, 0], p7/m, z0.d\n\t"
                     :
                     :
                     : "p7", "w12", "z0", "za", "memory");
    }
}

__attribute__((noinline)) static void mova_read_1(int iters)
{
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p7.d\n\t"
                     "mov w12, #0\n\t"
                     "mova z0.d, p7/m, za0v.d[w12, 0]\n\t"
                     :
                     :
                     : "p7", "w12", "z0", "za", "memory");
    }
}

__attribute__((noinline)) static void ld1_1(int iters)
{
    svbool_t pg = svptrue_b64();
    double *ptr = g_mem_buf;
    for (int i = 0; i < iters; ++i)
    {
        svfloat64_t v = svld1(pg, ptr);
        asm volatile("" : : "w"(v) : "memory");
    }
}

__attribute__((noinline)) static void st1_1(int iters)
{
    svbool_t pg = svptrue_b64();
    svfloat64_t v = svdup_f64(3.0);
    double *ptr = g_mem_buf;
    for (int i = 0; i < iters; ++i)
    {
        svst1(pg, ptr, v);
    }
}

__attribute__((noinline)) static void sel_1(int iters)
{
    svbool_t pg = svptrue_b64();
    svfloat64_t a = svdup_f64(1.0);
    svfloat64_t b = svdup_f64(2.0);
    for (int i = 0; i < iters; ++i)
    {
        svfloat64_t v = svsel(pg, a, b);
        asm volatile("" : : "w"(v) : "memory");
    }
}

__attribute__((noinline)) static void dupm_1(int iters)
{
    svbool_t pg = svptrue_b64();
    svfloat64_t ori = svdup_f64(0.0);
    for (int i = 0; i < iters; ++i)
    {
        svfloat64_t v = svdup_f64_m(ori, pg, 1.25);
        asm volatile("" : : "w"(v) : "memory");
    }
}

__attribute__((noinline)) static void mix_mopa1_mova_write1(int iters)
{
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p0.d\n\t"
                     "fmopa za0.d, p0/m, p0/m, z0.d, z1.d\n\t"
                     "ptrue p7.d\n\t"
                     "mov w12, #0\n\t"
                     "mova za0h.d[w12, 0], p7/m, z2.d\n\t"
                     :
                     :
                     : "p0", "p7", "w12", "z0", "z1", "z2", "za", "memory");
    }
}

__attribute__((noinline)) static void mix_mopa1_mova_read1(int iters)
{
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p0.d\n\t"
                     "fmopa za0.d, p0/m, p0/m, z0.d, z1.d\n\t"
                     "ptrue p7.d\n\t"
                     "mov w12, #0\n\t"
                     "mova z2.d, p7/m, za0v.d[w12, 0]\n\t"
                     :
                     :
                     : "p0", "p7", "w12", "z0", "z1", "z2", "za", "memory");
    }
}

__attribute__((noinline)) static void mix_mopa1_ld1_1(int iters)
{
    svbool_t pg = svptrue_b64();
    double *ptr = g_mem_buf;
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p0.d\n\t"
                     "fmopa za0.d, p0/m, p0/m, z0.d, z1.d\n\t"
                     :
                     :
                     : "p0", "z0", "z1", "za", "memory");
        svfloat64_t v = svld1(pg, ptr);
        asm volatile("" : : "w"(v) : "memory");
    }
}

__attribute__((noinline)) static void mix_mopa1_st1_1(int iters)
{
    svbool_t pg = svptrue_b64();
    svfloat64_t v = svdup_f64(2.0);
    double *ptr = g_mem_buf;
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p0.d\n\t"
                     "fmopa za0.d, p0/m, p0/m, z0.d, z1.d\n\t"
                     :
                     :
                     : "p0", "z0", "z1", "za", "memory");
        svst1(pg, ptr, v);
    }
}

__attribute__((noinline)) static void mix_mopa1_sel1(int iters)
{
    svbool_t pg = svptrue_b64();
    svfloat64_t a = svdup_f64(1.0);
    svfloat64_t b = svdup_f64(2.0);
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p0.d\n\t"
                     "fmopa za0.d, p0/m, p0/m, z0.d, z1.d\n\t"
                     :
                     :
                     : "p0", "z0", "z1", "za", "memory");
        svfloat64_t v = svsel(pg, a, b);
        asm volatile("" : : "w"(v) : "memory");
    }
}

__attribute__((noinline)) static void mix_mopa1_dupm1(int iters)
{
    svbool_t pg = svptrue_b64();
    svfloat64_t ori = svdup_f64(0.0);
    for (int i = 0; i < iters; ++i)
    {
        asm volatile("ptrue p0.d\n\t"
                     "fmopa za0.d, p0/m, p0/m, z0.d, z1.d\n\t"
                     :
                     :
                     : "p0", "z0", "z1", "za", "memory");
        svfloat64_t v = svdup_f64_m(ori, pg, 1.75);
        asm volatile("" : : "w"(v) : "memory");
    }
}

template <typename Fn>
static double bench_ns(Fn fn, int iters, int warmup)
{
    char sme_ctx[512] = {0};
    // Keep FP/SVE state intact when toggling SME; plain enable/disable clears regs.
    enable_sme_context_unchange(sme_ctx);
    fn(warmup);

    const double st = now_ns();
    fn(iters);
    const double ed = now_ns();

    disable_sme_context_unchange(sme_ctx);
    return ed - st;
}

int main(int argc, char **argv)
{
    int iters = 2000000;
    if (argc > 1)
    {
        iters = std::max(1, std::atoi(argv[1]));
    }
    const int warmup = std::max(1000, iters / 50);

    std::cout << "iters=" << iters << "\n";

    const double t_sve1 = bench_ns(sve_1, iters, warmup);
    const double t_sve2 = bench_ns(sve_2, iters, warmup);
    const double t_sve4 = bench_ns(sve_4, iters, warmup);
    const double t_sve8 = bench_ns(sve_8, iters, warmup);

    const double t_mopa1 = bench_ns(mopa_1, iters, warmup);
    const double t_mopa2 = bench_ns(mopa_2, iters, warmup);
    const double t_mopa4 = bench_ns(mopa_4, iters, warmup);

    const double t_mix1 = bench_ns(mix_mopa1_sve1, iters, warmup);
    const double t_mix2 = bench_ns(mix_mopa1_sve2, iters, warmup);
    const double t_mix4 = bench_ns(mix_mopa1_sve4, iters, warmup);
    const double t_mix8 = bench_ns(mix_mopa1_sve8, iters, warmup);

    const bool sme2_enabled = host_supports_sme2();
    double t_sme2_mla1 = -1.0;
    double t_mix_mopa1_sme2_mla1 = -1.0;
#if defined(__ARM_FEATURE_SME2)
    if (sme2_enabled)
    {
        t_sme2_mla1 = bench_ns(sme2_mla1, iters, warmup);
        t_mix_mopa1_sme2_mla1 = bench_ns(mix_mopa1_sme2_mla1, iters, warmup);
    }
#endif

    const double t_mova_write1 = bench_ns(mova_write_1, iters, warmup);
    const double t_mova_read1 = bench_ns(mova_read_1, iters, warmup);
    const double t_ld1 = bench_ns(ld1_1, iters, warmup);
    const double t_st1 = bench_ns(st1_1, iters, warmup);
    const double t_sel1 = bench_ns(sel_1, iters, warmup);
    const double t_dupm1 = bench_ns(dupm_1, iters, warmup);

    const double t_mix_mopa1_mova_w1 = bench_ns(mix_mopa1_mova_write1, iters, warmup);
    const double t_mix_mopa1_mova_r1 = bench_ns(mix_mopa1_mova_read1, iters, warmup);
    const double t_mix_mopa1_ld1 = bench_ns(mix_mopa1_ld1_1, iters, warmup);
    const double t_mix_mopa1_st1 = bench_ns(mix_mopa1_st1_1, iters, warmup);
    const double t_mix_mopa1_sel1 = bench_ns(mix_mopa1_sel1, iters, warmup);
    const double t_mix_mopa1_dupm1 = bench_ns(mix_mopa1_dupm1, iters, warmup);

    auto covered_sve = [&](double t_sve_n, double t_mix_n) {
        if (t_sve_n <= 0.0)
        {
            return 0.0;
        }
        const double extra_due_to_sve = std::max(0.0, t_mix_n - t_mopa1);
        const double visible_sve_ratio = std::clamp(extra_due_to_sve / t_sve_n, 0.0, 1.0);
        return 1.0 - visible_sve_ratio;
    };

    const double hide_ratio_1 = covered_sve(t_sve1, t_mix1);
    const double hide_ratio_2 = covered_sve(t_sve2, t_mix2);
    const double hide_ratio_4 = covered_sve(t_sve4, t_mix4);
    const double hide_ratio_8 = covered_sve(t_sve8, t_mix8);

    auto covered_single = [&](double t_single, double t_mix) {
        if (t_single <= 0.0)
        {
            return 0.0;
        }
        const double extra = std::max(0.0, t_mix - t_mopa1);
        const double visible_ratio = std::clamp(extra / t_single, 0.0, 1.0);
        return 1.0 - visible_ratio;
    };

    const double hidden_mova_write_ratio = covered_single(t_mova_write1, t_mix_mopa1_mova_w1);
    const double hidden_mova_read_ratio = covered_single(t_mova_read1, t_mix_mopa1_mova_r1);
    const double hidden_ld1_ratio = covered_single(t_ld1, t_mix_mopa1_ld1);
    const double hidden_st1_ratio = covered_single(t_st1, t_mix_mopa1_st1);
    const double hidden_sel_ratio = covered_single(t_sel1, t_mix_mopa1_sel1);
    const double hidden_dupm_ratio = covered_single(t_dupm1, t_mix_mopa1_dupm1);
    const double hidden_sme2_mla_ratio =
        (t_sme2_mla1 > 0.0 && t_mix_mopa1_sme2_mla1 > 0.0) ? covered_single(t_sme2_mla1, t_mix_mopa1_sme2_mla1) : 0.0;

    std::cout << "sve1_ns=" << t_sve1 << "\n";
    std::cout << "sve2_ns=" << t_sve2 << "\n";
    std::cout << "sve4_ns=" << t_sve4 << "\n";
    std::cout << "sve8_ns=" << t_sve8 << "\n";
    std::cout << "mopa1_ns=" << t_mopa1 << "\n";
    std::cout << "mopa2_ns=" << t_mopa2 << "\n";
    std::cout << "mopa4_ns=" << t_mopa4 << "\n";
    std::cout << "mix_mopa1_sve1_ns=" << t_mix1 << "\n";
    std::cout << "mix_mopa1_sve2_ns=" << t_mix2 << "\n";
    std::cout << "mix_mopa1_sve4_ns=" << t_mix4 << "\n";
    std::cout << "mix_mopa1_sve8_ns=" << t_mix8 << "\n";
    std::cout << "sme2_enabled=" << (sme2_enabled ? 1 : 0) << "\n";
    if (t_sme2_mla1 > 0.0)
    {
        std::cout << "sme2_mla1_ns=" << t_sme2_mla1 << "\n";
        std::cout << "mix_mopa1_sme2_mla1_ns=" << t_mix_mopa1_sme2_mla1 << "\n";
    }
    std::cout << "mova_write1_ns=" << t_mova_write1 << "\n";
    std::cout << "mova_read1_ns=" << t_mova_read1 << "\n";
    std::cout << "ld1_ns=" << t_ld1 << "\n";
    std::cout << "st1_ns=" << t_st1 << "\n";
    std::cout << "sel1_ns=" << t_sel1 << "\n";
    std::cout << "dupm1_ns=" << t_dupm1 << "\n";
    std::cout << "mix_mopa1_mova_write1_ns=" << t_mix_mopa1_mova_w1 << "\n";
    std::cout << "mix_mopa1_mova_read1_ns=" << t_mix_mopa1_mova_r1 << "\n";
    std::cout << "mix_mopa1_ld1_ns=" << t_mix_mopa1_ld1 << "\n";
    std::cout << "mix_mopa1_st1_ns=" << t_mix_mopa1_st1 << "\n";
    std::cout << "mix_mopa1_sel1_ns=" << t_mix_mopa1_sel1 << "\n";
    std::cout << "mix_mopa1_dupm1_ns=" << t_mix_mopa1_dupm1 << "\n";
    std::cout << "hidden_sve_ratio_n1=" << hide_ratio_1 << "\n";
    std::cout << "hidden_sve_ratio_n2=" << hide_ratio_2 << "\n";
    std::cout << "hidden_sve_ratio_n4=" << hide_ratio_4 << "\n";
    std::cout << "hidden_sve_ratio_n8=" << hide_ratio_8 << "\n";
    std::cout << "hidden_mova_write_ratio=" << hidden_mova_write_ratio << "\n";
    std::cout << "hidden_mova_read_ratio=" << hidden_mova_read_ratio << "\n";
    std::cout << "hidden_ld1_ratio=" << hidden_ld1_ratio << "\n";
    std::cout << "hidden_st1_ratio=" << hidden_st1_ratio << "\n";
    std::cout << "hidden_sel_ratio=" << hidden_sel_ratio << "\n";
    std::cout << "hidden_dupm_ratio=" << hidden_dupm_ratio << "\n";
    if (t_sme2_mla1 > 0.0)
    {
        std::cout << "hidden_sme2_mla_ratio=" << hidden_sme2_mla_ratio << "\n";
    }

    const double mopa_speedup_2 = (t_mopa1 > 0.0) ? (2.0 * t_mopa1 / t_mopa2) : 0.0;
    const double mopa_speedup_4 = (t_mopa1 > 0.0) ? (4.0 * t_mopa1 / t_mopa4) : 0.0;
    std::cout << "mopa_chain_speedup_2=" << mopa_speedup_2 << "\n";
    std::cout << "mopa_chain_speedup_4=" << mopa_speedup_4 << "\n";
    return 0;
}
