#include <stdint.h>
#include <arm_sme.h>
#include <arm_sve.h>
#include <iostream>
#include <chrono>
#include <random>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <stdio.h>
#include <stdint.h>

#include <time.h>
#include "macro.h"

#if !defined(__clang__) && defined(__APPLE__)
extern "C" uintptr_t _arm_tpidr2_save(void) __attribute__((weak))
{
    uintptr_t val;
    asm("mrs %0, tpidr2_el0" : "=r"(val));
    return val;
}
#endif

#define TEST_GROUP(...) FOR_EACH(TEST_GEMM, __VA_ARGS__)

template <bool Cond, typename T>
__attribute__((always_inline)) inline T &select(T &a, T &b)
{
    if constexpr (Cond)
    {
        return a;
    }
    else
    {
        return b;
    }
}
consteval int za_fmla_d(int idx) { return idx % 8; }
template <int I, typename... Ts>
constexpr decltype(auto) selectI(Ts &&...args)
{
    return std::get<I>(std::forward_as_tuple(std::forward<Ts>(args)...));
}

static inline double now_sec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void print_batch_matrix(const char *name, int64_t batch_id, const double *C, int64_t M, int64_t N)
{
    printf("==== Batch %lld ====\n", batch_id);
    printf("%s[%lld] (%lld x %lld):\n", name, batch_id, M, N);

    /* 打印列索引 */
    printf("        ");
    for (int64_t j = 0; j < N; ++j)
    {
        printf("   j=%-8lld", j);
    }
    printf("\n");

    for (int64_t i = 0; i < M; ++i)
    {
        printf("i=%-3lld ", i);
        for (int64_t j = 0; j < N; ++j)
        {
            printf("%12.6f ", C[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}
void print_batch_gemm_output(
    const char *name, double **C, int64_t batch, int64_t M, int64_t N, int64_t max_batch_to_print)
{
    int64_t limit = batch < max_batch_to_print ? batch : max_batch_to_print;

    for (int64_t b = 0; b < limit; ++b)
    {
        print_batch_matrix(name, b, C[b], M, N);
    }

    if (limit < batch)
    {
        printf("... (%lld more batches omitted)\n", batch - limit);
    }
}

#if defined(__clang__)
#define ALWAYS_INLINE
#else
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

#define SME_STATE __arm_streaming __arm_inout("za")

#define TIMES 10000
extern "C" void enable_sme(char *);
extern "C" void disable_sme(char *);
extern "C" void enable_sme_context_unchange(char *);
extern "C" void disable_sme_context_unchange(char *);
template <std::size_t... Is>
__attribute__((always_inline)) constexpr void loop(std::index_sequence<Is...>, auto &&f) SME_STATE
{
    (f.template operator()<Is>(), ...);
}

#define CONSTEXPR_FOR_BEGIN(ITER, RANGE, ...)                                                                          \
    loop(std::make_index_sequence<RANGE>{}, [=, ##__VA_ARGS__]<size_t ITER>()  SME_STATE   ALWAYS_INLINE             \
    {
#define CONSTEXPR_FOR_END                                                                                              \
    }                                                                                                                  \
    );

__arm_new("za") void dgemm_4x4_fmla_tn(
    int64_t batch, double **A, double **B, double **C, int64_t M, int64_t N, int64_t K) __arm_streaming
{
    for (int64_t b = 0; b < batch; b += 16)
    {
        CONSTEXPR_FOR_BEGIN(bb, 4)
        {
            if (b + bb * 4 >= batch)
            {
                return;
            }
            // svzero_mask_za(1<<bb);
            svfloat64_t va;
            svfloat64x4_t vb;
            svmla_za64_vg1x4(0, vb, va);
        }
        CONSTEXPR_FOR_END
    }
}
__arm_new("za") void dgemm_4x4_8_tn(
    int64_t batch, double **A, double **B, double **C, int64_t M, int64_t N, int64_t K) __arm_streaming
{
    for (int64_t b = 0; b < batch; b += 8)
    {
        // for (int64_t bb = 0; b + bb < batch; ++bb)
        // CONSTEXPR_FOR_BEGIN(bb, 4)
        {
            // if (b + bb * 2 >= batch)
            // {
            //     return;
            // }
            // svzero_mask_za(1<<bb);
            svzero_za();
            for (int k = 0; k < K; k++)
            {
                CONSTEXPR_FOR_BEGIN(bb, 4)
                auto p0 = svwhilelt_b64(M, 8l);
                auto p1 = svwhilelt_b64(N, 8l);
                svfloat64_t va0 = svld1(p0, &A[b + bb * 2][k * M]);
                svfloat64_t va1 = svld1(p0, &A[b + bb * 2 + 1][k * M]);
                svfloat64_t va = svzip1(va0, va1);
                svfloat64_t vb0 = svld1(p1, &B[b + bb][k * N]);
                svfloat64_t vb1 = svld1(p1, &B[b + bb * 2 + 1][k * N]);
                svfloat64_t vb = svzip1(vb0, vb1);
                svmopa_za64_m(bb, p0, p1, va, vb);
                CONSTEXPR_FOR_END
            }
            CONSTEXPR_FOR_BEGIN(bb, 4)
            for (int i = 0; i < M; ++i)
            {
                auto p0 = svwhilelt_b64(M, 8l);
                auto p1 = svwhilelt_b64(N, 8l);
                svfloat64_t vc;
                vc = svread_hor_za64_m(vc, p1, bb, i);
                svfloat64_t vc1 = svread_hor_za64_m(svundef_f64(), p1, bb, i + 4);
                {
                    svbool_t p_start = svwhilelt_b64(0, i * 2);   // start之前为false
                    svbool_t p_end = svwhilelt_b64(0, i * 2 + 4); // end之前为false
                    svbool_t p_mid = sveor_z(svptrue_b64(), p_start, p_end);
                    // svbool_t p_mid = p1;
                    vc = svuzp1(vc, svundef_f64());
                    svst1(p_mid, &C[b + bb * 2][i * N], vc);
                }
                {
                    svbool_t p_start = svwhilelt_b64(0, i * 2 + 1);   // start之前为false
                    svbool_t p_end = svwhilelt_b64(0, i * 2 + 4 + 1); // end之前为false
                    svbool_t p_mid = sveor_z(svptrue_b64(), p_start, p_end);

                    // svbool_t p_mid = p1;
                    vc1 = svuzp2(vc1, svundef_f64());
                    svst1(p_mid, &C[b + bb * 2 + 1][i * N], vc1);
                }
            }
            CONSTEXPR_FOR_END
        }
        // CONSTEXPR_FOR_END
    }
}
__arm_new("za") void dgemm_less_than_8_tn(
    int64_t batch, double **A, double **B, double **C, int64_t M, int64_t N, int64_t K) __arm_streaming
{
    for (int64_t b = 0; b < batch; b += 4)
    {
        // for (int64_t bb = 0; b + bb < batch; ++bb)
        CONSTEXPR_FOR_BEGIN(bb, 4)
        {
            if (b + bb >= batch)
            {
                return;
            }
            svzero_mask_za(1 << bb);

            auto p0 = svwhilelt_b64(M, 8l);
            auto p1 = svwhilelt_b64(N, 8l);
            for (int k = 0; k < K; k++)
            {
                svfloat64_t va = svld1(p0, &A[b + bb][k * M]);
                svfloat64_t vb = svld1(p1, &B[b + bb][k * N]);
                svmopa_za64_m(bb, p0, p1, va, vb);
            }
            for (int i = 0; i < M; ++i)
            {
                svfloat64_t vc;
                vc = svread_hor_za64_m(vc, p1, bb, i);
                svst1(p1, &C[b + bb][i * N], vc);
            }
        }
        CONSTEXPR_FOR_END
    }
}

__arm_new("za") void dgemm_less_than_8_nn(
    int64_t batch, double **A, double **B, double **C, int64_t M, int64_t N, int64_t K) __arm_streaming
{
    for (int64_t b = 0; b < batch; b += 4)
    {
        CONSTEXPR_FOR_BEGIN(bb, 4)
        {
            auto p0 = svwhilelt_b64(M, 8l);
            auto p1 = svwhilelt_b64(N, 8l);
            for (int i = 0; i < M; i++)
            {
                svfloat64_t va = svld1(p0, &A[b + bb][i * K]);
                svwrite_ver_za64_m(bb + 4, i, svptrue_b8(), va);
            }
        }
        CONSTEXPR_FOR_END

        // for (int64_t bb = 0; b + bb < batch; ++bb)
        CONSTEXPR_FOR_BEGIN(bb, 4)
        {
            if (b + bb >= batch)
            {
                return;
            }
            svzero_mask_za(1 << bb);

            auto p0 = svwhilelt_b64(M, 8l);
            auto p1 = svwhilelt_b64(N, 8l);
            // for (int i = 0; i < M; i++)
            // {
            //     svfloat64_t va = svld1(p0, &A[b+bb][i*K]);
            //     svwrite_ver_za64_m(bb+4, i, svptrue_b8(), va);
            // }
            for (int k = 0; k < K; k++)
            {
                svfloat64_t va = svread_hor_za64_m(svundef_f64(), p0, bb + 4, k);
                svfloat64_t vb = svld1(p1, &B[b + bb][k * N]);
                svmopa_za64_m(bb, p0, p1, va, vb);
            }
            for (int i = 0; i < M; ++i)
            {
                svfloat64_t vc;
                vc = svread_hor_za64_m(vc, p1, bb, i);
                svst1(p1, &C[b + bb][i * 8], vc);
            }
        }
        CONSTEXPR_FOR_END
    }
}
// 随机初始化矩阵
void rand_fill(double *p, int64_t size)
{
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int64_t i = 0; i < size; i++)
    {
        p[i] = dist(rng);
    }
}

int fn(int argc, char **argv)
{
    char buffer[512];
    disable_sme(buffer);
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " <batch> <M> <N> <K>\n";
        return 1;
    }

    int64_t batch = std::atoll(argv[1]);
    int64_t M = std::atoll(argv[2]);
    int64_t N = std::atoll(argv[3]);
    int64_t K = std::atoll(argv[4]);

    std::cout << "Batch GEMM Benchmark\n";
    std::cout << "Batch = " << batch << "\n";
    std::cout << "M,N,K = " << M << "," << N << "," << K << "\n";

    // 分配 batch 指针数组
    double **A = new double *[batch];
    double **B = new double *[batch];
    double **C = new double *[batch];

    // 分配矩阵
    for (int64_t i = 0; i < batch; i++)
    {
        A[i] = new double[M * K];
        B[i] = new double[K * N];
        C[i] = new double[M * N * 10];

        rand_fill(A[i], M * K);
        rand_fill(B[i], K * N);
        std::fill(C[i], C[i] + M * N, 0.0);
    }

#define TEST(test)                                                                                                     \
    {                                                                                                                  \
        enable_sme(buffer);                                                                                            \
        for (int i = 0; i < 10; ++i)                                                                                   \
        {                                                                                                              \
            test(batch, A, B, C, M, N, K);                                                                             \
        }                                                                                                              \
        disable_sme(buffer);                                                                                           \
        auto start = std::chrono::high_resolution_clock::now();                                                        \
        for (int i = 0; i < TIMES; ++i)                                                                                \
        {                                                                                                              \
            enable_sme(buffer);                                                                                        \
            test(batch, A, B, C, M, N, K);                                                                             \
            disable_sme(buffer);                                                                                       \
        }                                                                                                              \
        auto end = std::chrono::high_resolution_clock::now();                                                          \
                                                                                                                       \
        double time_s = std::chrono::duration<double>(end - start).count();                                            \
                                                                                                                       \
        double flops = (double)batch * 2.0 * M * N * K;                                                                \
        double gflops = flops / time_s / 1e9 * TIMES;                                                                  \
                                                                                                                       \
        std::cout << #test << "\n";                                                                                    \
        std::cout << "Time    = " << time_s << " sec\n";                                                               \
        std::cout << "GFLOPS  = " << gflops << "\n";                                                                   \
    };

    TEST(dgemm_4x4_8_tn);
    TEST(dgemm_less_than_8_tn);

    // 释放内存
    for (int64_t i = 0; i < batch; i++)
    {
        delete[] A[i];
        delete[] B[i];
        delete[] C[i];
    }
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

#include <stdint.h>

__attribute__((noinline)) void
batch_dgemm_sve_tn(int64_t batch, double **A, double **B, double **C, int64_t M, int64_t N, int64_t K) __arm_streaming
{
    for (int64_t b = 0; b < batch; ++b)
    {
        double *Ab = A[b]; // K x M
        double *Bb = B[b]; // K x N
        double *Cb = C[b]; // M x N

        for (int64_t i = 0; i < M; ++i)
        {
            for (int64_t j = 0; j < N; j += 8)
            {
                svfloat64_t sum = svdup_f64(0);
                svbool_t p0 = svwhilelt_b64(j, N);
                for (int64_t k = 0; k < K; ++k)
                {
                    // A^T(i, k) = A(k, i)
                    // sum += Ab[k * M + i] * Bb[k * N + j];
                    svfloat64_t vb = svld1(p0, &Bb[k * N + j]);
                    sum = svmla_m(p0, sum, vb, Ab[k * M + i]);
                }
                // Cb[i * N + j] = sum;
                svst1(p0, &Cb[i * N + j], sum);
            }
        }
    }
}

__attribute__((noinline)) __attribute__((target("+nosve+nosme"))) void
batch_dgemm_tn(int64_t batch, double **A, double **B, double **C, int64_t M, int64_t N, int64_t K)
{
    asm volatile("smstop");
    for (int64_t b = 0; b < batch; ++b)
    {
        double *Ab = A[b]; // K x M
        double *Bb = B[b]; // K x N
        double *Cb = C[b]; // M x N

        for (int64_t i = 0; i < M; ++i)
        {
            for (int64_t j = 0; j < N; ++j)
            {
                double sum = 0.0;
                for (int64_t k = 0; k < K; ++k)
                {
                    // A^T(i, k) = A(k, i)
                    sum += Ab[k * M + i] * Bb[k * N + j];
                }
                Cb[i * N + j] = sum;
            }
        }
    }
}

__arm_new("za") void batch_dgemm_sme_b4_tn(
    int64_t batch, double **A, double **B, double **C, int64_t M, int64_t N, int64_t K) __arm_streaming
{
    for (int64_t b = 0; b < batch; b += 4)
    {
        for (int i = 0; i < M; i += 8)
        {
            for (int64_t j = 0; j < N; j += 8)
            {
                svzero_za();
                for (int k = 0; k < K; ++k)
                {
                    svbool_t p0 = svptrue_b64();
                    svbool_t ptrue = svptrue_b64();
                    svfloat64_t a00 = svld1(p0, &A[b + 0][k * M + i]);
                    svfloat64_t b00 = svld1(p0, &B[b + 0][k * N + j]);

                    svfloat64_t a10 = svld1(p0, &A[b + 1][k * M + i]);
                    svfloat64_t b10 = svld1(p0, &B[b + 1][k * N + j]);

                    svfloat64_t a20 = svld1(p0, &A[b + 2][k * M + i]);
                    svfloat64_t b20 = svld1(p0, &B[b + 2][k * N + j]);

                    svfloat64_t a30 = svld1(p0, &A[b + 3][k * M + i]);
                    svfloat64_t b30 = svld1(p0, &B[b + 3][k * N + j]);

                    svmopa_za64_m(0, ptrue, ptrue, a00, b00);
                    svmopa_za64_m(1, ptrue, ptrue, a10, b10);
                    svmopa_za64_m(2, ptrue, ptrue, a20, b20);
                    svmopa_za64_m(3, ptrue, ptrue, a30, b30);
                }

                CONSTEXPR_FOR_BEGIN(za, 4){CONSTEXPR_FOR_BEGIN(ii, 8){if (i + ii >= M){return;
            }
            svbool_t ptrue = svptrue_b64();
            svbool_t p0 = svwhilelt_b64(j, N);
            svfloat64_t vc = svread_hor_za64_m(svundef_f64(), ptrue, za, ii);
            svst1(p0, &C[b + za][(i + ii) * N + j], vc);
        }
        CONSTEXPR_FOR_END
    }
    CONSTEXPR_FOR_END
}
}
}
}
__arm_new("za") void batch_dgemm_5x5xk_4_tn(
    int64_t batch, double **A, double **B, double **C, int64_t M, int64_t N, int64_t K) __arm_streaming
{
    for (int64_t b = 0; b < batch; b += 4)
    {
        constexpr int g = 0;
        // CONSTEXPR_FOR_BEGIN(g, 1)
        {
            // svzero_mask_za(0b1111 << (4 * g));
            svzero_za();
            svbool_t ptrue = svptrue_b64();    // p6
            svbool_t p0 = svwhilelt_b64(0, 5); // p7
            for (int k = 0; k < K; ++k)
            {
                __asm__ volatile(
                    /* =========================
     * Load A
     * ========================= */
                    "ld1d z0.d, p7/z, [%[A0]] \n" // a00
                    "ld1d z1.d, p7/z, [%[A1]] \n" // a10
                    "ld1d z2.d, p7/z, [%[A2]] \n" // a20
                    "ld1d z3.d, p7/z, [%[A3]] \n" // a30

                    /* =========================
     * Load B
     * ========================= */
                    "ld1d z4.d, p7/z, [%[B0]] \n" // b00
                    "ld1d z5.d, p7/z, [%[B1]] \n" // b10
                    "ld1d z6.d, p7/z, [%[B2]] \n" // b20
                    "ld1d z7.d, p7/z, [%[B3]] \n" // b30

                    /* =========================
     * MOPA (ZA accumulators)
     * ========================= */
                    "fmopa za0.d, p6/m, p6/m, z0.d, z4.d \n"
                    "fmopa za1.d, p6/m, p6/m, z1.d, z5.d \n"
                    "fmopa za2.d, p6/m, p6/m, z2.d, z6.d \n"
                    "fmopa za3.d, p6/m, p6/m, z3.d, z7.d \n"
                    :
                    : [A0] "r"(&A[b + 4 * g + 0][k * M]),
                      [A1] "r"(&A[b + 4 * g + 1][k * M]),
                      [A2] "r"(&A[b + 4 * g + 2][k * M]),
                      [A3] "r"(&A[b + 4 * g + 3][k * M]),

                      [B0] "r"(&B[b + 4 * g + 0][k * N]),
                      [B1] "r"(&B[b + 4 * g + 1][k * N]),
                      [B2] "r"(&B[b + 4 * g + 2][k * N]),
                      [B3] "r"(&B[b + 4 * g + 3][k * N])

                    : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "memory");

                // svfloat64_t a00 = svld1(p0, &A[b+4*g+0][k*M]);
                // svfloat64_t b00 = svld1(p0, &B[b+4*g+0][k*N]);

                // svfloat64_t a10 = svld1(p0, &A[b+4*g+1][k*M]);
                // svfloat64_t b10 = svld1(p0, &B[b+4*g+1][k*N]);

                // svfloat64_t a20 = svld1(p0, &A[b+4*g+2][k*M]);
                // svfloat64_t b20 = svld1(p0, &B[b+4*g+2][k*N]);

                // svfloat64_t a30 = svld1(p0, &A[b+4*g+3][k*M]);
                // svfloat64_t b30 = svld1(p0, &B[b+4*g+3][k*N]);

                // svmopa_za64_m(g*4+0, ptrue, ptrue, a00, b00);
                // svmopa_za64_m(g*4+1, ptrue, ptrue, a10, b10);
                // svmopa_za64_m(g*4+2, ptrue, ptrue, a20, b20);
                // svmopa_za64_m(g*4+3, ptrue, ptrue, a30, b30);
            }

            CONSTEXPR_FOR_BEGIN(za, 4){CONSTEXPR_FOR_BEGIN(i, 5){svbool_t ptrue = svptrue_b64();
            svbool_t p0 = svwhilelt_b64(0, 5);
            svbool_t p40 = svnot_z(ptrue, p0);
            svbool_t p41 = svnot_z(ptrue, svwhilelt_b64(0, 6));
            svfloat64_t vc = svread_hor_za64_m(svundef_f64(), ptrue, za, i);
            svst1(p0, &C[b + 4 * g + za][i * N], vc);
        }
        CONSTEXPR_FOR_END
    }
    CONSTEXPR_FOR_END
}
// CONSTEXPR_FOR_END
}
}

__arm_new("za") void batch_dgemm_5x5xk_5_no_cat_tn(
    int64_t batch, double **A, double **B, double **C, int64_t M, int64_t N, int64_t K) __arm_streaming
{
    for (int64_t b = 0; b < batch; b += 5)
    {
        svzero_za();
        svbool_t ptrue = svptrue_b64();
        svbool_t p0 = svwhilelt_b64(0, 5);
        for (int k = 0; k < K; ++k)
        {
            svfloat64_t a00 = svld1(p0, &A[b + 0][k * M]);
            svfloat64_t b00 = svld1(p0, &B[b + 0][k * N]);

            svfloat64_t a10 = svld1(p0, &A[b + 1][k * M]);
            svfloat64_t b10 = svld1(p0, &B[b + 1][k * N]);

            svfloat64_t a20 = svld1(p0, &A[b + 2][k * M]);
            svfloat64_t b20 = svld1(p0, &B[b + 2][k * N]);

            svfloat64_t a30 = svld1(p0, &A[b + 3][k * M]);
            svfloat64_t b30 = svld1(p0, &B[b + 3][k * N]);

            svfloat64_t a40 = svld1(p0, &A[b + 4][k * M]);
            svfloat64_t b40 = svld1(p0, &B[b + 4][k * N]);

            svmopa_za64_m(0, ptrue, ptrue, a00, b00);
            svmopa_za64_m(1, ptrue, ptrue, a10, b10);
            svmopa_za64_m(2, ptrue, ptrue, a20, b20);
            svmopa_za64_m(3, ptrue, ptrue, a30, b30);
            svmopa_za64_m(4, ptrue, ptrue, a40, b40);
        }

        CONSTEXPR_FOR_BEGIN(za, 5){CONSTEXPR_FOR_BEGIN(i, 5){svbool_t ptrue = svptrue_b64();
        svbool_t p0 = svwhilelt_b64(0, 5);
        svbool_t p40 = svnot_z(ptrue, p0);
        svbool_t p41 = svnot_z(ptrue, svwhilelt_b64(0, 6));
        svfloat64_t vc = svread_hor_za64_m(svundef_f64(), ptrue, za, i);
        svst1(p0, &C[b + za][i * N], vc);
    }
    CONSTEXPR_FOR_END
}
CONSTEXPR_FOR_END
}
}

__arm_new("za") void batch_dgemm_4x12xk_8p4_tn(
    int64_t batch, double **A, double **B, double **C, int64_t _M, int64_t _N, int64_t _K) __arm_streaming
{
    constexpr int M = 4;
    constexpr int N = 12;
    // constexpr int K = 32;
    int K = _K;
    for (int64_t b = 0; b < batch; b += 2)
    {
        svzero_za();

        svbool_t ptrue = svptrue_b64();
        svbool_t p0 = svwhilelt_b64(0, 5);
        // for (int k = 0; k < K; k += 2)
        for (int k = 0; k < K; k += 1)
        // CONSTEXPR_FOR_BEGIN(k, K)
        {
            svbool_t ptrue = svptrue_b64();
            svbool_t p0 = svwhilelt_b64(0, 4);
            svfloat64_t a00 = svld1(ptrue, &A[b + 0][k * M]);
            svfloat64_t b00 = svld1(ptrue, &B[b + 0][k * N]);
            svfloat64_t b01 = svld1(p0, &B[b + 0][k * N + 8]);

            svfloat64_t a10 = svld1(ptrue, &A[b + 1][k * M]);
            svfloat64_t b10 = svld1(ptrue, &B[b + 1][k * N]);
            svfloat64_t b11 = svld1(p0, &B[b + 1][k * N + 8]);

            // svfloat64_t a01 = svld1(ptrue, &A[b+0][(k+1)*M]);
            // svfloat64_t b02 = svld1(ptrue, &B[b+0][(k+1)*N]);
            // svfloat64_t b03 = svld1(p0, &B[b+0][(k+1)*N + 8]);

            // svfloat64_t a11 = svld1(ptrue, &A[b+1][(1+k)*M]);
            // svfloat64_t b12 = svld1(ptrue, &B[b+1][(1+k)*N]);
            // svfloat64_t b13 = svld1(p0, &B[b+1][(1+k)*N + 8]);

            svmopa_za64_m(0, ptrue, ptrue, a00, b00);
            svmopa_za64_m(1, ptrue, ptrue, a00, b01);
            svmopa_za64_m(2, ptrue, ptrue, a10, b10);
            svmopa_za64_m(3, ptrue, ptrue, a10, b11);
            // svmopa_za64_m(0, ptrue, ptrue, a01, b02);
            // svmopa_za64_m(1, ptrue, ptrue, a01, b03);
            // svmopa_za64_m(2, ptrue, ptrue, a11, b12);
            // svmopa_za64_m(3, ptrue, ptrue, a11, b13);
        }
        // CONSTEXPR_FOR_END

        CONSTEXPR_FOR_BEGIN(za, 2){CONSTEXPR_FOR_BEGIN(i, 4){svbool_t ptrue = svptrue_b64();
        svbool_t p0 = svwhilelt_b64(0, 5);
        svfloat64_t vc0 = svread_hor_za64_m(svundef_f64(), ptrue, za * 2, i);
        svfloat64_t vc1 = svread_hor_za64_m(svundef_f64(), ptrue, za * 2 + 1, i);
        svst1(ptrue, &C[b + za][i * N], vc0);
        svst1(p0, &C[b + za][i * N + 8], vc1);
    }
    CONSTEXPR_FOR_END
}
CONSTEXPR_FOR_END
}
}
__arm_new("za") void batch_dgemm_4x13xk_8p4p1_1sve_b4_tn(
    int64_t batch, double **A, double **B, double **C, int64_t _M, int64_t _N, int64_t _K) __arm_streaming
{
    constexpr int M = 4;
    constexpr int N = 13;
    int K = _K;
    int64_t index[8] = {0, 13, 13 * 2, 13 * 3, 0, 13, 13 * 2, 13 * 3};
    svint64_t v_index = svld1(svptrue_b64(), index);
    for (int64_t b = 0; b < batch; b += 4)
    {
        svzero_za();

        svbool_t ptrue = svptrue_b64();
        svbool_t p0 = svwhilelt_b64(0, 4);
        svbool_t p2 = svnot_z(ptrue, p0);
        svfloat64_t C0 = svdup_f64(0.);
        svfloat64_t C1 = svdup_f64(0.);
        for (int k = 0; k < K; k += 1)
        // CONSTEXPR_FOR_BEGIN(k, K)
        {
            svfloat64_t a00 = svld1(ptrue, &A[b + 0][k * M]);
            svfloat64_t b00 = svld1(ptrue, &B[b + 0][k * N]);
            svfloat64_t b01 = svld1(p0, &B[b + 0][k * N + 8]);

            svfloat64_t a10 = svld1(p2, &A[b + 1][k * M - 4]);
            svfloat64_t b10 = svld1(ptrue, &B[b + 1][k * N]);
            // svfloat64_t b11 = svld1(p2, &B[b+1][k*N + 8]-4);
            svfloat64_t b11 = svrev(svld1(p0, &B[b + 1][k * N + 8]));

            svfloat64_t a0 = svsel(p0, a00, a10);
            svfloat64_t b1 = svsel(p0, b01, b11);

            svfloat64_t a20 = svld1(ptrue, &A[b + 2][k * M]);
            svfloat64_t b20 = svld1(ptrue, &B[b + 2][k * N]);
            svfloat64_t b21 = svld1(p0, &B[b + 2][k * N + 8]);

            svfloat64_t a30 = svld1(p2, &A[b + 3][k * M - 4]);
            svfloat64_t b30 = svld1(ptrue, &B[b + 3][k * N]);
            // svfloat64_t b31 = svld1(p2, &B[b+3][k*N + 8]-4);
            svfloat64_t b31 = svrev(svld1(p0, &B[b + 3][k * N + 8]));

            svfloat64_t a4 = svsel(p0, a20, a30);
            svfloat64_t b4 = svsel(p0, b21, b31);

            svmopa_za64_m(0, ptrue, ptrue, a00, b00);
            svmopa_za64_m(1, ptrue, ptrue, a0, b1);
            svmopa_za64_m(2, ptrue, ptrue, a10, b10);

            svmopa_za64_m(3, ptrue, ptrue, a20, b20);
            svmopa_za64_m(4, ptrue, ptrue, a4, b4);
            svmopa_za64_m(5, ptrue, ptrue, a30, b30);
            {
                C0 = svmla_m(p0, C0, a0, B[b][k * N + N - 1]);
                C0 = svmla_m(p2, C0, a0, B[b + 1][k * N + N - 1]);
                C1 = svmla_m(p0, C1, a4, B[b + 2][k * N + N - 1]);
                C1 = svmla_m(p2, C1, a4, B[b + 3][k * N + N - 1]);
            }
        }
        // svuint64_t ptr0 = svdup_s64_m(svwhilelt_b64(0, 4), (uint64_t)(&C[b]), svdup_s64((uint64_t)(&C[b+1])));
        // svuint64_t ptr1 = svdup_s64_m(svwhilelt_b64(0, 4), (uint64_t)(&C[b+2]), svdup_s64((uint64_t)(&C[b+3])));
        // svst1_scatter(ptrue, ptr0, C0);
        // svst1_scatter(ptrue, ptr1, C1);
        // FIXME: use scatter
        // svst1(ptrue, &C[b][0], C0);
        // svst1(ptrue, &C[b+1][0], C1);
        // CONSTEXPR_FOR_END

        CONSTEXPR_FOR_BEGIN(g, 2, &C0, &C1){CONSTEXPR_FOR_BEGIN(i, 4, &C0, &C1){svbool_t ptrue = svptrue_b64();
        svbool_t p0 = svwhilelt_b64(0, 4);
        svbool_t p2 = svnot_z(ptrue, p0);
        svbool_t p1 = svwhilelt_b64(0, 5);
        svfloat64_t vc0 = svread_hor_za64_m(svundef_f64(), ptrue, g * 3, i);
        svfloat64_t vc1 = svread_hor_za64_m(svdup_lane(select < g == 0 > (C0, C1), i), p0, g * 3 + 1, i);
        svfloat64_t vc2 = svread_hor_za64_m(svdup_lane(select < g == 0 > (C0, C1), i + 4), p2, g * 3 + 1, i + 4);
        svfloat64_t vc3 = svread_hor_za64_m(svundef_f64(), ptrue, g * 3 + 2, i + 4);
        svst1(ptrue, &C[b + g * 2][i * N], vc0);
        svst1(p1, &C[b + g * 2][i * N + 8], vc1);
        // svst1(svnot_z(ptrue, p0), &C[b+g*2+1][i*N+8-4], vc2);
        svst1(p1, &C[b + g * 2 + 1][i * N + 8], svrev(vc2));
        // svst1(p1, &C[b+g*2+1][i*N+8],  vc2);
        svst1(ptrue, &C[b + g * 2 + 1][i * N], vc3);
    }
    CONSTEXPR_FOR_END
}
CONSTEXPR_FOR_END
}
}

__arm_new("za") void batch_dgemm_4x9xk_8p1_1sve_b4_tn(
    int64_t batch, double **A, double **B, double **C, int64_t _M, int64_t _N, int64_t _K) __arm_streaming
{
    constexpr int M = 4;
    constexpr int N = 9;
    int K = _K;
    for (int64_t b = 0; b < batch; b += 4)
    {
        svzero_za();

        svbool_t ptrue = svptrue_b64();
        // svbool_t p0 = svwhilelt_b64(0, 4);
        svbool_t p0 = svwhilelt_b64(0, 4);
        svbool_t p1 = svwhilelt_b64(0, 5);
        svbool_t p2 = svnot_z(ptrue, p0);
        svfloat64_t C0 = svdup_f64(0.);
        svfloat64_t C1 = svdup_f64(0.);
        svfloat64_t C2 = svdup_f64(0.);
        svfloat64_t C3 = svdup_f64(0.);
        for (int k = 0; k < K; k += 1)
        // CONSTEXPR_FOR_BEGIN(k, K)
        {
            CONSTEXPR_FOR_BEGIN(za, 2, &C0, &C1, &C2, &C3, &p0, &p2, &ptrue)
            {
                svfloat64_t a00 = svld1(p0, &A[b + za * 2][k * M]);
                svfloat64_t a10 = svld1(svnot_z(ptrue, p0), &A[b + za * 2 + 1][k * M - 4]);

                svfloat64_t b00 = svld1(ptrue, &B[b + za * 2][k * N]);
                svfloat64_t b10 = svld1(ptrue, &B[b + za * 2 + 1][k * N]);
                auto &Ci = selectI<za>(C0, C1, C2, C3);
                svmopa_za64_m(za * 2, ptrue, ptrue, a00, b00);
                svmopa_za64_m(za * 2 + 1, ptrue, ptrue, a10, b10);
                Ci = svmla_m(p0, Ci, a00, B[b + 2 * za][k * N + N - 1]);
                Ci = svmla_m(p2, Ci, a10, B[b + 2 * za + 1][k * N + N - 1]);
            }
            CONSTEXPR_FOR_END
        }

        CONSTEXPR_FOR_BEGIN(za, 2, &C0, &C1, &C2, &C3, &p0, &p1, &p2, &ptrue){
            CONSTEXPR_FOR_BEGIN(i, 4, &C0, &C1, &C2, &C3, &p0, &p1, &p2, &ptrue){
                svfloat64_t vc0 = svread_hor_za64_m(svundef_f64(), ptrue, za * 2, i);
        svfloat64_t vc1 = svread_hor_za64_m(svundef_f64(), ptrue, za * 2 + 1, i + 4);

        auto &Ci = selectI<za>(C0, C1, C2, C3);

        svst1(ptrue, &C[b + za * 2][i * N], vc0);
        svst1(ptrue, &C[b + za * 2 + 1][i * N], (vc1));
        svst1(svwhilelt_b64(0, 1), &C[b + za * 2][i * N + 8], svdup_lane(Ci, i));
        svst1(svwhilelt_b64(0, 1), &C[b + za * 2 + 1][i * N + 8], svdup_lane(Ci, i + 4));
    }
    CONSTEXPR_FOR_END
}
CONSTEXPR_FOR_END
}
}
__arm_new("za") void batch_dgemm_4x5xk_4p1_1sve_b8_tn(
    int64_t batch, double **A, double **B, double **C, int64_t _M, int64_t _N, int64_t _K) __arm_streaming
{
    constexpr int M = 4;
    constexpr int N = 5;
    int K = _K;
    for (int64_t b = 0; b < batch; b += 8)
    {
        svzero_za();

        svbool_t ptrue = svptrue_b64();
        // svbool_t p0 = svwhilelt_b64(0, 4);
        svbool_t p0 = svwhilelt_b64(0, 4);
        svbool_t p1 = svwhilelt_b64(0, 5);
        svbool_t p2 = svnot_z(ptrue, p0);
        svfloat64_t C0 = svdup_f64(0.);
        svfloat64_t C1 = svdup_f64(0.);
        svfloat64_t C2 = svdup_f64(0.);
        svfloat64_t C3 = svdup_f64(0.);
        for (int k = 0; k < K; k += 1)
        // CONSTEXPR_FOR_BEGIN(k, K)
        {
            CONSTEXPR_FOR_BEGIN(za, 4, &C0, &C1, &C2, &C3, &p0, &p2, &ptrue)
            {
                svfloat64_t a00 = svld1(p0, &A[b + za * 2][k * M]);
                svfloat64_t a10 = svld1(svnot_z(ptrue, p0), &A[b + za * 2 + 1][k * M - 4]);

                svfloat64_t b00 = svld1(p0, &B[b + za * 2][k * N]);
                svfloat64_t b10 = svrev(svld1(p0, &B[b + za * 2 + 1][k * N]));
                auto &Ci = selectI<za>(C0, C1, C2, C3);
                svfloat64_t a0 = svsel(p0, a00, a10);
                svfloat64_t b0 = svsel(p0, b00, b10);
                svmopa_za64_m(za, ptrue, ptrue, a0, b0);
                Ci = svmla_m(p0, Ci, a0, B[b + 2 * za][k * N + N - 1]);
                Ci = svmla_m(p2, Ci, a0, B[b + 2 * za + 1][k * N + N - 1]);
            }
            CONSTEXPR_FOR_END
        }

        CONSTEXPR_FOR_BEGIN(za, 4, &C0, &C1, &C2, &C3, &p0, &p1, &p2, &ptrue){
            CONSTEXPR_FOR_BEGIN(i, 4, &C0, &C1, &C2, &C3, &p0, &p1, &p2, &ptrue){
                svfloat64_t vc0 = svread_hor_za64_m(svundef_f64(), ptrue, za, i);
        svfloat64_t vc1 = svread_hor_za64_m(svundef_f64(), ptrue, za, i + 4);

        auto &Ci = selectI<za>(C0, C1, C2, C3);

        svst1(p0, &C[b + za * 2][i * N], vc0);
        svst1(p1, &C[b + za * 2 + 1][i * N], svrev(vc1));
        svst1(svwhilelt_b64(0, 1), &C[b + za * 2][i * N + 4], svdup_lane(Ci, i));
        svst1(svwhilelt_b64(0, 1), &C[b + za * 2 + 1][i * N + 4], svdup_lane(Ci, i + 4));
    }
    CONSTEXPR_FOR_END
}
CONSTEXPR_FOR_END
}
}
__arm_new("za") void batch_dgemm_4x13xk_8p4p1_1scalar_b4_tn(
    int64_t batch, double **A, double **B, double **C, int64_t _M, int64_t _N, int64_t _K) __arm_streaming
{
    constexpr int M = 4;
    constexpr int N = 13;
    int K = _K;
    for (int64_t b = 0; b < batch; b += 4)
    {
        svzero_za();

        svbool_t ptrue = svptrue_b64();
        svbool_t p0 = svwhilelt_b64(0, 4);
        svbool_t p2 = svnot_z(ptrue, p0);
        double C0 = 0;
        double C1 = 0;
        double C2 = 0;
        double C3 = 0;
        double C4 = 0;
        double C5 = 0;
        double C6 = 0;
        double C7 = 0;
        double C8 = 0;
        double C9 = 0;
        double C10 = 0;
        double C11 = 0;
        double C12 = 0;
        double C13 = 0;
        double C14 = 0;
        double C15 = 0;
        for (int k = 0; k < K; k += 1)
        // CONSTEXPR_FOR_BEGIN(k, K)
        {
            svfloat64_t a00 = svld1(ptrue, &A[b + 0][k * M]);
            svfloat64_t b00 = svld1(ptrue, &B[b + 0][k * N]);
            svfloat64_t b01 = svld1(p0, &B[b + 0][k * N + 8]);

            svfloat64_t a10 = svld1(p2, &A[b + 1][k * M - 4]);
            svfloat64_t b10 = svld1(ptrue, &B[b + 1][k * N]);
            svfloat64_t b11 = svld1(p2, &B[b + 1][k * N + 8] - 4);

            svfloat64_t a0 = svsel(p0, a00, a10);
            svfloat64_t b1 = svsel(p0, b01, b11);

            svfloat64_t a20 = svld1(ptrue, &A[b + 2][k * M]);
            svfloat64_t b20 = svld1(ptrue, &B[b + 2][k * N]);
            svfloat64_t b21 = svld1(p0, &B[b + 2][k * N + 8]);

            svfloat64_t a30 = svld1(p2, &A[b + 3][k * M - 4]);
            svfloat64_t b30 = svld1(ptrue, &B[b + 3][k * N]);
            svfloat64_t b31 = svld1(p2, &B[b + 3][k * N + 8] - 4);

            svfloat64_t a4 = svsel(p0, a20, a30);
            svfloat64_t b4 = svsel(p0, b21, b31);

            svmopa_za64_m(0, ptrue, ptrue, a00, b00);
            svmopa_za64_m(1, ptrue, ptrue, a0, b1);
            svmopa_za64_m(2, ptrue, ptrue, a10, b10);

            svmopa_za64_m(3, ptrue, ptrue, a20, b20);
            svmopa_za64_m(4, ptrue, ptrue, a4, b4);
            svmopa_za64_m(5, ptrue, ptrue, a30, b30);
#define SCALAR_C(id) C##id += A[b + id / 4][id % 4 + k * M] * B[b + id / 4][k * N + N - 1];
            SCALAR_C(0);
            SCALAR_C(1);
            SCALAR_C(2);
            SCALAR_C(3);
            SCALAR_C(4);
            SCALAR_C(5);
            SCALAR_C(6);
            SCALAR_C(7);
            SCALAR_C(8);
            SCALAR_C(9);
            SCALAR_C(10);
            SCALAR_C(11);
            SCALAR_C(12);
            SCALAR_C(13);
            SCALAR_C(14);
            SCALAR_C(15);
#undef SCALAR_C
            // CONSTEXPR_FOR_BEGIN(bb, 4)
            // {
            //     CONSTEXPR_FOR_BEGIN(ii, 4)
            //     {
            //         C[b+bb][(ii)*N+N-1] += A[b+bb][(ii)*K+k] * B[b+bb][(k)*N+N-1];
            //     }
            //     CONSTEXPR_FOR_END
            // }
            // CONSTEXPR_FOR_END
        }
#define STORE_C(id) C[b + id / 4][id % 4 * N + N - 1] = C##id;
        STORE_C(0);
        STORE_C(1);
        STORE_C(2);
        STORE_C(3);
        STORE_C(4);
        STORE_C(5);
        STORE_C(6);
        STORE_C(7);
        STORE_C(8);
        STORE_C(9);
        STORE_C(10);
        STORE_C(11);
        STORE_C(12);
        STORE_C(13);
        STORE_C(14);
        STORE_C(15);
#undef STORE_C
        // CONSTEXPR_FOR_END

        CONSTEXPR_FOR_BEGIN(g, 2){CONSTEXPR_FOR_BEGIN(i, 4){svbool_t ptrue = svptrue_b64();
        svbool_t p0 = svwhilelt_b64(0, 4);
        svbool_t p2 = svnot_z(ptrue, p0);
        svfloat64_t vc0 = svread_hor_za64_m(svundef_f64(), ptrue, g * 3, i);
        svfloat64_t vc1 = svread_hor_za64_m(svundef_f64(), ptrue, g * 3 + 1, i);
        svfloat64_t vc2 = svread_hor_za64_m(svundef_f64(), ptrue, g * 3 + 1, i + 4);
        svfloat64_t vc3 = svread_hor_za64_m(svundef_f64(), ptrue, g * 3 + 2, i + 4);
        svst1(ptrue, &C[b + g * 2][i * N], vc0);
        svst1(p0, &C[b + g * 2][i * N + 8], vc1);
        svst1(svnot_z(ptrue, p0), &C[b + g * 2 + 1][i * N + 8 - 4], vc2);
        svst1(ptrue, &C[b + g * 2 + 1][i * N], vc3);
    }
    CONSTEXPR_FOR_END
}
CONSTEXPR_FOR_END
}
}
__arm_new("za") void batch_dgemm_4x12xk_8p4_p4_b4_tn(
    int64_t batch, double **A, double **B, double **C, int64_t _M, int64_t _N, int64_t _K) __arm_streaming
{
    constexpr int M = 4;
    constexpr int N = 12;
    int K = _K;
    for (int64_t b = 0; b < batch; b += 4)
    {
        svzero_za();

        svbool_t ptrue = svptrue_b64();
        svbool_t p0 = svwhilelt_b64(0, 4);
        svbool_t p2 = svnot_z(ptrue, p0);
        for (int k = 0; k < K; k += 1)
        // CONSTEXPR_FOR_BEGIN(k, K)
        {
            svfloat64_t a00 = svld1(ptrue, &A[b + 0][k * M]);
            svfloat64_t b00 = svld1(ptrue, &B[b + 0][k * N]);
            svfloat64_t b01 = svld1(p0, &B[b + 0][k * N + 8]);

            svfloat64_t a10 = svld1(p2, &A[b + 1][k * M - 4]);
            svfloat64_t b10 = svld1(ptrue, &B[b + 1][k * N]);
            svfloat64_t b11 = svld1(p2, &B[b + 1][k * N + 8] - 4);

            svfloat64_t a0 = svsel(p0, a00, a10);
            svfloat64_t b1 = svsel(p0, b01, b11);

            svfloat64_t a20 = svld1(ptrue, &A[b + 2][k * M]);
            svfloat64_t b20 = svld1(ptrue, &B[b + 2][k * N]);
            svfloat64_t b21 = svld1(p0, &B[b + 2][k * N + 8]);

            svfloat64_t a30 = svld1(p2, &A[b + 3][k * M - 4]);
            svfloat64_t b30 = svld1(ptrue, &B[b + 3][k * N]);
            svfloat64_t b31 = svld1(p2, &B[b + 3][k * N + 8] - 4);

            svfloat64_t a4 = svsel(p0, a20, a30);
            svfloat64_t b4 = svsel(p0, b21, b31);

            svmopa_za64_m(0, ptrue, ptrue, a00, b00);
            svmopa_za64_m(1, ptrue, ptrue, a0, b1);
            svmopa_za64_m(2, ptrue, ptrue, a10, b10);

            svmopa_za64_m(3, ptrue, ptrue, a20, b20);
            svmopa_za64_m(4, ptrue, ptrue, a4, b4);
            svmopa_za64_m(5, ptrue, ptrue, a30, b30);
        }
        // CONSTEXPR_FOR_END

        CONSTEXPR_FOR_BEGIN(g, 2){CONSTEXPR_FOR_BEGIN(i, 4){svbool_t ptrue = svptrue_b64();
        svbool_t p0 = svwhilelt_b64(0, 4);
        svbool_t p2 = svnot_z(ptrue, p0);
        svfloat64_t vc0 = svread_hor_za64_m(svundef_f64(), ptrue, g * 3, i);
        svfloat64_t vc1 = svread_hor_za64_m(svundef_f64(), ptrue, g * 3 + 1, i);
        svfloat64_t vc2 = svread_hor_za64_m(svundef_f64(), ptrue, g * 3 + 1, i + 4);
        svfloat64_t vc3 = svread_hor_za64_m(svundef_f64(), ptrue, g * 3 + 2, i + 4);
        svst1(ptrue, &C[b + g * 2][i * N], vc0);
        svst1(p0, &C[b + g * 2][i * N + 8], vc1);
        svst1(svnot_z(ptrue, p0), &C[b + g * 2 + 1][i * N + 8 - 4], vc2);
        svst1(ptrue, &C[b + g * 2 + 1][i * N], vc3);
    }
    CONSTEXPR_FOR_END
}
CONSTEXPR_FOR_END
}
}
__arm_new("za") void batch_dgemm_4x13xk_8p5_p3_b3_tn(
    int64_t batch, double **A, double **B, double **C, int64_t _M, int64_t _N, int64_t _K) __arm_streaming
{
    constexpr int M = 4;
    constexpr int N = 13;
    int K = _K;
    for (int64_t b = 0; b < batch; b += 3)
    {
        svzero_za();

        svbool_t ptrue = svptrue_b64();
        svbool_t p0 = svwhilelt_b64(0, 5);
        svbool_t p2 = svnot_z(ptrue, p0);
        svbool_t pa0 = svwhilelt_b64(0, 4);
        svbool_t pa2 = svnot_z(ptrue, pa0);
        for (int k = 0; k < K; k += 1)
        // CONSTEXPR_FOR_BEGIN(k, K)
        {
            svfloat64_t a00 = svld1(ptrue, &A[b + 0][k * M]);
            svfloat64_t b00 = svld1(ptrue, &B[b + 0][k * N]);
            svfloat64_t b01 = svld1(p0, &B[b + 0][k * N + 8]);

            svfloat64_t a20 = svld1(pa2, &A[b + 2][k * M - 4]);

            svfloat64_t a0 = svsel(pa0, a00, a20);

            svfloat64_t b21 = svld1(p2, &B[b + 2][k * N + 3]);
            svfloat64_t b22 = svld1(p2, &B[b + 2][k * N + 6]);
            svfloat64_t b1 = svsel(p0, b01, b21);

            svfloat64_t a10 = svld1(ptrue, &A[b + 1][k * M]);
            svfloat64_t b10 = svld1(ptrue, &B[b + 1][k * N]);
            svfloat64_t b11 = svld1(p0, &B[b + 1][k * N + 8]);
            svfloat64_t b3 = svsel(p0, b11, b22);

            svfloat64_t b20 = svld1(ptrue, &B[b + 2][k * N]);

            svfloat64_t a2 = svsel(pa0, a10, a20);

            svmopa_za64_m(0, ptrue, ptrue, a00, b00);
            svmopa_za64_m(1, ptrue, ptrue, a0, b1);
            svmopa_za64_m(2, ptrue, ptrue, a10, b10);
            svmopa_za64_m(3, ptrue, ptrue, a2, b3);
            svmopa_za64_m(4, ptrue, ptrue, a2, b20);
        }
        // CONSTEXPR_FOR_END

        CONSTEXPR_FOR_BEGIN(za, 2){CONSTEXPR_FOR_BEGIN(i, 4){svbool_t ptrue = svptrue_b64();
        svbool_t p0 = svwhilelt_b64(0, 5);
        svfloat64_t vc0 = svread_hor_za64_m(svundef_f64(), ptrue, za * 2, i);
        svfloat64_t vc1 = svread_hor_za64_m(svundef_f64(), ptrue, za * 2 + 1, i);
        svfloat64_t vc2 = svread_hor_za64_m(svundef_f64(), ptrue, za * 2 + 1, i + 4);
        svst1(ptrue, &C[b + za][i * N], vc0);
        svst1(p0, &C[b + za][i * N + 8], vc1);
        svst1(svnot_z(ptrue, p0), &C[b + 2][i * N + 8 + za * 3 - 5], vc2);
    }
    CONSTEXPR_FOR_END
}
CONSTEXPR_FOR_END

CONSTEXPR_FOR_BEGIN(i, 4)
{
    svbool_t ptrue = svptrue_b64();
    svbool_t p0 = svwhilelt_b64(0, 5);
    svfloat64_t vc0 = svread_hor_za64_m(svundef_f64(), ptrue, 4, i + 4);
    svst1(ptrue, &C[b + 2][i * N], vc0);
}
CONSTEXPR_FOR_END
}
}

__arm_new("za") void batch_dgemm_4x13xk_8p5_tn(
    int64_t batch, double **A, double **B, double **C, int64_t _M, int64_t _N, int64_t _K) __arm_streaming
{
    constexpr int M = 4;
    constexpr int N = 13;
    // constexpr int K = 32;
    int K = _K;
    for (int64_t b = 0; b < batch; b += 2)
    {
        svzero_za();

        svbool_t ptrue = svptrue_b64();
        svbool_t p0 = svwhilelt_b64(0, 5);
        // for (int k = 0; k < K; k += 2)
        for (int k = 0; k < K; k += 1)
        // CONSTEXPR_FOR_BEGIN(k, K)
        {
            svbool_t ptrue = svptrue_b64();
            svbool_t p0 = svwhilelt_b64(0, 5);
            svfloat64_t a00 = svld1(ptrue, &A[b + 0][k * M]);
            svfloat64_t b00 = svld1(ptrue, &B[b + 0][k * N]);
            svfloat64_t b01 = svld1(p0, &B[b + 0][k * N + 8]);

            svfloat64_t a10 = svld1(ptrue, &A[b + 1][k * M]);
            svfloat64_t b10 = svld1(ptrue, &B[b + 1][k * N]);
            svfloat64_t b11 = svld1(p0, &B[b + 1][k * N + 8]);

            // svfloat64_t a01 = svld1(ptrue, &A[b+0][(k+1)*M]);
            // svfloat64_t b02 = svld1(ptrue, &B[b+0][(k+1)*N]);
            // svfloat64_t b03 = svld1(p0, &B[b+0][(k+1)*N + 8]);

            // svfloat64_t a11 = svld1(ptrue, &A[b+1][(1+k)*M]);
            // svfloat64_t b12 = svld1(ptrue, &B[b+1][(1+k)*N]);
            // svfloat64_t b13 = svld1(p0, &B[b+1][(1+k)*N + 8]);

            svmopa_za64_m(0, ptrue, ptrue, a00, b00);
            svmopa_za64_m(1, ptrue, ptrue, a00, b01);
            svmopa_za64_m(2, ptrue, ptrue, a10, b10);
            svmopa_za64_m(3, ptrue, ptrue, a10, b11);
            // svmopa_za64_m(0, ptrue, ptrue, a01, b02);
            // svmopa_za64_m(1, ptrue, ptrue, a01, b03);
            // svmopa_za64_m(2, ptrue, ptrue, a11, b12);
            // svmopa_za64_m(3, ptrue, ptrue, a11, b13);
        }
        // CONSTEXPR_FOR_END

        CONSTEXPR_FOR_BEGIN(za, 2){CONSTEXPR_FOR_BEGIN(i, 4){svbool_t ptrue = svptrue_b64();
        svbool_t p0 = svwhilelt_b64(0, 5);
        svfloat64_t vc0 = svread_hor_za64_m(svundef_f64(), ptrue, za * 2, i);
        svfloat64_t vc1 = svread_hor_za64_m(svundef_f64(), ptrue, za * 2 + 1, i);
        svst1(ptrue, &C[b + za][i * N], vc0);
        svst1(p0, &C[b + za][i * N + 8], vc1);
    }
    CONSTEXPR_FOR_END
}
CONSTEXPR_FOR_END
}
}

#define BUILD_FMLA_B(v0, v1, v2, v3, k)                                                                                \
    svzip1_f64(svzip1_f64(svdup_lane(v0, k), svdup_lane(v1, k)), svzip1_f64(svdup_lane(v2, k), svdup_lane(v3, k)))

__arm_new("za") void batch_dgemm_9x9xk_fmla_b4_tn(
    int64_t batch, double **A, double **B, double **C, int64_t _M, int64_t _N, int64_t K) __arm_streaming
{
    constexpr int M = 9;
    constexpr int N = 9;

    for (int64_t b = 0; b < batch; b += 4)
    {
        svzero_za();
        svbool_t ptrue = svptrue_b64();
        svbool_t p0 = svwhilelt_b64(0, 1);
        svbool_t p1 = svnot_z(ptrue, p1);
        double C_last0 = 0;
        double C_last1 = 0;
        double C_last2 = 0;
        double C_last3 = 0;
        // #pragma unroll(4)
        for (int k = 0; k < K; k += 1)
        {
            CONSTEXPR_FOR_BEGIN(za, 4, &ptrue, &p0, &p1, &C_last0, &C_last1, &C_last2, &C_last3)
            {
                svfloat64_t a0 = svld1(ptrue, &A[b + za][k * M]);
                double a1_scalar = A[b + za][k * M + M - 1];
                svfloat64_t b0 = svld1(ptrue, &B[b + za][k * N]);
                double b1_scalar = B[b + za][k * N + N - 1];
                svmopa_za64_m(za * 2, ptrue, ptrue, a0, b0);
                // svfloat64_t a1 = svdup_f64_m(svdup_f64(a1_scalar), p1, b1_scalar);
                // svmla_za64_vg1x4(za*2+1, svcreate4(a0, b0, svundef_f64(), svundef_f64()), a1);
                svmla_za64_vg1x2(za * 2 + 1, svcreate2(a0, b0), svcreate2(svdup_f64(b1_scalar), svdup_f64(a1_scalar)));
                // C[b+za][N*K-1] += a1_scalar * b1_scalar;
                selectI<za>(C_last0, C_last1, C_last2, C_last3) += a1_scalar * b1_scalar;
            }
            CONSTEXPR_FOR_END
        }
        CONSTEXPR_FOR_BEGIN(za, 4) { C[b + za][N * M - 1] = selectI<za>(C_last0, C_last1, C_last2, C_last3); }
        CONSTEXPR_FOR_END

        CONSTEXPR_FOR_BEGIN(bb, 4, &ptrue, &p0)
        {
            svfloat64_t vc2 = svread_hor_za64_m(svundef_f64(), ptrue, bb * 2 + 1, 0);
            CONSTEXPR_FOR_BEGIN(i, 8, &ptrue, &p0, &vc2)
            {
                svfloat64_t vc0 = svread_hor_za64_m(svundef_f64(), ptrue, bb * 2, i);
                svst1(ptrue, &C[b + bb][i * N], vc0);
                svst1(p0, &C[b + bb][i * N + 8], svdup_lane(vc2, i));
            }
            CONSTEXPR_FOR_END
            svfloat64_t vc1 = svread_hor_za64_m(svundef_f64(), ptrue, bb * 2 + 1, 4);
            svst1(ptrue, &C[b + bb][8 * N], vc1);
        }
        CONSTEXPR_FOR_END
    }
}
__arm_new("za") void batch_dgemm_2x8xk_fmla_b4_tn(
    int64_t batch, double **A, double **B, double **C, int64_t _M, int64_t _N, int64_t K) __arm_streaming
{
    constexpr int M = 2;
    constexpr int N = 8;

    for (int64_t b = 0; b < batch; b += 8)
    {
        svzero_za();
        svbool_t ptrue = svptrue_b64();
        svbool_t p0 = svwhilelt_b64(0, 2);
        svbool_t p1 = svnot_z(svwhilelt_b64(0, 4), p0);
        svbool_t p2 = svnot_z(svwhilelt_b64(0, 6), svwhilelt_b64(0, 4));
        svbool_t p3 = svnot_z(ptrue, svwhilelt_b64(0, 6));
        // #pragma unroll(4)
        for (int k = 0; k < K; k += 1)
        {
            svfloat64_t a0 = svld1(p0, &A[b + 0][k * M]);
            svfloat64_t a1 = svld1(p1, &A[b + 1][k * M - 2]);
            svfloat64_t a2 = svld1(p2, &A[b + 2][k * M - 4]);
            svfloat64_t a3 = svld1(p3, &A[b + 3][k * M - 6]);
            svfloat64_t a = svsel(p0, a0, svsel(p1, a1, svsel(p2, a2, a3)));
            // svfloat64_t a = a0;
            svfloat64_t b0 = svld1(ptrue, &B[b + 0][k * N]);
            svfloat64_t b1 = svld1(ptrue, &B[b + 1][k * N]);
            svfloat64_t b2 = svld1(ptrue, &B[b + 2][k * N]);
            svfloat64_t b3 = svld1(ptrue, &B[b + 3][k * N]);

            svwrite_ver_za64_vg4(2, 0, svcreate4(b0, b0, b1, b1));
            svwrite_ver_za64_vg4(2, 4, svcreate4(b2, b2, b3, b3));
            svfloat64_t a0_next = svld1(p0, &A[b + 0 + 4][k * M]);
            svfloat64_t a1_next = svld1(p1, &A[b + 1 + 4][k * M - 2]);
            svfloat64_t a2_next = svld1(p2, &A[b + 2 + 4][k * M - 4]);
            svfloat64_t a3_next = svld1(p3, &A[b + 3 + 4][k * M - 6]);
            svfloat64_t a_next = svsel(p0, a0_next, svsel(p1, a1_next, svsel(p2, a2_next, a3_next)));
            svfloat64_t b0_next = svld1(ptrue, &B[b + 0 + 4][k * N]);
            svfloat64_t b1_next = svld1(ptrue, &B[b + 1 + 4][k * N]);
            svfloat64_t b2_next = svld1(ptrue, &B[b + 2 + 4][k * N]);
            svfloat64_t b3_next = svld1(ptrue, &B[b + 3 + 4][k * N]);

            // svfloat64_t vb0 = BUILD_FMLA_B(b0, b1, b2, b3, 0);
            // svfloat64_t vb1 = BUILD_FMLA_B(b0, b1, b2, b3, 1);
            // svfloat64_t vb2 = BUILD_FMLA_B(b0, b1, b2, b3, 2);
            // svfloat64_t vb3 = BUILD_FMLA_B(b0, b1, b2, b3, 3);
            // svfloat64_t vb4 = BUILD_FMLA_B(b0, b1, b2, b3, 4);
            // svfloat64_t vb5 = BUILD_FMLA_B(b0, b1, b2, b3, 5);
            // svfloat64_t vb6 = BUILD_FMLA_B(b0, b1, b2, b3, 6);
            // svfloat64_t vb7 = BUILD_FMLA_B(b0, b1, b2, b3, 7);
            svwrite_ver_za64_vg4(3, 0, svcreate4(b0_next, b0_next, b1_next, b1_next));
            svwrite_ver_za64_vg4(3, 4, svcreate4(b2_next, b2_next, b3_next, b3_next));
            svfloat64x4_t vec_b0 = svread_hor_za64_f64_vg4(2, 0);
            svfloat64x4_t vec_b1 = svread_hor_za64_f64_vg4(2, 4);

            svfloat64x4_t vec_b0_next = svread_hor_za64_f64_vg4(3, 0);
            svfloat64x4_t vec_b1_next = svread_hor_za64_f64_vg4(3, 4);
            // svfloat64_t vb0 = svread_hor_za64_m(svundef_f64(), ptrue, 2, 0);
            // svfloat64_t vb1 = svread_hor_za64_m(svundef_f64(), ptrue, 2, 1);
            // svfloat64_t vb2 = svread_hor_za64_m(svundef_f64(), ptrue, 2, 2);
            svmla_za64_vg1x4(0, vec_b0, a);
            svmla_za64_vg1x4(1, vec_b1, a);
            svmla_za64_vg1x4(4, vec_b0_next, a_next);
            svmla_za64_vg1x4(5, vec_b1_next, a_next);

            // svfloat64x4_t b_vec = svcreate4(b0, b1, b2, b3);
            // svmla_za64_vg1x4(0, b_vec, a);
            // svmla_za64_vg1x4(1, b_vec, a);
            // svfloat64_t a0_2 = svld1(p0, &A[b+0][(k+1)*M]);
            // svfloat64_t b01 = svld1(svwhilelt_b64(0, 4), &B[b+0][(1+k)*N]);
            // svfloat64_t b11 = svld1(svwhilelt_b64(0, 4), &B[b+1][(1+k)*N]);
            // svfloat64_t b21 = svld1(svwhilelt_b64(0, 4), &B[b+2][(1+k)*N]);
            // svfloat64_t b31 = svld1(svwhilelt_b64(0, 4), &B[b+3][(1+k)*N]);
            // svmla_za64_vg1x4(0, svcreate4(b01, b11, b21, b31), a0_2);
            // CONSTEXPR_FOR_BEGIN(lane, 2, &ptrue, &p0, &p1, &p2, &p3, &a)
            // {
            // svfloat64_t b0 = svld1(ptrue, &B[b+0][k*N]);
            // svfloat64_t b1 = svld1(ptrue, &B[b+1][k*N]);
            // svfloat64_t b2 = svld1(ptrue, &B[b+2][k*N]);
            // svfloat64_t b3 = svld1(ptrue, &B[b+3][k*N]);
            //     // svfloat64_t b0 = svdup_f64_m(svundef_f64(), p0, B[b+0][k*N + lane]);
            //     // b0 = svdup_f64_m(b0, p1, B[b+1][k*N+lane]);
            //     // // b0 = svdup_f64_m(b0, p2, B[b+2][k*N+lane]);
            //     // // b0 = svdup_f64_m(b0, p3, B[b+3][k*N+lane]);

            //     // svfloat64_t b1 = svdup_f64_m(svundef_f64(), p0, B[b+0][k*N + 2+lane]);
            //     // b1 = svdup_f64_m(b1, p1, B[b+1][k*N + 2+lane]);
            //     // // b1 = svdup_f64_m(b1, p2, B[b+2][k*N + 2+lane]);
            //     // // b1 = svdup_f64_m(b1, p3, B[b+3][k*N + 2+lane]);

            //     // svfloat64_t b2 = svdup_f64_m(svundef_f64(), p0, B[b+0][k*N + 4+lane]);
            //     // b2 = svdup_f64_m(b2, p1, B[b+1][k*N + 4+lane]);
            //     // // b2 = svdup_f64_m(b2, p2, B[b+2][k*N + 4+lane]);
            //     // // b2 = svdup_f64_m(b2, p3, B[b+3][k*N + 4+lane]);

            //     // svfloat64_t b3 = svdup_f64_m(svundef_f64(), p0, B[b+0][k*N + 6+lane]);
            //     // b3 = svdup_f64_m(b3, p1, B[b+1][k*N + 6+lane]);
            //     // // b3 = svdup_f64_m(b3, p2, B[b+2][k*N + 6+lane]);
            //     // // b3 = svdup_f64_m(b3, p3, B[b+3][k*N + 6+lane]);

            //     svfloat64x4_t b_vec = svcreate4(b0, b1, b2, b3);
            //     svmla_za64_vg1x4(lane, b_vec, a);
            //     // svmla_za64_vg1x4(lane + 2, b_vec, a);
            //     // svmla_za64_vg1x4(lane + 4, b_vec, a);
            //     // svmla_za64_vg1x4(lane + 6, b_vec, a);
            // }
            // CONSTEXPR_FOR_END
        }

        CONSTEXPR_FOR_BEGIN(za, 1){CONSTEXPR_FOR_BEGIN(i, 4){svbool_t ptrue = svptrue_b64();
        svfloat64_t vc0 = svread_ver_za64_m(svundef_f64(), ptrue, za, i * 2);
        svfloat64_t vc1 = svread_ver_za64_m(svundef_f64(), ptrue, za, i * 2 + 1);
        svfloat64_t vc2 = svread_ver_za64_m(svundef_f64(), ptrue, za + 4, i * 2);
        svfloat64_t vc3 = svread_ver_za64_m(svundef_f64(), ptrue, za + 4, i * 2 + 1);
        svst1(ptrue, &C[b + za][i * N], vc0);
        svst1(ptrue, &C[b + za + 1][i * N], vc1);
        svst1(ptrue, &C[b + za + 2][i * N], vc2);
        svst1(ptrue, &C[b + za + 3][i * N + 2], vc3);
    }
    CONSTEXPR_FOR_END
}
CONSTEXPR_FOR_END
}
}
__arm_new("za") void batch_dgemm_5x5xk_5_tn(
    int64_t batch, double **A, double **B, double **C, int64_t M, int64_t N, int64_t K) __arm_streaming
{
    for (int64_t b = 0; b < batch; b += 5)
    {
        svzero_za();
        svbool_t ptrue = svptrue_b64();                     // p6
        svbool_t p0 = svwhilelt_b64(0, 5);                  // p7
        svbool_t p40 = svnot_z(ptrue, p0);                  // p5
        svbool_t p41 = svnot_z(ptrue, svwhilelt_b64(0, 6)); // p3
        for (int k = 0; k < K; ++k)
        {
            // __asm__ volatile (
            //     /* =========================
            //      * Load A
            //      * ========================= */
            //     "ld1d z0.d,  p7/z,  [%[A0]]            \n" // a00
            //     "ld1d z1.d,  p5/z, [%[A4m5]]           \n" // a40

            //     "ld1d z2.d,  p7/z,  [%[A1]]            \n" // a10
            //     "ld1d z3.d,  p3/z, [%[A4m3]]           \n" // a41

            //     "ld1d z4.d,  p7/z,  [%[A2]]            \n" // a20
            //     "ld1d z5.d,  p7/z,  [%[A3]]            \n" // a30

            //     /* =========================
            //      * Load B
            //      * ========================= */
            //     "ld1d z6.d,  p7/z,  [%[B0]]            \n" // b00
            //     "ld1d z7.d,  p5/z, [%[B4m5]]           \n" // b40

            //     "ld1d z8.d,  p7/z,  [%[B1]]            \n" // b10
            //     "ld1d z9.d,  p7/z,  [%[B2]]            \n" // b20
            //     "ld1d z10.d, p3/z, [%[B4m3]]           \n" // b41
            //     "ld1d z11.d, p7/z,  [%[B3]]            \n" // b30

            //     /* =========================
            //      * svsel
            //      * ========================= */
            //     "sel z12.d, p7, z0.d,  z1.d             \n" // a0
            //     "sel z13.d, p7, z6.d,  z7.d             \n" // b0

            //     "sel z14.d, p7, z2.d,  z3.d             \n" // a1
            //     "sel z15.d, p7, z8.d,  z7.d             \n" // b1

            //     "sel z16.d, p7, z4.d,  z1.d             \n" // a2
            //     "sel z17.d, p7, z9.d,  z10.d            \n" // b2

            //     "sel z18.d, p7, z5.d,  z3.d             \n" // a3
            //     "sel z19.d, p7, z11.d, z10.d            \n" // b3

            //     /* =========================
            //      * MOPA (ZA accumulators)
            //      * ========================= */
            //     "fmopa za0.d, p6/m, p6/m, z12.d, z13.d \n"
            //     "fmopa za1.d, p6/m, p6/m, z14.d, z15.d \n"
            //     "fmopa za2.d, p6/m, p6/m, z16.d, z17.d \n"
            //     "fmopa za3.d, p6/m, p6/m, z18.d, z19.d \n"

            //     :
            //     : [A0]   "r"(&A[b+0][k*M]),
            //       [A1]   "r"(&A[b+1][k*M]),
            //       [A2]   "r"(&A[b+2][k*M]),
            //       [A3]   "r"(&A[b+3][k*M]),
            //       [A4m5] "r"(&A[b+4][k*M - 5]),
            //       [A4m3] "r"(&A[b+4][k*M + 3 - 6]),

            //       [B0]   "r"(&B[b+0][k*N]),
            //       [B1]   "r"(&B[b+1][k*N]),
            //       [B2]   "r"(&B[b+2][k*N]),
            //       [B3]   "r"(&B[b+3][k*N]),
            //       [B4m5] "r"(&B[b+4][k*N - 5]),
            //       [B4m3] "r"(&B[b+4][k*N + 3 - 6])

            //     //   [p0]   "P"(p0),
            //     //   [p40]  "P"(p40),
            //     //   [p41]  "P"(p41),
            //     //   [pt]   "P"(ptrue)
            //     : "z0","z1","z2","z3","z4","z5",
            //       "z6","z7","z8","z9","z10","z11",
            //       "z12","z13","z14","z15","z16","z17","z18","z19",
            //       "memory"
            // );

            svfloat64_t a00 = svld1(p0, &A[b + 0][k * M]);
            svfloat64_t a40 = svld1(p40, &A[b + 4][k * M - 5]);
            svfloat64_t b00 = svld1(p0, &B[b + 0][k * N]);
            svfloat64_t b40 = svld1(p40, &B[b + 4][k * N - 5]);

            svfloat64_t a10 = svld1(p0, &A[b + 1][k * M]);
            svfloat64_t a41 = svld1(p41, &A[b + 4][k * M + 3 - 6]);
            svfloat64_t b10 = svld1(p0, &B[b + 1][k * N]);

            svfloat64_t a20 = svld1(p0, &A[b + 2][k * M]);
            svfloat64_t b20 = svld1(p0, &B[b + 2][k * N]);
            svfloat64_t b41 = svld1(p41, &B[b + 4][k * N + 3 - 6]);

            svfloat64_t a30 = svld1(p0, &A[b + 3][k * M]);
            svfloat64_t b30 = svld1(p0, &B[b + 3][k * N]);

            svfloat64_t a0 = svsel(p0, a00, a40);
            svfloat64_t b0 = svsel(p0, b00, b40);
            svfloat64_t a1 = svsel(p0, a10, a41); //
            svfloat64_t b1 = svsel(p0, b10, b40);
            svfloat64_t a2 = svsel(p0, a20, a40);
            svfloat64_t b2 = svsel(p0, b20, b41); //
            svfloat64_t a3 = svsel(p0, a30, a41); //
            svfloat64_t b3 = svsel(p0, b30, b41); //

            svmopa_za64_m(0, ptrue, ptrue, a0, b0);
            svmopa_za64_m(1, ptrue, ptrue, a1, b1);
            svmopa_za64_m(2, ptrue, ptrue, a2, b2);
            svmopa_za64_m(3, ptrue, ptrue, a3, b3);
        }
        CONSTEXPR_FOR_BEGIN(za, 4){CONSTEXPR_FOR_BEGIN(i, 5){svbool_t ptrue = svptrue_b64();
        svbool_t p0 = svwhilelt_b64(0, 5);
        svbool_t p40 = svnot_z(ptrue, p0);
        svbool_t p41 = svnot_z(ptrue, svwhilelt_b64(0, 6));
        svfloat64_t vc = svread_hor_za64_m(svundef_f64(), ptrue, za, i);
        svst1(p0, &C[b + za][i * N], vc);
    }
    CONSTEXPR_FOR_END
}
CONSTEXPR_FOR_END

// za = 0, batch = 4
CONSTEXPR_FOR_BEGIN(i, 3)
{
    svbool_t ptrue = svptrue_b64();
    svbool_t p0 = svwhilelt_b64(0, 5);
    svbool_t p40 = svnot_z(ptrue, p0);
    svbool_t p41 = svnot_z(ptrue, svwhilelt_b64(0, 6));
    svfloat64_t vc = svread_hor_za64_m(svundef_f64(), ptrue, 0, i + 5);
    svst1(p40, &C[b + 4][i * N - 5], vc);
}
CONSTEXPR_FOR_END

// za = 1, batch = 4
CONSTEXPR_FOR_BEGIN(i, 2)
{
    svbool_t ptrue = svptrue_b64();
    svbool_t p0 = svwhilelt_b64(0, 5);
    svbool_t p40 = svnot_z(ptrue, p0);
    svbool_t p41 = svnot_z(ptrue, svwhilelt_b64(0, 6));
    svfloat64_t vc = svread_hor_za64_m(svundef_f64(), ptrue, 1, i + 6);
    svst1(p40, &C[b + 4][(i + 3) * N - 5], vc);
}
CONSTEXPR_FOR_END

// za = 2, batch = 4
CONSTEXPR_FOR_BEGIN(i, 3)
{
    svbool_t ptrue = svptrue_b64();
    svbool_t p0 = svwhilelt_b64(0, 5);
    svbool_t p40 = svnot_z(ptrue, p0);
    svbool_t p41 = svnot_z(ptrue, svwhilelt_b64(0, 6));
    svfloat64_t vc = svread_hor_za64_m(svundef_f64(), ptrue, 2, i + 5);
    svst1(p41, &C[b + 4][(i)*N - 6 + 3], vc);
}
CONSTEXPR_FOR_END

// za = 3, batch = 4
CONSTEXPR_FOR_BEGIN(i, 2)
{
    svbool_t ptrue = svptrue_b64();
    svbool_t p0 = svwhilelt_b64(0, 5);
    svbool_t p40 = svnot_z(ptrue, p0);
    svbool_t p41 = svnot_z(ptrue, svwhilelt_b64(0, 6));
    svfloat64_t vc = svread_hor_za64_m(svundef_f64(), ptrue, 3, i + 6);
    svst1(p41, &C[b + 4][(i + 3) * N - 6 + 3], vc);
}
CONSTEXPR_FOR_END
}
}

/* ==================== 工具函数 ==================== */

static double rand_double()
{
    // return 1.;
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0; // [-1, 1]
}

static double max_abs_diff(const double *a, const double *b, int64_t size)
{
    double maxd = 0.0;
    for (int64_t i = 0; i < size; ++i)
    {
        double d = fabs(a[i] - b[i]);
        if (d > maxd)
        {
            maxd = d;
        }
    }
    return maxd;
}

void clear_matrix(double **C_test, int64_t batch, int64_t M, int64_t N, int64_t K)
{
    for (int64_t b = 0; b < batch; ++b)
    {
        memset(C_test[b], 0, sizeof(double) * M * N);
    }
}
__attribute__((optimize("O0"))) void
init_matrix(double **A, double **B, double **C_ref, double **C_test, int64_t batch, int64_t M, int64_t N, int64_t K)
{
    double *A_real = (double *)aligned_alloc(64, sizeof(double) * K * M * 2 * batch);
    double *B_real = (double *)aligned_alloc(64, sizeof(double) * K * N * 2 * batch);
    double *C_ref_real = (double *)aligned_alloc(64, sizeof(double) * N * M * 2 * batch);
    double *C_test_real = (double *)aligned_alloc(64, sizeof(double) * N * M * 2 * batch);
    for (int64_t b = 0; b < batch; ++b)
    {
        // disable_sme(buffer);
        // A[b] = (double*)aligned_alloc(64, sizeof(double) * K * M * 2); // K x M
        // B[b] = (double*)aligned_alloc(64, sizeof(double) * K * N * 2); // K x N
        // C_ref[b]  = (double*)aligned_alloc(64, sizeof(double) * M * N* 2);
        // C_test[b] = (double*)aligned_alloc(64, sizeof(double) * M * N* 2);
        A[b] = A_real + b * K * M * 2; // K x M
        B[b] = B_real + b * K * N * 2; // K x N
        C_ref[b] = C_ref_real + b * M * N * 2;
        C_test[b] = C_test_real + b * M * N * 2;
        // enable_sme(buffer);

        /* 初始化输入 */
        for (int64_t i = 0; i < K * M; ++i)
        {
            A[b][i] = rand_double();
        }
        // A[b][i] = 1.;

        for (int64_t i = 0; i < K * N; ++i)
        {
            B[b][i] = rand_double();
        }
        // B[b][i] = 1.;

        /* 输出清零（重要，防止 kernel 未完全覆盖） */
        // disable_sme(buffer);
        memset(C_ref[b], 0, sizeof(double) * M * N);
        memset(C_test[b], 0, sizeof(double) * M * N);
        // enable_sme(buffer);
    }
}

#define CHECK_GEMM(fn)                                                                                                 \
    {                                                                                                                  \
        clear_matrix(C_test, batch, M, N, K);                                                                          \
        enable_sme(buffer);                                                                                            \
        fn(batch, A, B, C_test, M, N, K);                                                                              \
        disable_sme(buffer);                                                                                           \
        int error = 0;                                                                                                 \
        for (int64_t b = 0; b < batch; ++b)                                                                            \
        {                                                                                                              \
            double diff = max_abs_diff(C_ref[b], C_test[b], M * N);                                                    \
            if (diff > 1e-9)                                                                                           \
            {                                                                                                          \
                printf("Mismatch at batch %lld, max abs diff = %.3e\n", b, diff);                                      \
                error = 1;                                                                                             \
                                                                                                                       \
                /* 打印第一个错误元素，方便 debug */                                                                   \
                for (int i = 0; i < M * N; ++i)                                                                        \
                {                                                                                                      \
                    double d = fabs(C_ref[b][i] - C_test[b][i]);                                                       \
                    if (d > 1e-9)                                                                                      \
                    {                                                                                                  \
                        printf("  C_ref[%d]=%.12f, C_test[%d]=%.12f\n", i, C_ref[b][i], i, C_test[b][i]);              \
                        break;                                                                                         \
                    }                                                                                                  \
                }                                                                                                      \
                break;                                                                                                 \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        if (!error)                                                                                                    \
            printf("[PASS] %s: batch=%ld, K=%lld\n\n", #fn, batch, K);                                                 \
        else                                                                                                           \
        {                                                                                                              \
            print_batch_gemm_output("C_ref", C_ref, batch, M, N, 100);                                                 \
            print_batch_gemm_output("C_test", C_test, batch, M, N, 100);                                               \
            printf("[ERROR] %s: batch=%lld, K=%lld\n\n", #fn, batch, K);                                               \
        }                                                                                                              \
    }

#define TEST_GEMM(fn)                                                                                                  \
    {                                                                                                                  \
        enable_sme(buffer);                                                                                            \
        for (int r = 0; r < 10; ++r)                                                                                   \
            fn(batch, A, B, C_test, M, N, K);                                                                          \
        disable_sme(buffer);                                                                                           \
        volatile auto st = now_sec();                                                                                  \
        enable_sme(buffer);                                                                                            \
        for (int r = 0; r < TIMES; ++r)                                                                                \
            fn(batch, A, B, C_test, M, N, K);                                                                          \
        disable_sme(buffer);                                                                                           \
        volatile auto et = now_sec();                                                                                  \
        volatile double time_s = et - st;                                                                              \
        double flops = (double)batch * 2.0 * M * N * K;                                                                \
        double gflops = flops / time_s / 1e9 * TIMES;                                                                  \
        std::cout << #fn << "\n";                                                                                      \
        std::cout << "Time    = " << time_s << " sec\n";                                                               \
        std::cout << "GFLOPS  = " << gflops << "\n";                                                                   \
        CHECK_GEMM(fn)                                                                                                 \
    }

/* ==================== 测试函数 ==================== */

#define TEST_4x13                                                                                                      \
    TEST_GROUP(batch_dgemm_tn,                                                                                         \
               batch_dgemm_sme_b4_tn,                                                                                  \
               batch_dgemm_4x13xk_8p5_tn,                                                                              \
               batch_dgemm_4x13xk_8p4p1_1scalar_b4_tn,                                                                 \
               batch_dgemm_4x13xk_8p4p1_1sve_b4_tn,                                                                    \
               batch_dgemm_sve_tn)

#define TEST_4x5 TEST_GROUP(batch_dgemm_tn, batch_dgemm_sme_b4_tn, batch_dgemm_sve_tn, batch_dgemm_4x5xk_4p1_1sve_b8_tn)
#define TEST_4x4 TEST_GROUP(batch_dgemm_tn, batch_dgemm_sme_b4_tn, batch_dgemm_sve_tn)
#define TEST_4x9 TEST_GROUP(batch_dgemm_tn, batch_dgemm_sme_b4_tn, batch_dgemm_4x9xk_8p1_1sve_b4_tn, batch_dgemm_sve_tn)

#define TEST_2x8 TEST_GROUP(batch_dgemm_tn, batch_dgemm_sme_b4_tn, batch_dgemm_2x8xk_fmla_b4_tn)
#define TEST_9x9 TEST_GROUP(batch_dgemm_tn, batch_dgemm_sme_b4_tn, batch_dgemm_9x9xk_fmla_b4_tn, batch_dgemm_sve_tn)
#define TEST_10x10 TEST_GROUP(batch_dgemm_tn, batch_dgemm_sme_b4_tn, batch_dgemm_sve_tn)

#define CONCAT(a, b) a##b
#define X_CONCAT(a, b) CONCAT(a, b)
#define DO_TEST(M, N) X_CONCAT(TEST_, X_CONCAT(M, X_CONCAT(x, N)))
// #define DO_TEST(a, b) CAT(TEST_, CAT(a, CAT(x, b)))
int test_batch_dgemm_5x5xk_5_tn(int64_t batch, int64_t K)
{
    char buffer[512];
// const int64_t M = 4;
// const int64_t N = 9;
#define M 10
#define N 10

    // if (batch % 5 != 0) {
    //     printf("ERROR: batch (%ld) is not a multiple of 5\n", batch);
    //     return -1;
    // }

    disable_sme(buffer);
    srand(0); // 保证可复现

    /* 分配指针数组 */
    double **A = (double **)malloc(batch * sizeof(double *));
    double **B = (double **)malloc(batch * sizeof(double *));
    double **C_ref = (double **)malloc(batch * sizeof(double *));
    double **C_test = (double **)malloc(batch * sizeof(double *));
    // enable_sme(buffer);

    /* 分配每个 batch 的矩阵 */
    init_matrix(A, B, C_ref, C_test, batch, M, N, K);

    // disable_sme(buffer);
    /* reference */
    batch_dgemm_tn(batch, A, B, C_ref, M, N, K);
    // TEST_GROUP(batch_dgemm_4x13xk_8p4p1_1scalar_b4_tn)
    DO_TEST(M, N)
    printf("---------------------\n");
    DO_TEST(M, N)
    // TEST_GROUP(batch_dgemm_sme_b4_tn)

    /* 释放内存 */
    // for (int64_t b = 0; b < batch; ++b) {
    //     free(A[b]);
    //     free(B[b]);
    //     free(C_ref[b]);
    //     free(C_test[b]);
    // }

    free(A);
    free(B);
    free(C_ref);
    free(C_test);

    // return error ? -1 : 0;
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Usage: %s <batch> <K>\n", argv[0]);
    }
    int64_t batch = atoi(argv[1]);
    int64_t K = atoi(argv[2]);
    asm volatile("smstart");
    // fn(argc, argv);
    test_batch_dgemm_5x5xk_5_tn(batch, K);
    // test();
    asm volatile("smstop");
}