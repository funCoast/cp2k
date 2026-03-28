
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
#include "shape.h"
bool no_check = std::getenv("NO_CHECK") ? true : false;
void gemm_kernel(int64_t batch, FPTYPE **A, FPTYPE **B, FPTYPE **C, int, int, int K);

#if !defined(__clang__) && defined(__APPLE__)
extern "C" uintptr_t _arm_tpidr2_save(void) __attribute__((weak))
{
    uintptr_t val;
    // asm("mrs %0, tpidr2_el0" : "=r"(val));
    return 0;
}
#endif
#define TEST_GROUP(...) FOR_EACH(TEST_GEMM, __VA_ARGS__)

static inline double now_nanosec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

void print_batch_matrix(const char *name, int64_t batch_id, const FPTYPE *C, int64_t M, int64_t N)
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
    const char *name, FPTYPE **C, int64_t batch, int64_t M, int64_t N, int64_t max_batch_to_print)
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

static int benchmark_times()
{
    const char *override = std::getenv("TIMES_OVERRIDE");
    if (!override)
    {
        return TIMES;
    }
    int value = std::atoi(override);
    return value > 0 ? value : TIMES;
}

// 随机初始化矩阵
void rand_fill(FPTYPE *p, int64_t size)
{
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<FPTYPE> dist(-1.0, 1.0);
    for (int64_t i = 0; i < size; i++)
    {
        p[i] = dist(rng);
    }
}

#include <stdint.h>

__attribute__((noinline)) __attribute__((target("+nosve+nosme"))) void
batch_dgemm_nt(int64_t batch, FPTYPE **A, FPTYPE **B, FPTYPE **C, int64_t M, int64_t N, int64_t K)
{
    asm volatile("smstop");
    for (int64_t b = 0; b < batch; ++b)
    {
        FPTYPE *Ab = A[b]; // K x M
        FPTYPE *Bb = B[b]; // K x N
        FPTYPE *Cb = C[b]; // M x N

        for (int64_t i = 0; i < M; ++i)
        {
            for (int64_t j = 0; j < N; ++j)
            {
                FPTYPE sum = 0.0;
                for (int64_t k = 0; k < K; ++k)
                {
                    // A^T(i, k) = A(k, i)
                    sum += Ab[k + i * K] * Bb[k + j * K];
                }
                Cb[i * N + j] = sum;
            }
        }
    }
}

__attribute__((noinline)) __attribute__((target("+nosve+nosme"))) void
batch_dgemm_tt(int64_t batch, FPTYPE **A, FPTYPE **B, FPTYPE **C, int64_t M, int64_t N, int64_t K)
{
    asm volatile("smstop");
    for (int64_t b = 0; b < batch; ++b)
    {
        FPTYPE *Ab = A[b]; // K x M
        FPTYPE *Bb = B[b]; // K x N
        FPTYPE *Cb = C[b]; // M x N

        for (int64_t i = 0; i < M; ++i)
        {
            for (int64_t j = 0; j < N; ++j)
            {
                FPTYPE sum = 0.0;
                for (int64_t k = 0; k < K; ++k)
                {
                    // A^T(i, k) = A(k, i)
                    sum += Ab[k * M + i] * Bb[k + j * K];
                }
                Cb[i * N + j] = sum;
            }
        }
    }
}
__attribute__((noinline)) __attribute__((target("+nosve+nosme"))) void
batch_dgemm_nn(int64_t batch, FPTYPE **A, FPTYPE **B, FPTYPE **C, int64_t M, int64_t N, int64_t K)
{
    asm volatile("smstop");
    for (int64_t b = 0; b < batch; ++b)
    {
        FPTYPE *Ab = A[b]; // K x M
        FPTYPE *Bb = B[b]; // K x N
        FPTYPE *Cb = C[b]; // M x N

        for (int64_t i = 0; i < M; ++i)
        {
            for (int64_t j = 0; j < N; ++j)
            {
                FPTYPE sum = 0.0;
                for (int64_t k = 0; k < K; ++k)
                {
                    // A^T(i, k) = A(k, i)
                    sum += Ab[k + i * K] * Bb[k * N + j];
                }
                Cb[i * N + j] = sum;
            }
        }
    }
}
__attribute__((noinline)) __attribute__((target("+nosve+nosme"))) void
batch_dgemm_tn(int64_t batch, FPTYPE **A, FPTYPE **B, FPTYPE **C, int64_t M, int64_t N, int64_t K)
{
    asm volatile("smstop");
    for (int64_t b = 0; b < batch; ++b)
    {
        FPTYPE *Ab = A[b]; // K x M
        FPTYPE *Bb = B[b]; // K x N
        FPTYPE *Cb = C[b]; // M x N

        for (int64_t i = 0; i < M; ++i)
        {
            for (int64_t j = 0; j < N; ++j)
            {
                FPTYPE sum = 0.0;
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

/* ==================== 工具函数 ==================== */

static FPTYPE rand_FPTYPE()
{
    // return 1.;
    return ((FPTYPE)rand() / RAND_MAX) * 2.0 - 1.0; // [-1, 1]
}

static FPTYPE max_abs_diff(const FPTYPE *a, const FPTYPE *b, int64_t size)
{
    FPTYPE maxd = 0.0;
    for (int64_t i = 0; i < size; ++i)
    {
        FPTYPE d = fabs(a[i] - b[i]);
        if (d > maxd)
        {
            maxd = d;
        }
    }
    return maxd;
}

void clear_matrix(FPTYPE **C_test, int64_t batch, int64_t M, int64_t N, int64_t K)
{
    for (int64_t b = 0; b < batch; ++b)
    {
        memset(C_test[b], 0, sizeof(FPTYPE) * M * N);
    }
}
__attribute__((optimize("O0"))) void
init_matrix(FPTYPE **A, FPTYPE **B, FPTYPE **C_ref, FPTYPE **C_test, int64_t batch, int64_t M, int64_t N, int64_t K)
{
    FPTYPE *A_real = (FPTYPE *)malloc(sizeof(FPTYPE) * K * M * 2 * batch);
    FPTYPE *B_real = (FPTYPE *)malloc(sizeof(FPTYPE) * K * N * 2 * batch);
    FPTYPE *C_ref_real = (FPTYPE *)malloc(sizeof(FPTYPE) * N * M * 2 * batch);
    FPTYPE *C_test_real = (FPTYPE *)malloc(sizeof(FPTYPE) * N * M * 2 * batch);
    for (int64_t b = 0; b < batch; ++b)
    {
        // disable_sme(buffer);
        // A[b] = (FPTYPE*)aligned_alloc(64, sizeof(FPTYPE) * K * M * 2); // K x M
        // B[b] = (FPTYPE*)aligned_alloc(64, sizeof(FPTYPE) * K * N * 2); // K x N
        // C_ref[b]  = (FPTYPE*)aligned_alloc(64, sizeof(FPTYPE) * M * N* 2);
        // C_test[b] = (FPTYPE*)aligned_alloc(64, sizeof(FPTYPE) * M * N* 2);
        A[b] = A_real + b * K * M * 2; // K x M
        B[b] = B_real + b * K * N * 2; // K x N
        C_ref[b] = C_ref_real + b * M * N * 2;
        C_test[b] = C_test_real + b * M * N * 2;
        // enable_sme(buffer);

        /* 初始化输入 */
        for (int64_t i = 0; i < K * M; ++i)
        {
            A[b][i] = rand_FPTYPE();
        }
        // A[b][i] = 1.;

        for (int64_t i = 0; i < K * N; ++i)
        {
            B[b][i] = rand_FPTYPE();
        }
        // B[b][i] = 1.;

        /* 输出清零（重要，防止 kernel 未完全覆盖） */
        // disable_sme(buffer);
        memset(C_ref[b], 0, sizeof(FPTYPE) * M * N);
        memset(C_test[b], 0, sizeof(FPTYPE) * M * N);
        // enable_sme(buffer);
    }
}

#define CHECK_GEMM(fn)                                                                                                 \
    if (!no_check && std::string(#fn) == "gemm_kernel")                                                                \
    {                                                                                                                  \
        clear_matrix(C_test, batch, M, N, K);                                                                          \
        enable_sme(buffer);                                                                                            \
        fn(batch, A, B, C_test, M, N, K);                                                                              \
        disable_sme(buffer);                                                                                           \
        int error = 0;                                                                                                 \
        for (int64_t b = 0; b < batch; ++b)                                                                            \
        {                                                                                                              \
            FPTYPE diff = max_abs_diff(C_ref[b], C_test[b], M * N);                                                    \
            if (diff / (fabs(C_ref[b][0]) + fabs(C_test[b][0]) + 1e-9) > 1e-3)                                         \
            {                                                                                                          \
                printf("Mismatch at batch %lld, max abs diff = %.3e\n", b, diff);                                      \
                error = 1;                                                                                             \
                                                                                                                       \
                /* 打印第一个错误元素，方便 debug */                                                                   \
                for (int i = 0; i < M * N; ++i)                                                                        \
                {                                                                                                      \
                    FPTYPE d = fabs(C_ref[b][i] - C_test[b][i]);                                                       \
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
            printf("[PASS] %s: batch=%lld, K=%lld\n\n", #fn, batch, K);                                                \
        else                                                                                                           \
        {                                                                                                              \
            print_batch_gemm_output("C_ref", C_ref, batch, M, N, 100);                                                 \
            print_batch_gemm_output("C_test", C_test, batch, M, N, 100);                                               \
            printf("[ERROR] %s: batch=%lld, K=%lld\n\n", #fn, batch, K);                                               \
            exit(-1);                                                                                                  \
        }                                                                                                              \
    }

#define TEST_GEMM(fn)                                                                                                  \
    {                                                                                                                  \
        int times = benchmark_times();                                                                                 \
        enable_sme(buffer);                                                                                            \
        for (int r = 0; r < 10; ++r)                                                                                   \
            fn(batch, A, B, C_test, M, N, K);                                                                          \
        disable_sme(buffer);                                                                                           \
        volatile auto st = now_nanosec();                                                                              \
        enable_sme(buffer);                                                                                            \
        for (int r = 0; r < times; ++r)                                                                                \
            fn(batch, A, B, C_test, M, N, K);                                                                          \
        disable_sme(buffer);                                                                                           \
        volatile auto et = now_nanosec();                                                                              \
        volatile double time_ns = et - st;                                                                             \
        double flops = (double)batch * 2.0 * M * N * K;                                                                \
        double gflops = flops / time_ns * TIMES;                                                                       \
        std::cout << #fn << "\n";                                                                                      \
        std::cout << "Time        = " << time_ns << " ns\n";                                                           \
        std::cout << "GFLOPS      = " << gflops << "\n";                                                               \
        std::cout << "Kernel Time = " << time_ns / TIMES << " ns\n";                                                   \
        CHECK_GEMM(fn)                                                                                                 \
    }

#define CONCAT(a, b) a##b
#define X_CONCAT(a, b) CONCAT(a, b)
#define DO_TEST(M, N) X_CONCAT(TEST_, X_CONCAT(M, X_CONCAT(x, N)))
// #define DO_TEST(a, b) CAT(TEST_, CAT(a, CAT(x, b)))

#define REF_FUNC X_CONCAT(batch_dgemm_, TRANS_TYPE)
int test_batch_dgemm(int64_t batch, int64_t K)
{
    char buffer[512];

    disable_sme(buffer);
    srand(0); // 保证可复现

    /* 分配指针数组 */
    FPTYPE **A = (FPTYPE **)malloc(batch * sizeof(FPTYPE *));
    FPTYPE **B = (FPTYPE **)malloc(batch * sizeof(FPTYPE *));
    FPTYPE **C_ref = (FPTYPE **)malloc(batch * sizeof(FPTYPE *));
    FPTYPE **C_test = (FPTYPE **)malloc(batch * sizeof(FPTYPE *));

    /* 分配每个 batch 的矩阵 */
    init_matrix(A, B, C_ref, C_test, batch, M, N, K);

    /* reference */
    REF_FUNC(batch, A, B, C_ref, M, N, K);
    TEST_GROUP(REF_FUNC, gemm_kernel)
    printf("---------------------\n");
    TEST_GROUP(REF_FUNC, gemm_kernel)

    free(A);
    free(B);
    free(C_ref);
    free(C_test);

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
    test_batch_dgemm(batch, K);
    // test();
    asm volatile("smstop");
    return 0;
}
