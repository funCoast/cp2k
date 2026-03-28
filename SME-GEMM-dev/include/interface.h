#pragma once
#include <cstddef>
#include <cstdint>

#if defined(__clang__) && defined(__aarch64__) && false
#define SMELT_API_SME_INOUT_ZA __arm_inout("za")
#define SMELT_API_SME_STREAMING_COMPAT __arm_streaming_compatible
#else
#define SMELT_API_SME_INOUT_ZA
#define SMELT_API_SME_STREAMING_COMPAT
#endif

namespace SMELT
{

using DgemmBatchKernelPtr = void (*)(std::int64_t batch,
                                     const double *const *a_array,
                                     const double *const *b_array,
                                     double *const *c_array,
                                     int m,
                                     int n,
                                     int k);

using SgemmBatchKernelPtr = void (*)(std::int64_t batch,
                                     const float *const *a_array,
                                     const float *const *b_array,
                                     float *const *c_array,
                                     int m,
                                     int n,
                                     int k);

enum class Strategy
{
    AUTO,
    COSTMODEL,
    SCALAR,
    SVE,
    FUSE_SVE,
    MOPA,
    FUSE_MOPA,
    STRATEGY1,
    SME2,
    FUSE_SME2,
};

struct RuntimeConfig
{
    Strategy strategy = Strategy::AUTO;
    bool cache_enabled = true;
    bool auto_context_switch = false;
};

void set_runtime_config(const RuntimeConfig &config);
RuntimeConfig get_runtime_config();

void set_strategy(Strategy strategy);
Strategy get_strategy();

void set_cache_enabled(bool enabled);
bool is_cache_enabled();

void set_auto_context_switch(bool enabled);
bool is_auto_context_switch_enabled();

void clear_jit_cache();
std::size_t jit_cache_size();

// BLAS-like interface (currently supports alpha=1 and beta=0).
void dgemm(char transa,
           char transb,
           int m,
           int n,
           int k,
           double alpha,
           const double *a,
           int lda,
           const double *b,
           int ldb,
           double beta,
           double *c,
           int ldc) SMELT_API_SME_INOUT_ZA SMELT_API_SME_STREAMING_COMPAT;

// BLAS-like interface (currently supports alpha=1 and beta=0).
void sgemm(char transa,
           char transb,
           int m,
           int n,
           int k,
           float alpha,
           const float *a,
           int lda,
           const float *b,
           int ldb,
           float beta,
           float *c,
           int ldc) SMELT_API_SME_INOUT_ZA SMELT_API_SME_STREAMING_COMPAT;

// Batched GEMM interface using arrays of matrix pointers.
void dgemm_batch(char transa,
                 char transb,
                 int m,
                 int n,
                 int k,
                 std::int64_t batch,
                 const double *const *a_array,
                 const double *const *b_array,
                 double *const *c_array) SMELT_API_SME_INOUT_ZA SMELT_API_SME_STREAMING_COMPAT;

// Batched GEMM interface using arrays of matrix pointers.
void sgemm_batch(char transa,
                 char transb,
                 int m,
                 int n,
                 int k,
                 std::int64_t batch,
                 const float *const *a_array,
                 const float *const *b_array,
                 float *const *c_array) SMELT_API_SME_INOUT_ZA SMELT_API_SME_STREAMING_COMPAT;

// Return a compiled batched GEMM kernel entry for the requested configuration.
// The returned function pointer has signature: (batch, a_array, b_array, c_array, m, n, k).
DgemmBatchKernelPtr get_dgemm_batch_kernel_ptr(char transa, char transb, int m, int n, int k, Strategy strategy);

// Return a compiled batched GEMM kernel entry for the requested configuration.
// The returned function pointer has signature: (batch, a_array, b_array, c_array, m, n, k).
SgemmBatchKernelPtr get_sgemm_batch_kernel_ptr(char transa, char transb, int m, int n, int k, Strategy strategy);

} // namespace SMELT

#undef SMELT_API_SME_INOUT_ZA
#undef SMELT_API_SME_STREAMING_COMPAT