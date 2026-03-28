#include "interface.h"
#include <iostream>
#include <chrono>

#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <ratio>
#include <stdexcept>
#include <unordered_map>
#include <map>
#include <utility>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <pthread.h>
#endif

#include "IR.h"
#include "block2Tile.h"
#include "frontend.h"

#if defined(__clang__) && defined(__aarch64__) && false
#define SMELT_SME_INOUT_ZA __arm_inout("za")
#define SMELT_SME_STREAMING_COMPAT __arm_streaming_compatible
#else
#define SMELT_SME_INOUT_ZA
#define SMELT_SME_STREAMING_COMPAT
#endif

namespace
{

extern "C" __attribute__((naked)) void
smelt_call_runtime_kernel_entry(void *entry,
                                std::int64_t batch,
                                const double *const *a,
                                const double *const *b,
                                double *const *c,
                                int m,
                                int n,
                                int k) SMELT_SME_INOUT_ZA SMELT_SME_STREAMING_COMPAT
{
    asm volatile("mov x16, x0\n"
                 "mov x0, x1\n"
                 "mov x1, x2\n"
                 "mov x2, x3\n"
                 "mov x3, x4\n"
                 "mov x4, x5\n"
                 "mov x5, x6\n"
                 "mov x6, x7\n"
                 "br x16\n");
}

extern "C" __attribute__((naked)) void smelt_call_runtime_kernel_entry_f32(
    void *entry, std::int64_t batch, const float *const *a, const float *const *b, float *const *c, int m, int n, int k)
    SMELT_SME_INOUT_ZA SMELT_SME_STREAMING_COMPAT
{
    asm volatile("mov x16, x0\n"
                 "mov x0, x1\n"
                 "mov x1, x2\n"
                 "mov x2, x3\n"
                 "mov x3, x4\n"
                 "mov x4, x5\n"
                 "mov x5, x6\n"
                 "mov x6, x7\n"
                 "br x16\n");
}

void enter_sme_context() { asm volatile("smstart"); }

void exit_sme_context() { asm volatile("smstop"); }

struct ExecutableBuffer
{
    void *base = nullptr;
    std::size_t alloc_size = 0;

    ~ExecutableBuffer()
    {
        if (base != nullptr)
        {
            munmap(base, alloc_size);
        }
    }
};

struct CompiledKernel
{
    std::shared_ptr<ExecutableBuffer> buffer;
    void *entry = nullptr;
};

struct KernelKey
{
    int m = 0;
    int n = 0;
    int k = 0;
    TilePrimitiveDescriptor::DTYPE dtype = TilePrimitiveDescriptor::DTYPE_FP64;
    TilePrimitiveDescriptor::TRANS_TYPE trans = TilePrimitiveDescriptor::UNDEF;
    SMELT::Strategy strategy = SMELT::Strategy::AUTO;

    bool operator==(const KernelKey &other) const
    {
        return m == other.m && n == other.n && k == other.k && dtype == other.dtype && trans == other.trans &&
               strategy == other.strategy;
    }
    bool operator<(const KernelKey &other) const
    {
        return std::tie(m, n, k, dtype, trans, strategy) <
               std::tie(other.m, other.n, other.k, other.dtype, other.trans, other.strategy);
    }
};

struct KernelKeyHash
{
    std::size_t operator()(const KernelKey &key) const
    {
        std::size_t seed = 0;
        auto mix = [&](std::size_t v) { seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2); };
        mix(static_cast<std::size_t>(key.m));
        mix(static_cast<std::size_t>(key.n));
        mix(static_cast<std::size_t>(key.k));
        mix(static_cast<std::size_t>(key.dtype));
        mix(static_cast<std::size_t>(key.trans));
        mix(static_cast<std::size_t>(key.strategy));
        return seed;
    }
};

struct GlobalState
{
    std::mutex mu;
    SMELT::RuntimeConfig config;
    std::map<KernelKey, std::shared_ptr<CompiledKernel>> cache;
};

GlobalState &state()
{
    static GlobalState s;
    return s;
}

TilePrimitiveDescriptor::TRANS_TYPE parse_trans(char transa, char transb)
{
    const char a = static_cast<char>(std::toupper(static_cast<unsigned char>(transa)));
    const char b = static_cast<char>(std::toupper(static_cast<unsigned char>(transb)));
    if (a == 'N' && b == 'N')
    {
        return TilePrimitiveDescriptor::GEMM_NN;
    }
    if (a == 'N' && b == 'T')
    {
        return TilePrimitiveDescriptor::GEMM_NT;
    }
    if (a == 'T' && b == 'N')
    {
        return TilePrimitiveDescriptor::GEMM_TN;
    }
    if (a == 'T' && b == 'T')
    {
        return TilePrimitiveDescriptor::GEMM_TT;
    }
    throw std::invalid_argument("transa/transb must be N or T");
}

bool should_rearrange(SMELT::Strategy strategy, TilePrimitiveDescriptor::TRANS_TYPE trans_type)
{
    if (strategy == SMELT::Strategy::SCALAR || strategy == SMELT::Strategy::SVE || strategy == SMELT::Strategy::SME2 ||
        strategy == SMELT::Strategy::COSTMODEL || strategy == SMELT::Strategy::AUTO)
    {
        return false;
    }
    if (strategy == SMELT::Strategy::MOPA)
    {
        return trans_type == TilePrimitiveDescriptor::GEMM_NT;
    }
    return true;
}

void validate_strategy_layout(SMELT::Strategy strategy, TilePrimitiveDescriptor::TRANS_TYPE trans_type)
{
    if (trans_type != TilePrimitiveDescriptor::GEMM_NT)
    {
        return;
    }
    if (strategy == SMELT::Strategy::SVE || strategy == SMELT::Strategy::FUSE_SVE ||
        strategy == SMELT::Strategy::SME2 || strategy == SMELT::Strategy::FUSE_SME2)
    {
        throw std::runtime_error("GEMM_NT is not supported for SVE/SME2 strategies");
    }
}

std::pair<std::vector<TilePrimitiveDescriptor>, int> build_primitives(Frontend::Frontend &fe, SMELT::Strategy strategy)
{
    Frontend::TileGenerator generator;
    switch (strategy)
    {
    case SMELT::Strategy::AUTO:
    case SMELT::Strategy::COSTMODEL:
        return generator.build_strategy_costmodel(fe);
    case SMELT::Strategy::SCALAR:
        return generator.build_strategy_scalar(fe);
    case SMELT::Strategy::SVE:
        return generator.build_strategy_mla(fe, TilePrimitiveDescriptor::SVE_MLA);
    case SMELT::Strategy::FUSE_SVE:
        return generator.build_strategy_fuse_mla(fe, TilePrimitiveDescriptor::SVE_MLA);
    case SMELT::Strategy::MOPA:
        return generator.build_strategy_mopa(fe);
    case SMELT::Strategy::FUSE_MOPA:
        return generator.build_strategy_fuse_mopa(fe);
    case SMELT::Strategy::STRATEGY1:
        return generator.build_strategy1(fe);
    case SMELT::Strategy::SME2:
        return generator.build_strategy_mla(fe, TilePrimitiveDescriptor::SME2_MLA);
    case SMELT::Strategy::FUSE_SME2:
        return generator.build_strategy_fuse_mla(fe, TilePrimitiveDescriptor::SME2_MLA);
    }
    throw std::runtime_error("unknown strategy");
}

std::shared_ptr<ExecutableBuffer> build_executable_buffer(const std::vector<std::uint32_t> &code)
{
    if (code.empty())
    {
        throw std::runtime_error("JIT produced empty code");
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
    {
        throw std::runtime_error("failed to query page size");
    }

    constexpr std::uint32_t kBtiC = 0xd503245fu;
    const std::size_t code_size = sizeof(kBtiC) + code.size() * sizeof(std::uint32_t);
    const std::size_t alloc_size =
        ((code_size + static_cast<std::size_t>(page_size) - 1) / static_cast<std::size_t>(page_size)) *
        static_cast<std::size_t>(page_size);

    void *mapping = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    bool using_map_jit = false;
#if defined(__APPLE__)
    if (mapping == MAP_FAILED)
    {
        mapping = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
        using_map_jit = mapping != MAP_FAILED;
    }
#endif
    if (mapping == MAP_FAILED)
    {
        throw std::runtime_error("mmap failed while allocating executable buffer");
    }

#if defined(__APPLE__)
    if (using_map_jit)
    {
        pthread_jit_write_protect_np(0);
    }
#endif
    std::memcpy(mapping, &kBtiC, sizeof(kBtiC));
    std::memcpy(static_cast<char *>(mapping) + sizeof(kBtiC), code.data(), code.size() * sizeof(std::uint32_t));
#if defined(__APPLE__)
    if (using_map_jit)
    {
        pthread_jit_write_protect_np(1);
    }
#endif

    __builtin___clear_cache(static_cast<char *>(mapping), static_cast<char *>(mapping) + code_size);
    if (!using_map_jit && mprotect(mapping, alloc_size, PROT_READ | PROT_EXEC) != 0)
    {
        munmap(mapping, alloc_size);
        throw std::runtime_error("mprotect failed while finalizing executable buffer");
    }

    auto buf = std::make_shared<ExecutableBuffer>();
    buf->base = mapping;
    buf->alloc_size = alloc_size;
    return buf;
}

std::shared_ptr<CompiledKernel> compile_kernel(int m,
                                               int n,
                                               int k,
                                               TilePrimitiveDescriptor::DTYPE dtype,
                                               TilePrimitiveDescriptor::TRANS_TYPE trans,
                                               SMELT::Strategy strategy)
{
    validate_strategy_layout(strategy, trans);

    Frontend::Frontend fe(m, n, k, 1, dtype, trans);
    fe.build();

    auto [primitives, batch_per_step] = build_primitives(fe, strategy);
    IR::Function func(m, n, batch_per_step, dtype, trans);
    for (auto &tile : primitives)
    {
        func.build(tile);
    }
    func.allocate_za();
    func.kLoopMerge();
    if (should_rearrange(strategy, trans))
    {
        func.rearrange();
    }

    IR::LoweredAResult lowered = IR::LowerToA(func);
    auto executable = build_executable_buffer(lowered.binary);

    auto compiled = std::make_shared<CompiledKernel>();
    compiled->buffer = executable;
    compiled->entry = static_cast<char *>(executable->base) + sizeof(std::uint32_t);
    return compiled;
}

std::shared_ptr<CompiledKernel> get_or_compile_kernel(int m,
                                                      int n,
                                                      int k,
                                                      TilePrimitiveDescriptor::DTYPE dtype,
                                                      TilePrimitiveDescriptor::TRANS_TYPE trans,
                                                      SMELT::Strategy strategy)
{
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);

    if (!s.config.cache_enabled)
    {
        return compile_kernel(m, n, k, dtype, trans, strategy);
    }

    KernelKey key{m, n, k, dtype, trans, strategy};
    auto it = s.cache.find(key);
    if (it != s.cache.end())
    {
        return it->second;
    }

    auto compiled = compile_kernel(m, n, k, dtype, trans, strategy);
    s.cache.emplace(key, compiled);
    return compiled;
}

std::shared_ptr<CompiledKernel> get_or_compile_kernel_persistent(int m,
                                                                 int n,
                                                                 int k,
                                                                 TilePrimitiveDescriptor::DTYPE dtype,
                                                                 TilePrimitiveDescriptor::TRANS_TYPE trans,
                                                                 SMELT::Strategy strategy)
{
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);

    KernelKey key{m, n, k, dtype, trans, strategy};
    auto it = s.cache.find(key);
    if (it != s.cache.end())
    {
        return it->second;
    }

    auto compiled = compile_kernel(m, n, k, dtype, trans, strategy);
    s.cache.emplace(key, compiled);
    return compiled;
}

template <typename T>
void validate_blas_inputs(int m,
                          int n,
                          int k,
                          T alpha,
                          T beta,
                          const T *a,
                          int lda,
                          const T *b,
                          int ldb,
                          T *c,
                          int ldc,
                          char transa,
                          char transb)
{
    if (m <= 0 || n <= 0 || k <= 0)
    {
        throw std::invalid_argument("m, n, k must be positive");
    }
    if (a == nullptr || b == nullptr || c == nullptr)
    {
        throw std::invalid_argument("A/B/C pointers must be non-null");
    }
    if (alpha != static_cast<T>(1) || beta != static_cast<T>(0))
    {
        throw std::invalid_argument("currently only alpha=1 and beta=0 are supported");
    }

    const char ta = static_cast<char>(std::toupper(static_cast<unsigned char>(transa)));
    const char tb = static_cast<char>(std::toupper(static_cast<unsigned char>(transb)));

    const int expect_lda = (ta == 'N') ? m : k;
    const int expect_ldb = (tb == 'N') ? k : n;
    const int expect_ldc = n;

    if (lda != expect_lda || ldb != expect_ldb || ldc != expect_ldc)
    {
        throw std::invalid_argument("unsupported leading dimensions: JIT kernel currently expects compact layout");
    }
}

} // namespace

namespace SMELT
{

void set_runtime_config(const RuntimeConfig &config)
{
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.config = config;
}

RuntimeConfig get_runtime_config()
{
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    return s.config;
}

void set_strategy(Strategy strategy)
{
    auto cfg = get_runtime_config();
    cfg.strategy = strategy;
    set_runtime_config(cfg);
}

Strategy get_strategy() { return get_runtime_config().strategy; }

void set_cache_enabled(bool enabled)
{
    auto cfg = get_runtime_config();
    cfg.cache_enabled = enabled;
    set_runtime_config(cfg);
}

void set_auto_context_switch(bool enabled)
{
    auto cfg = get_runtime_config();
    cfg.auto_context_switch = enabled;
    set_runtime_config(cfg);
}

bool is_auto_context_switch_enabled() { return get_runtime_config().auto_context_switch; }

bool is_cache_enabled() { return get_runtime_config().cache_enabled; }

void clear_jit_cache()
{
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.cache.clear();
}

std::size_t jit_cache_size()
{
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    return s.cache.size();
}

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
           int ldc) SMELT_SME_INOUT_ZA SMELT_SME_STREAMING_COMPAT
{
    validate_blas_inputs(m, n, k, alpha, beta, a, lda, b, ldb, c, ldc, transa, transb);

    std::array<const double *, 1> a_arr{a};
    std::array<const double *, 1> b_arr{b};
    std::array<double *, 1> c_arr{c};
    dgemm_batch(transa, transb, m, n, k, 1, a_arr.data(), b_arr.data(), c_arr.data());
}

void dgemm_batch(char transa,
                 char transb,
                 int m,
                 int n,
                 int k,
                 std::int64_t batch,
                 const double *const *a_array,
                 const double *const *b_array,
                 double *const *c_array) SMELT_SME_INOUT_ZA SMELT_SME_STREAMING_COMPAT
{
    if (m <= 0 || n <= 0 || k <= 0 || batch <= 0)
    {
        throw std::invalid_argument("m, n, k, batch must be positive");
    }
    if (a_array == nullptr || b_array == nullptr || c_array == nullptr)
    {
        throw std::invalid_argument("batch pointer arrays must be non-null");
    }

    TilePrimitiveDescriptor::TRANS_TYPE trans = parse_trans(transa, transb);
    Strategy strategy = get_strategy();

    auto kernel = get_or_compile_kernel(m, n, k, TilePrimitiveDescriptor::DTYPE_FP64, trans, strategy);

    for (std::int64_t i = 0; i < batch; ++i)
    {
        if (a_array[i] == nullptr || b_array[i] == nullptr || c_array[i] == nullptr)
        {
            throw std::invalid_argument("batch element pointer must be non-null");
        }
    }
    if (is_auto_context_switch_enabled())
    {
        enter_sme_context();
    }

    smelt_call_runtime_kernel_entry(kernel->entry, batch, a_array, b_array, c_array, m, n, k);
    if (is_auto_context_switch_enabled())
    {
        exit_sme_context();
    }
}

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
           int ldc) SMELT_SME_INOUT_ZA SMELT_SME_STREAMING_COMPAT
{
    validate_blas_inputs(m, n, k, alpha, beta, a, lda, b, ldb, c, ldc, transa, transb);

    std::array<const float *, 1> a_arr{a};
    std::array<const float *, 1> b_arr{b};
    std::array<float *, 1> c_arr{c};
    sgemm_batch(transa, transb, m, n, k, 1, a_arr.data(), b_arr.data(), c_arr.data());
}

void sgemm_batch(char transa,
                 char transb,
                 int m,
                 int n,
                 int k,
                 std::int64_t batch,
                 const float *const *a_array,
                 const float *const *b_array,
                 float *const *c_array) SMELT_SME_INOUT_ZA SMELT_SME_STREAMING_COMPAT
{
    if (m <= 0 || n <= 0 || k <= 0 || batch <= 0)
    {
        throw std::invalid_argument("m, n, k, batch must be positive");
    }
    if (a_array == nullptr || b_array == nullptr || c_array == nullptr)
    {
        throw std::invalid_argument("batch pointer arrays must be non-null");
    }

    TilePrimitiveDescriptor::TRANS_TYPE trans = parse_trans(transa, transb);
    Strategy strategy = get_strategy();

    auto kernel = get_or_compile_kernel(m, n, k, TilePrimitiveDescriptor::DTYPE_FP32, trans, strategy);

    for (std::int64_t i = 0; i < batch; ++i)
    {
        if (a_array[i] == nullptr || b_array[i] == nullptr || c_array[i] == nullptr)
        {
            throw std::invalid_argument("batch element pointer must be non-null");
        }
    }
    if (is_auto_context_switch_enabled())
    {
        enter_sme_context();
    }

    smelt_call_runtime_kernel_entry_f32(kernel->entry, batch, a_array, b_array, c_array, m, n, k);
    if (is_auto_context_switch_enabled())
    {
        exit_sme_context();
    }
}

DgemmBatchKernelPtr get_dgemm_batch_kernel_ptr(char transa, char transb, int m, int n, int k, Strategy strategy)
{
    if (m <= 0 || n <= 0 || k <= 0)
    {
        throw std::invalid_argument("m, n, k must be positive");
    }

    TilePrimitiveDescriptor::TRANS_TYPE trans = parse_trans(transa, transb);
    auto kernel = get_or_compile_kernel_persistent(m, n, k, TilePrimitiveDescriptor::DTYPE_FP64, trans, strategy);
    return reinterpret_cast<DgemmBatchKernelPtr>(kernel->entry);
}

SgemmBatchKernelPtr get_sgemm_batch_kernel_ptr(char transa, char transb, int m, int n, int k, Strategy strategy)
{
    if (m <= 0 || n <= 0 || k <= 0)
    {
        throw std::invalid_argument("m, n, k must be positive");
    }

    TilePrimitiveDescriptor::TRANS_TYPE trans = parse_trans(transa, transb);
    auto kernel = get_or_compile_kernel_persistent(m, n, k, TilePrimitiveDescriptor::DTYPE_FP32, trans, strategy);
    return reinterpret_cast<SgemmBatchKernelPtr>(kernel->entry);
}

} // namespace SMELT
