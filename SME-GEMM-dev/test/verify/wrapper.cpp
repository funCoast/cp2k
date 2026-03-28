#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#include <stdint.h>
#if defined(__APPLE__)
#include <pthread.h>
#endif
#include "shape.h"

#define SME_NEW_ZA
#define SME_INOUT_ZA
#define SME_STREAMING
#define SME_STREAMING_COMPAT

SME_NEW_ZA void gemm_kernel_opt(int64_t batch, FPTYPE **A, FPTYPE **B, FPTYPE **C, int, int, int K) SME_STREAMING;
using runtime_kernel_fn = void(int64_t batch, FPTYPE **A, FPTYPE **B, FPTYPE **C, int, int, int K) SME_STREAMING;

static void *load_runtime_kernel()
{
    const char *path = std::getenv("KERNEL_BIN_PATH");
    if (path == nullptr || path[0] == '\0')
    {
        return nullptr;
    }

    static void *cached_entry = nullptr;
    if (cached_entry != nullptr)
    {
        return cached_entry;
    }

    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        std::fprintf(stderr, "failed to open kernel binary: %s\n", path);
        std::abort();
    }

    input.seekg(0, std::ios::end);
    const std::streamsize size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (size <= 0)
    {
        std::fprintf(stderr, "kernel binary is empty: %s\n", path);
        std::abort();
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
    {
        std::fprintf(stderr, "failed to query page size\n");
        std::abort();
    }

    constexpr std::uint32_t kBtiC = 0xd503245fu;
    const std::size_t code_size = static_cast<std::size_t>(size) + sizeof(kBtiC);
    const std::size_t alloc_size =
        ((code_size + static_cast<std::size_t>(page_size) - 1) / static_cast<std::size_t>(page_size)) *
        static_cast<std::size_t>(page_size);
    void *buffer = MAP_FAILED;
    bool using_map_jit = false;

    // Prefer the plain RW->RX path first. It matches the object-file execution
    // model more closely and avoids Apple MAP_JIT write-protect semantics unless
    // we actually need that fallback.
    buffer = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
#if defined(__APPLE__)
    if (buffer == MAP_FAILED)
    {
        buffer = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
        using_map_jit = buffer != MAP_FAILED;
    }
#endif
    if (buffer == MAP_FAILED)
    {
        std::perror("mmap");
        std::abort();
    }

#if defined(__APPLE__)
    if (using_map_jit)
    {
        pthread_jit_write_protect_np(0);
    }
#endif
    std::memcpy(buffer, &kBtiC, sizeof(kBtiC));
    if (!input.read(reinterpret_cast<char *>(buffer) + sizeof(kBtiC), size))
    {
        std::fprintf(stderr, "failed to read kernel binary: %s\n", path);
        std::abort();
    }
#if defined(__APPLE__)
    if (using_map_jit)
    {
        pthread_jit_write_protect_np(1);
    }
#endif

    __builtin___clear_cache(reinterpret_cast<char *>(buffer), reinterpret_cast<char *>(buffer) + code_size);
    if (!using_map_jit && mprotect(buffer, alloc_size, PROT_READ | PROT_EXEC) != 0)
    {
        std::perror("mprotect");
        std::abort();
    }

    cached_entry = buffer;
    return cached_entry;
}

void gemm_kernel(int64_t batch, FPTYPE **A, FPTYPE **B, FPTYPE **C, int, int, int K) SME_INOUT_ZA SME_STREAMING_COMPAT
{
    if (void *entry = load_runtime_kernel())
    {
        auto runtime_kernel = reinterpret_cast<runtime_kernel_fn *>(entry);
        runtime_kernel(batch, A, B, C, 0, 0, K);
        return;
    }
    gemm_kernel_opt(batch, A, B, C, 0, 0, K);
}

#if !defined(__clang__) && defined(__APPLE__)
extern "C" uintptr_t _arm_tpidr2_save(void) __attribute__((weak))
{
    uintptr_t val;
    asm("mrs %0, tpidr2_el0" : "=r"(val));
    return val;
}
#endif
