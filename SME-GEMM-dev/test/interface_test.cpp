#include "interface.h"

#include <iostream>
#include <stdexcept>
#include <vector>

#if defined(__clang__) && defined(__aarch64__) && false
#define INTERFACE_TEST_SME_INOUT_ZA __arm_inout("za")
#define INTERFACE_TEST_SME_STREAMING_COMPAT __arm_streaming_compatible
#else
#define INTERFACE_TEST_SME_INOUT_ZA
#define INTERFACE_TEST_SME_STREAMING_COMPAT
#endif

namespace
{

void expect(bool cond, const char *msg)
{
    if (!cond)
    {
        throw std::runtime_error(msg);
    }
}

void run_dgemm_cache_check() INTERFACE_TEST_SME_INOUT_ZA INTERFACE_TEST_SME_STREAMING_COMPAT
{
    constexpr int m = 3;
    constexpr int n = 3;
    constexpr int k = 4;
    constexpr int batch = 8;

    std::vector<double> a(m * k * batch, 1.0);
    std::vector<double> b(k * n * batch, 2.0);
    std::vector<double> c(m * n * batch, 0.0);

    std::vector<const double *> a_ptrs(batch);
    std::vector<const double *> b_ptrs(batch);
    std::vector<double *> c_ptrs(batch);
    for (int i = 0; i < batch; ++i)
    {
        a_ptrs[i] = a.data() + i * m * k;
        b_ptrs[i] = b.data() + i * k * n;
        c_ptrs[i] = c.data() + i * m * n;
    }

    SMELT::clear_jit_cache();
    SMELT::set_cache_enabled(true);
    SMELT::set_strategy(SMELT::Strategy::AUTO);
    SMELT::set_auto_context_switch(true);

    expect(SMELT::jit_cache_size() == 0, "cache should be empty before first dgemm");
    SMELT::dgemm_batch('N', 'N', m, n, k, 8, a_ptrs.data(), b_ptrs.data(), c_ptrs.data());
    expect(SMELT::jit_cache_size() == 1, "cache size should become 1 after first dgemm");

    SMELT::set_auto_context_switch(false);
    asm volatile("smstart");
    SMELT::dgemm_batch('N', 'N', m, n, k, 8, a_ptrs.data(), b_ptrs.data(), c_ptrs.data());
    SMELT::dgemm_batch('N', 'N', m, n, k, 8, a_ptrs.data(), b_ptrs.data(), c_ptrs.data());
    asm volatile("smstop");
    expect(SMELT::jit_cache_size() == 1, "cache size should stay 1 for repeated dgemm signature");
}

} // namespace

int main() INTERFACE_TEST_SME_INOUT_ZA INTERFACE_TEST_SME_STREAMING_COMPAT
{
    try
    {
        run_dgemm_cache_check();
        std::cout << "interface_test passed" << std::endl;
        return 0;
    } catch (const std::exception &e)
    {
        std::cerr << "interface_test failed: " << e.what() << std::endl;
        return 1;
    }
}

#undef INTERFACE_TEST_SME_INOUT_ZA
#undef INTERFACE_TEST_SME_STREAMING_COMPAT
