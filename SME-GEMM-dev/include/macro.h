#pragma once

#define CAT(a, b) CAT_I(a, b)
#define CAT_I(a, b) a##b

/* 计算参数个数 */
#define VA_COUNT(...) VA_COUNT_I(__VA_ARGS__, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#define VA_COUNT_I(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, N, ...) N
#define FOR_EACH_1(m, x) m(x)
#define FOR_EACH_2(m, x, ...) m(x) FOR_EACH_1(m, __VA_ARGS__)
#define FOR_EACH_3(m, x, ...) m(x) FOR_EACH_2(m, __VA_ARGS__)
#define FOR_EACH_4(m, x, ...) m(x) FOR_EACH_3(m, __VA_ARGS__)
#define FOR_EACH_5(m, x, ...) m(x) FOR_EACH_4(m, __VA_ARGS__)
#define FOR_EACH_6(m, x, ...) m(x) FOR_EACH_5(m, __VA_ARGS__)
#define FOR_EACH_7(m, x, ...) m(x) FOR_EACH_6(m, __VA_ARGS__)
#define FOR_EACH_8(m, x, ...) m(x) FOR_EACH_7(m, __VA_ARGS__)
#define FOR_EACH_9(m, x, ...) m(x) FOR_EACH_8(m, __VA_ARGS__)
#define FOR_EACH_10(m, x, ...) m(x) FOR_EACH_9(m, __VA_ARGS__)
#define FOR_EACH_11(m, x, ...) m(x) FOR_EACH_10(m, __VA_ARGS__)
#define FOR_EACH_12(m, x, ...) m(x) FOR_EACH_11(m, __VA_ARGS__)
#define FOR_EACH_13(m, x, ...) m(x) FOR_EACH_12(m, __VA_ARGS__)
#define FOR_EACH_14(m, x, ...) m(x) FOR_EACH_13(m, __VA_ARGS__)
#define FOR_EACH_15(m, x, ...) m(x) FOR_EACH_14(m, __VA_ARGS__)
#define FOR_EACH_16(m, x, ...) m(x) FOR_EACH_15(m, __VA_ARGS__)

#define FOR_EACH(m, ...) CAT(FOR_EACH_, VA_COUNT(__VA_ARGS__))(m, __VA_ARGS__)

#if ENABLE_MY_ASSERT
#include <cstdio>
#include <cstdlib>

#define MY_ASSERT(cond)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            std::fprintf(stderr, "ASSERT FAILED: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__);                        \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)
#else
#define MY_ASSERT(cond) ((void)0)
#endif