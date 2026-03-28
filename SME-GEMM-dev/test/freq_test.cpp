#include <arm_sme.h>
#include <math.h>
#include <stdio.h>
#include <arm_sve.h>
#include <iostream>
#include <chrono>
#include <sys/ioctl.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <kblas.h>

#define FIVE ONE ONE ONE ONE ONE
#define TEN FIVE FIVE
#define FIFTY TEN TEN TEN TEN TEN
#define HUNDRED FIFTY FIFTY
    using namespace std::chrono;
static int perf_event_open(struct perf_event_attr *evt_attr, pid_t pid,
                                int cpu, int group_fd, unsigned long flags)
{
    int ret;
    ret = syscall(__NR_perf_event_open, evt_attr, pid, cpu, group_fd, flags);
    return ret;
}

void __attribute__ ((noinline)) test(uint64_t cnt)
{
    float* buffer = new float[1024];
    buffer = (float*)(((uint64_t)buffer & (~0xff))+0xff);
    asm volatile (
        "smstart\n"
    );
    #define ONE  \
"        //ld1w    {z0.s}, p0/z, [%1]\n"\
"        //ld1w    {z1.s}, p0/z, [%2]\n"\
"        //ld1w    {z2.s}, p0/z, [%3]\n"\
"        //ld1w    {z3.s}, p0/z, [%4]\n"\
"        //ld1w    {z4.s}, p0/z, [%5]\n"\
"        fmopa   za0.d, p0/m, p0/m, z0.d, z1.d\n"\
"        fmopa   za1.d, p0/m, p0/m, z0.d, z2.d\n"\
"        fmopa   za2.d, p0/m, p0/m, z4.d, z5.d\n"\
"        fmopa   za3.d, p0/m, p0/m, z4.d, z6.d\n"\
         "//fmad z5.s, p0/m, z0.s, z0.s\n" \
         "//fmad z6.s, p0/m, z1.s, z1.s\n"\
         "//fmad z7.s, p0/m, z2.s, z2.s\n"\
         "//fmad z8.s, p0/m, z2.s, z2.s\n"\
         "//fmad z9.s, p0/m, z2.s, z2.s\n"\
         "//fmad z10.s, p0/m, z2.s, z2.s\n"\
         "//fmad z11.s, p0/m, z2.s, z2.s\n"\
         "//fmad z12.s, p0/m, z2.s, z2.s\n"\
         "//fmad z13.s, p0/m, z2.s, z2.s\n"\
         "//fmad z14.s, p0/m, z2.s, z2.s\n"\
         "//fmad z15.s, p0/m, z2.s, z2.s\n"\
         "//fmad z16.s, p0/m, z2.s, z2.s\n"\
         "//fmad z17.s, p0/m, z2.s, z2.s\n"\
         "//fmad z18.s, p0/m, z2.s, z2.s\n"\
         "//fmad z19.s, p0/m, z2.s, z2.s\n"\
         "//fmad z20.s, p0/m, z2.s, z2.s\n"\
         "//fmad z21.s, p0/m, z2.s, z2.s\n"\
         "//fmad z22.s, p0/m, z2.s, z2.s\n"\
        "//st1w {z5.s}, p0, [%6]\n" \
        "//st1w {z6.s}, p0, [%7]\n"

    asm volatile(
        ".align 8 \n"
        "1: \n"
        HUNDRED
        "subs %0, %0, #1 \n"
        "b.ne 1b \n"
        :
        :"r"(cnt),  "r"((float*)buffer), "r"((float*)buffer+8), "r"((float*)buffer+16), "r"((float*)(buffer+24)), "r"((float*)(buffer+32)), "r"((float*)(buffer+40)), "r"((float*)(buffer+48)), "r"(buffer+56), "r"(buffer+64)
        :"x9", "x10", "v0", "x4", "x5", "memory");
}


struct read_format
{
    uint64_t nr;
    uint64_t values[2];
};


long x12;
template<int lda, int ldb, int ldc>
void gemm_8x8x8(const float* a, const float *b, float* c)
{
    // asm(
    //     "mov    %0, x12\n"
    //     : "=r" (x12)
    // );
    asm(
        "mov    w12, 0\n"
        "ld1w   za0v.s[w12, 0], p0/z, [%0]\n"
        "ld1w   za0v.s[w12, 1], p0/z, [%1]\n"
        "ld1w   za0v.s[w12, 2], p0/z, [%2]\n"
        "ld1w   za0v.s[w12, 3], p0/z, [%3]\n"
        "mov    w12, 4\n"
        "ld1w   za0v.s[w12, 0], p0/z, [%4]\n"
        "ld1w   za0v.s[w12, 1], p0/z, [%5]\n"
        "ld1w   za0v.s[w12, 2], p0/z, [%6]\n"
        "ld1w   za0v.s[w12, 3], p0/z, [%7]\n"
        : 
        : "r" (c), "r"(c+ldc), "r"(c+ldc*2), "r"(c+ldc*3), "r"(c+ldc*4), "r"(c+ldc*5), "r"(c+ldc*6), "r"(c+ldc*7) 
        : "x12"
    );
    #pragma GCC unroll 8
    for (int i = 0; i < 8; ++i)
    {
        asm(
            "ld1w   z0.s, p0/z, [%0]\n"
            "ld1w   z1.s,  p0/z,[%1]\n"
            "fmopa  za0.s, p0/m, p0/m, z0.s, z0.s\n"
            :
            : "r"(a + i * lda), "r"(b + i * ldb)
        );
    }
    asm(
        "mov    w12, 0\n"
        "st1w   za0v.s[w12, 0], p0, [%0]\n"
        "st1w   za0v.s[w12, 1], p0, [%1]\n"
        "st1w   za0v.s[w12, 2], p0, [%2]\n"
        "st1w   za0v.s[w12, 3], p0, [%3]\n"
        "mov    w12, 4\n"
        "st1w   za0v.s[w12, 0], p0, [%4]\n"
        "st1w   za0v.s[w12, 1], p0, [%5]\n"
        "st1w   za0v.s[w12, 2], p0, [%6]\n"
        "st1w   za0v.s[w12, 3], p0, [%7]\n"
        :
        : "r" (c), "r"(c+ldc), "r"(c+ldc*2), "r"(c+ldc*3), "r"(c+ldc*4), "r"(c+ldc*5), "r"(c+ldc*6), "r"(c+ldc*7) 
        : "x12", "memory"
    );
    // asm(
    //     "mov    x12, %0\n"
    //     :
    //     : "r" (x12)
    // );
}
// template<int lda, int ldb, int ldc>
template<int lda, int ldb, int ldc>
// __arm_inout("za") 
void gemm_8x8x8_acle(const float* a, const float *b, float* c)    __arm_inout("za")  __arm_streaming
{
    svbool_t pg = svptrue_b32();
    // #pragma GCC unroll 8
    // for (int i = 0; i < 8; ++i)
    // {
    //     svld1_hor_za32(0, i, pg, c+i*ldc);
    // }

    #pragma GCC unroll 8
    for (int i = 0; i < 8; ++i)
    {
        svfloat32_t l = svld1(pg, a + i * lda);
        svfloat32_t r = svld1(pg, b + i * ldb);
        svmopa_za32_m(0, pg, pg, l, r);
    }
    // #pragma GCC unroll 8
    // for (int i = 0; i < 8; ++i)
    // {
    //     svst1_hor_za32(0, i, pg, c+i*ldc);
    // }
}
template<int lda, int ldb, int ldc>
// __arm_inout("za") 
void gemm_8x8x8_x2_acle(const float* a, const float *b, float* c)    __arm_inout("za")  __arm_streaming
{
    svbool_t pg = svptrue_b32();
    // #pragma GCC unroll 8
    // for (int i = 0; i < 8; ++i)
    // {
    //     svld1_hor_za32(0, i, pg, c+i*ldc);
    // }

    #pragma GCC unroll 8
    for (int i = 0; i < 8; ++i)
    {
        svfloat32_t l = svld1(pg, a + i * lda);
        svfloat32_t r = svld1(pg, b + i * ldb);
        svmopa_za32_m(0, pg, pg, l, r);
        svfloat32_t r2 = svld1(pg, b + i * ldb + 8);
        // svfloat32_t r3 = svld1(pg, b + i * ldb + 16);
        // svfloat32_t r4 = svld1(pg, b + i * ldb + 24);
        svmopa_za32_m(1, pg, pg, l, r2);
        // svmopa_za32_m(2, pg, pg, l, r3);
        // svmopa_za32_m(3, pg, pg, l, r4);
    }
    // #pragma GCC unroll 8
    // for (int i = 0; i < 8; ++i)
    // {
    //     svst1_hor_za32(0, i, pg, c+i*ldc);
    // }
}
template<int lda, int ldb, int ldc, int base=0>
// __arm_inout("za") 
// inline
// __attribute__((always_inline))
__attribute__((no_inline))
void gemm_8x8x8_x2_reorder_acle(const float* a, const float *b, float* c)    __arm_inout("za")  __arm_streaming
{
    svbool_t pg = svptrue_b32();
    // #pragma GCC unroll 8
    // for (int i = 0; i < 8; ++i)
    // {
    //     svld1_hor_za32(0, i, pg, c+i*ldc);
    // }

    #pragma GCC unroll 8
    for (int i = 0; i < 8; ++i)
    {
        svfloat32_t l = svld1(pg, a + i * 8);
        svfloat32_t r = svld1(pg, b + i * 16);
        svmopa_za32_m(0+base, pg, pg, l, r);
        svfloat32_t r2 = svld1(pg, b + i * 16 + 8);
        // svfloat32_t r3 = svld1(pg, b + i * ldb + 16);
        // svfloat32_t r4 = svld1(pg, b + i * ldb + 24);
        svmopa_za32_m(1+base, pg, pg, l, r2);
        // svmopa_za32_m(2, pg, pg, l, r3);
        // svmopa_za32_m(3, pg, pg, l, r4);

    }
    // #pragma GCC unroll 8
    // for (int i = 0; i < 8; ++i)
    // {
    //     svst1_hor_za32(0, i, pg, c+i*ldc);
    // }
}
// inline
// __attribute__((always_inline))
extern "C" void gemm_8x8x8_asm(const float* a, const float* b ) __arm_inout("za") __arm_streaming_compatible;
template<int M, int N, int K>
__arm_new("za")
void gemm_nn(const float* a, const float *b, float* c)  __arm_streaming
{
    asm(
        "smstart\n"
        "ptrue p0.s"
    );
    constexpr int M_tile = 256;
    constexpr int K_tile = 128;
    constexpr int N_tile = 64;

    for (int i = 0; i < M; i += M_tile)
    {
        for (int j = 0; j < K; j +=K_tile)
        {
            float A_buffer[K_tile][M_tile];
            for (int u = 0; u < M_tile; ++u)
            {
                for (int v = 0; v < K_tile; ++v)
                {
                    A_buffer[v][u] = a[(i + u) * K + j + v];
                }
            }
            for (int k = 0; k < N; k += N_tile)
            {
                float B_buffer[K_tile][N_tile];
                for (int u = 0; u < N_tile; u++)
                {
                    for (int v = 0; v < K_tile; v++)
                    {
                        B_buffer[v][u] = b[(j+v)*N + k+u];
                    }
                }
                // float C_buffer[M_tile][N_tile];
                // for (int u = 0; u < M_tile; ++u)
                // {
                //     for (int v = 0; v < N_tile; ++v)
                //     {
                //         C_buffer[u][v] = c[(i+u)*N+k+v];
                //     }
                // }
                for (int ii = 0; ii < M_tile; ii += 8)
                {
                    for (int kk = 0; kk < N_tile; kk += 8)
                    {
                        svbool_t pg = svptrue_b32();
                        #pragma GCC unroll 8
                        for (int iii = 0; iii < 8; ++iii)
                        {
                            // svld1_hor_za32(0, i, pg, (&C_buffer[ii][kk])+i*N_tile);
                            svld1_hor_za32(0, iii, pg, (c+(i+ii)*N+(k+kk))+iii*N);
                        }
                        for (int jj = 0; jj < K_tile; jj += 8)
                        {
                            // gemm_8x8x8_acle<K_tile,K_tile,N>((const float*)&A_buffer[ii][jj], b + (j+jj)  + (k+kk) * K , c + (i+ii) * N + (k +kk));
                            // gemm_8x8x8_acle<M_tile,N_tile>((const float*)&A_buffer[jj][ii], &B_buffer[jj][kk] , c + (i+ii) * N + (k +kk), N);
                            // gemm_8x8x8_acle<M_tile,N_tile, N_tile>((const float*)&A_buffer[jj][ii], &B_buffer[jj][kk] , &C_buffer[ii][kk]);
                            gemm_8x8x8_acle<M_tile,N_tile, N_tile>((const float*)&A_buffer[jj][ii], &B_buffer[jj][kk] , nullptr);
                        }
                        #pragma GCC unroll 8
                        for (int iii = 0; iii < 8; ++iii)
                        {
                            // svst1_hor_za32(0, i, pg, (&C_buffer[ii][kk])+i*N_tile);
                            svst1_hor_za32(0, iii, pg, (c+(i+ii)*N+(k+kk))+iii*N);
                        }
                    }
                }
                // for (int u = 0; u < M_tile; ++u)
                // {
                //     for (int v = 0; v < N_tile; ++v)
                //     {
                //          c[(i+u)*N+k+v] = C_buffer[u][v];
                //     }
                // }
            }
        }
    }
}
template<int M, int N, int K>
__arm_new("za")
void gemm_nn_x2(const float* a, const float *b, float* c)  __arm_streaming
{
    asm(
        "smstart\n"
        "ptrue p0.s"
    );
    constexpr int M_tile = 256;
    constexpr int K_tile = 128;
    constexpr int N_tile = 64;

    for (int i = 0; i < M; i += M_tile)
    {
        for (int j = 0; j < K; j +=K_tile)
        {
            float A_buffer[K_tile][M_tile];
            for (int v = 0; v < K_tile; ++v)
            {
                for (int u = 0; u < M_tile; ++u)
                {
                    A_buffer[v][u] = a[(i + u) * K + j + v];
                }
            }
            for (int k = 0; k < N; k += N_tile)
            {
                float B_buffer[K_tile][N_tile];
                for (int u = 0; u < N_tile; u++)
                {
                    for (int v = 0; v < K_tile; v++)
                    {
                        B_buffer[v][u] = b[(j+v)*N + k+u];
                    }
                }
                // float C_buffer[M_tile][N_tile];
                // for (int u = 0; u < M_tile; ++u)
                // {
                //     for (int v = 0; v < N_tile; ++v)
                //     {
                //         C_buffer[u][v] = c[(i+u)*N+k+v];
                //     }
                // }
                svbool_t pg = svptrue_b32();
                for (int ii = 0; ii < M_tile; ii += 8)
                {
                    // #pragma GCC unroll 2
                    for (int kk = 0; kk < N_tile; kk += 16)
                    {
                        #pragma GCC unroll 8
                        for (int iii = 0; iii < 8; ++iii)
                        {
                            // svld1_hor_za32(0, i, pg, (&C_buffer[ii][kk])+i*N_tile);
                            svld1_hor_za32(0, iii, pg, (c+(i+ii)*N+(k+kk))+iii*N);
                            svld1_hor_za32(1, iii, pg, (c+(i+ii)*N+(k+kk+8))+iii*N);
                            // svld1_hor_za32(2, iii, pg, (c+(i+ii)*N+(k+kk+16))+iii*N);
                            // svld1_hor_za32(3, iii, pg, (c+(i+ii)*N+(k+kk+24))+iii*N);
                        }
                        #pragma GCC unroll 8
                        for (int jj = 0; jj < K_tile; jj += 8)
                        {
                            // gemm_8x8x8_acle<K_tile,K_tile,N>((const float*)&A_buffer[ii][jj], b + (j+jj)  + (k+kk) * K , c + (i+ii) * N + (k +kk));
                            // gemm_8x8x8_acle<M_tile,N_tile>((const float*)&A_buffer[jj][ii], &B_buffer[jj][kk] , c + (i+ii) * N + (k +kk), N);
                            // gemm_8x8x8_acle<M_tile,N_tile, N_tile>((const float*)&A_buffer[jj][ii], &B_buffer[jj][kk] , &C_buffer[ii][kk]);
                            gemm_8x8x8_x2_acle<M_tile,N_tile, N_tile>((const float*)&A_buffer[jj][ii], &B_buffer[jj][kk] , nullptr);
                        }
                        #pragma GCC unroll 8
                        for (int iii = 0; iii < 8; ++iii)
                        {
                            // svst1_hor_za32(0, i, pg, (&C_buffer[ii][kk])+i*N_tile);
                            svst1_hor_za32(0, iii, pg, (c+(i+ii)*N+(k+kk))+iii*N);
                            svst1_hor_za32(1, iii, pg, (c+(i+ii)*N+(k+kk+8))+iii*N);
                            // svst1_hor_za32(2, iii, pg, (c+(i+ii)*N+(k+kk+16))+iii*N);
                            // svst1_hor_za32(3, iii, pg, (c+(i+ii)*N+(k+kk+24))+iii*N);
                        }
                    }
                }
                // for (int u = 0; u < M_tile; ++u)
                // {
                //     for (int v = 0; v < N_tile; ++v)
                //     {
                //          c[(i+u)*N+k+v] = C_buffer[u][v];
                //     }
                // }
            }
        }
    }
}
template<int M, int N, int K>
__arm_new("za")
void gemm_nn_x2_reorder(const float* a, const float *b, float* c)  __arm_streaming
{
    asm(
        "smstart\n"
        "ptrue p0.s"
    );
    constexpr int M_tile = 256;
    constexpr int K_tile = 128;
    constexpr int N_tile = 64;

    for (int i = 0; i < M; i += M_tile)
    {
        for (int j = 0; j < K; j +=K_tile)
        {
            float A_buffer[K_tile][M_tile];
            // for (int u = 0; u < M_tile; ++u)
            // {
            //     for (int v = 0; v < K_tile; ++v)
            //     {
            //         A_buffer[v][u] = a[(i + u) * K + j + v];
            //     }
            // }
            auto ptr = &A_buffer[0][0];

            for (int u = 0; u < M_tile; u += 8)
            {
                for (int v = 0; v < K_tile; v += 8)
                {
                    for (int p = 0; p < 8; ++p)
                    {
                        for (int q = 0; q < 8; ++q)
                        {
                            *ptr = a[(i+u+q) * K + j + v + p];
                            ptr++;
                        }
                    }
                }
            }

            for (int k = 0; k < N; k += N_tile)
            {
                float B_buffer[K_tile][N_tile];
                // for (int u = 0; u < N_tile; u++)
                // {
                //     for (int v = 0; v < K_tile; v++)
                //     {
                //         B_buffer[v][u] = b[(j+v)*N + k+u];
                //     }
                // }
                float* B_ptr = &B_buffer[0][0];
                for (int u = 0; u < N_tile; u += 16)
                {
                    for (int v = 0; v < K_tile; v += 8)
                    {
                        for (int p = 0; p < 8; ++p)
                        {
                            #pragma simd
                            for (int q = 0; q < 16; ++q)
                            {
                                // B_buffer[v][u] = b[(j+v)*N + k+u];
                                *B_ptr = b[(j+v+p)*N + k+u+q];
                                B_ptr++;
                            }
                        }
                    }
                }
                // float C_buffer[M_tile][N_tile];
                // for (int u = 0; u < M_tile; ++u)
                // {
                //     for (int v = 0; v < N_tile; ++v)
                //     {
                //         C_buffer[u][v] = c[(i+u)*N+k+v];
                //     }
                // }
                svbool_t pg = svptrue_b32();
                for (int ii = 0; ii < M_tile; ii += 8)
                {
                    // #pragma GCC unroll 2
                    for (int kk = 0; kk < N_tile; kk += 16)
                    {
                        #pragma GCC unroll 8
                        for (int iii = 0; iii < 8; ++iii)
                        {
                            // svld1_hor_za32(0, i, pg, (&C_buffer[ii][kk])+i*N_tile);
                            svld1_hor_za32(0, iii, pg, (c+(i+ii)*N+(k+kk))+iii*N);
                            svld1_hor_za32(1, iii, pg, (c+(i+ii)*N+(k+kk+8))+iii*N);
                            // svld1_hor_za32(2, iii, pg, (c+(i+ii)*N+(k+kk+16))+iii*N);
                            // svld1_hor_za32(3, iii, pg, (c+(i+ii)*N+(k+kk+24))+iii*N);
                        }
                        #pragma GCC unroll 8
                        for (int jj = 0; jj < K_tile; jj += 8)
                        {
                            // gemm_8x8x8_acle<K_tile,K_tile,N>((const float*)&A_buffer[ii][jj], b + (j+jj)  + (k+kk) * K , c + (i+ii) * N + (k +kk));
                            // gemm_8x8x8_acle<M_tile,N_tile>((const float*)&A_buffer[jj][ii], &B_buffer[jj][kk] , c + (i+ii) * N + (k +kk), N);
                            // gemm_8x8x8_acle<M_tile,N_tile, N_tile>((const float*)&A_buffer[jj][ii], &B_buffer[jj][kk] , &C_buffer[ii][kk]);
                            // gemm_8x8x8_x2_acle<M_tile,N_tile, N_tile>((const float*)&A_buffer[jj][ii], &B_buffer[jj][kk] , nullptr);
                            // gemm_8x8x8_x2_reorder_acle<M_tile,N_tile, N_tile>(((const float*)&A_buffer[0][0]) + jj/8*64 + ii * K_tile, &B_buffer[jj][kk] , nullptr);

                            gemm_8x8x8_x2_reorder_acle<M_tile,N_tile, N_tile, 0>(((const float*)&A_buffer[0][0]) + jj/8*64 + ii * K_tile, (&B_buffer[0][0])+jj/8*128 + kk*K_tile , nullptr);
                        }
                        #pragma GCC unroll 8
                        for (int iii = 0; iii < 8; ++iii)
                        {
                            // svst1_hor_za32(0, i, pg, (&C_buffer[ii][kk])+i*N_tile);
                            svst1_hor_za32(0, iii, pg, (c+(i+ii)*N+(k+kk))+iii*N);
                            svst1_hor_za32(1, iii, pg, (c+(i+ii)*N+(k+kk+8))+iii*N);

                            // svst1_hor_za32(2, iii, pg, (c+(i+ii)*N+(k+kk+16))+iii*N);
                            // svst1_hor_za32(3, iii, pg, (c+(i+ii)*N+(k+kk+24))+iii*N);
                        }
                    }
                }
                // for (int u = 0; u < M_tile; ++u)
                // {
                //     for (int v = 0; v < N_tile; ++v)
                //     {
                //          c[(i+u)*N+k+v] = C_buffer[u][v];
                //     }
                // }
            }
        }
    }
}
// template<int M, int N, int K>
// __arm_new("za")
// void gemm_nn_x2_reorder_change_za(const float* a, const float *b, float* c)  __arm_streaming
// {
//     asm(
//         "smstart\n"
//         "ptrue p0.s"
//     );
//     constexpr int M_tile = 256;
//     constexpr int K_tile = 128;
//     constexpr int N_tile = 64;

//     for (int i = 0; i < M; i += M_tile)
//     {
//         for (int j = 0; j < K; j +=K_tile)
//         {
//             float A_buffer[K_tile][M_tile];
//             // for (int u = 0; u < M_tile; ++u)
//             // {
//             //     for (int v = 0; v < K_tile; ++v)
//             //     {
//             //         A_buffer[v][u] = a[(i + u) * K + j + v];
//             //     }
//             // }
//             auto ptr = &A_buffer[0][0];

//             for (int u = 0; u < M_tile; u += 8)
//             {
//                 for (int v = 0; v < K_tile; v += 8)
//                 {
//                     for (int p = 0; p < 8; ++p)
//                     {
//                         for (int q = 0; q < 8; ++q)
//                         {
//                             *ptr = a[(i+u+q) * K + j + v + p];
//                             ptr++;
//                         }
//                     }
//                 }
//             }

//             for (int k = 0; k < N; k += N_tile)
//             {
//                 float B_buffer[K_tile][N_tile];
//                 // for (int u = 0; u < N_tile; u++)
//                 // {
//                 //     for (int v = 0; v < K_tile; v++)
//                 //     {
//                 //         B_buffer[v][u] = b[(j+v)*N + k+u];
//                 //     }
//                 // }
//                 svbool_t pg = svptrue_b32();
//                 float* B_ptr = &B_buffer[0][0];
//                 for (int u = 0; u < N_tile; u += 16)
//                 {
//                     for (int v = 0; v < K_tile; v += 8)
//                     {
//                         #pragma GCC unroll 8
//                         for (int p = 0; p < 8; ++p)
//                         {
//                             // for (int q = 0; q < 16; ++q)
//                             // {
//                             //     // B_buffer[v][u] = b[(j+v)*N + k+u];
//                             //     *B_ptr = b[(j+v+p)*N + k+u+q];
//                             //     B_ptr++;
//                             // }
//                             // memcpy(B_ptr, &b[(j+v+p)*N+k+u], sizeof(float)*16);
//                             svfloat32_t v1 = svld1(pg, &b[(j+v+p)*N+k+u]);
//                             svfloat32_t v2 = svld1(pg, &b[(j+v+p)*N+k+u+8]);
//                             svst1(pg, B_ptr, v1);
//                             svst1(pg, B_ptr+8, v2);
//                             B_ptr += 16;
//                         }
//                     }
//                 }
//                 // float C_buffer[M_tile][N_tile];
//                 // for (int u = 0; u < M_tile; ++u)
//                 // {
//                 //     for (int v = 0; v < N_tile; ++v)
//                 //     {
//                 //         C_buffer[u][v] = c[(i+u)*N+k+v];
//                 //     }
//                 // }
//                 for (int ii = 0; ii < M_tile; ii += 8)
//                 {
//                     // #pragma GCC unroll 2
//                     for (int kk = 0; kk < N_tile; kk += 16)
//                     {
//                         #pragma GCC unroll 8
//                         for (int iii = 0; iii < 8; ++iii)
//                         {
//                             // svld1_hor_za32(0, i, pg, (&C_buffer[ii][kk])+i*N_tile);
//                             if (1)
//                             {
//                                 svld1_hor_za32(0, iii, pg, (c+(i+ii)*N+(k+kk))+iii*N);
//                                 svld1_hor_za32(1, iii, pg, (c+(i+ii)*N+(k+kk+8))+iii*N);
//                             }
//                             else
//                             {
//                                 svld1_hor_za32(2, iii, pg, (c+(i+ii)*N+(k+kk))+iii*N);
//                                 svld1_hor_za32(3, iii, pg, (c+(i+ii)*N+(k+kk+8))+iii*N);
//                             }
//                             // svld1_hor_za32(2, iii, pg, (c+(i+ii)*N+(k+kk+16))+iii*N);
//                             // svld1_hor_za32(3, iii, pg, (c+(i+ii)*N+(k+kk+24))+iii*N);
//                         }
//                         #pragma GCC unroll 8
//                         for (int jj = 0; jj < K_tile; jj += 8)
//                         {
//                             // gemm_8x8x8_acle<K_tile,K_tile,N>((const float*)&A_buffer[ii][jj], b + (j+jj)  + (k+kk) * K , c + (i+ii) * N + (k +kk));
//                             // gemm_8x8x8_acle<M_tile,N_tile>((const float*)&A_buffer[jj][ii], &B_buffer[jj][kk] , c + (i+ii) * N + (k +kk), N);
//                             // gemm_8x8x8_acle<M_tile,N_tile, N_tile>((const float*)&A_buffer[jj][ii], &B_buffer[jj][kk] , &C_buffer[ii][kk]);
//                             // gemm_8x8x8_x2_acle<M_tile,N_tile, N_tile>((const float*)&A_buffer[jj][ii], &B_buffer[jj][kk] , nullptr);
//                             // gemm_8x8x8_x2_reorder_acle<M_tile,N_tile, N_tile>(((const float*)&A_buffer[0][0]) + jj/8*64 + ii * K_tile, &B_buffer[jj][kk] , nullptr);
//                             if (1)
//                             {

//                             gemm_8x8x8_x2_reorder_acle<M_tile,N_tile, N_tile, 0>(((const float*)&A_buffer[0][0]) + jj/8*64 + ii * K_tile, (&B_buffer[0][0])+jj/8*128 + kk*K_tile , nullptr);
//                             // gemm_8x8x8_asm(((const float*)&A_buffer[0][0]) + jj/8*64 + ii * K_tile, (&B_buffer[0][0])+jj/8*128 + kk*K_tile );
//                             }
//                             else
//                             {
//                             gemm_8x8x8_x2_reorder_acle<M_tile,N_tile, N_tile, 2>(((const float*)&A_buffer[0][0]) + jj/8*64 + ii * K_tile, (&B_buffer[0][0])+jj/8*128 + kk*K_tile , nullptr);

//                             }
//                         }
//                         #pragma GCC unroll 8
//                         for (int iii = 0; iii < 8; ++iii)
//                         {
//                             // svst1_hor_za32(0, i, pg, (&C_buffer[ii][kk])+i*N_tile);
//                             if (1)
//                             {
//                             svst1_hor_za32(0, iii, pg, (c+(i+ii)*N+(k+kk))+iii*N);
//                             svst1_hor_za32(1, iii, pg, (c+(i+ii)*N+(k+kk+8))+iii*N);

//                             }
//                             else
//                             {

//                             svst1_hor_za32(2, iii, pg, (c+(i+ii)*N+(k+kk))+iii*N);
//                             svst1_hor_za32(3, iii, pg, (c+(i+ii)*N+(k+kk+8))+iii*N);
//                             }
//                             // svst1_hor_za32(2, iii, pg, (c+(i+ii)*N+(k+kk+16))+iii*N);
//                             // svst1_hor_za32(3, iii, pg, (c+(i+ii)*N+(k+kk+24))+iii*N);
//                         }
//                     }
//                 }
//                 // for (int u = 0; u < M_tile; ++u)
//                 // {
//                 //     for (int v = 0; v < N_tile; ++v)
//                 //     {
//                 //          c[(i+u)*N+k+v] = C_buffer[u][v];
//                 //     }
//                 // }
//             }
//         }
//     }
// }


// void test_gemm()
// {
//     BlasSetNumThreads(1);
//     constexpr int M = 1024*2;
//     constexpr int N = 1024*2;
//     constexpr int K = 1024*4;
//     // constexpr int M = 128;
//     // constexpr int N = 256;
//     // constexpr int K = 512;
//     // constexpr int M = 8;
//     // constexpr int N = 8;
//     // constexpr int K = 8;
//     float* a = new float[M*K];
//     float* b = new float[K*N];
//     float* c1 = new float[M*N];
//     float* c2 = new float[M*N];
//     for (long i =0 ;i  < M*K; ++i)
//     {
//         a[i] = (double)i/1024;
//         // a[i] = 1;
//     }
//     for (long i =0 ;i  < N*K; ++i)
//     {
//         b[i] = i%1024;
//         // b[i] = 1;
//     }
//     memset(c1, 0, M*N*sizeof(float));
//     memset(c2, 0, M*N*sizeof(float));
//     // for (int i = 0; i < 5; ++i)
//     {
//         auto start = high_resolution_clock::now();
//     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, 
//                 a, K, 
//                 b, N,
//                 1, c2, N);
//         auto end = high_resolution_clock::now();
//         auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
//         std::cout <<"kblas: " << (double)elapsed_time / 1e9 << "s\n";
//     }
//     // for (int i = 0; i < 5; ++i)
//     {
//         auto start = high_resolution_clock::now();
//         gemm_nn_x2_reorder_change_za<M,N,K>(a, b, c1);
//         auto end = high_resolution_clock::now();
//         auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
//         std::cout <<"sme: " << (double)elapsed_time / 1e9 << "s\n";
//     }
    
    
//     for (int i = 0; i < M*N; ++i)
//     {
//         if (fabs(c1[i]-c2[i])/fabs(c1[i]+c2[i]) > 1e-3)
//         {
//             printf("error at %d: %f %f\n", i, c1[i], c2[i]);
//         }
//     }
// }



#define TIMES ((long)1e8)

// Function to read the ARM cycle counter (cntvct_el0)
inline uint64_t read_cycle_counter() {
    uint64_t cycle_count;
    asm volatile("mrs %0, cntvct_el0" : "=r" (cycle_count));
    return cycle_count;
}

// Function to read the timer frequency (cntfrq_el0)
inline uint64_t read_timer_frequency() {
    uint64_t timer_freq;
    asm volatile("mrs %0, cntfrq_el0" : "=r" (timer_freq));
    return timer_freq;
}

int init_perf();
void enable_perf(int fd);
void disable_perf(int fd, read_format* vals);

// int main() __arm_inout("za") __arm_streaming{
int main(){
    // Get the timer frequency (ticks per second)
    // uint64_t timer_freq = read_timer_frequency();
    // std::cout << "Timer frequency: " << timer_freq << " Hz" << std::endl;

    // test_gemm();
    // return 0;
 
  //Reset counters and start counting
    int fd = init_perf();
    read_format vals;

    // Prepare some sample data for the FMA
    float a[16], b[16], c[16], d[16];
    for (int i = 0; i < 16; ++i) {
        a[i] = 1.0f * i;
        b[i] = 2.0f * i;
        c[i] = 3.0f * i;
        d[i] = 0.0f;  // Result
    }

    // Load data into vector registers (before timing)
    asm volatile (
        "ptrue p0.b\n"                      // Predicate all elements active
        "ptrue p1.b\n"                      // Predicate all elements active
        "ptrue p2.b\n"                      // Predicate all elements active
        "ptrue p3.b\n"                      // Predicate all elements active
        "ptrue p4.b\n"                      // Predicate all elements active
        "ptrue p5.b\n"                      // Predicate all elements active
        "ptrue p6.b\n"                      // Predicate all elements active
        "ptrue p7.b\n"                      // Predicate all elements active
        "ld1w z0.s, p0/z, %[a]\n"           // Load a[] into z0 (vector register)
        "ld1w z1.s, p0/z, %[b]\n"           // Load b[] into z1
        "ld1w z2.s, p0/z, %[c]\n"           // Load c[] into z2
        "ld1w z3.s, p0/z, %[a]\n"           // Load more data into z3
        "ld1w z4.s, p0/z, %[b]\n"           // Load more data into z4
        "ld1w z5.s, p0/z, %[c]\n"           // Load more data into z5
        "ld1w z6.s, p0/z, %[a]\n"           // Load more data into z6
        "ld1w z7.s, p0/z, %[b]\n"           // Load more data into z7
        :
        : [a] "m" (a), [b] "m" (b), [c] "m" (c)
        : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "p0"
    );
    long svl = 0;
    asm volatile (
        "smstart\n"
        "rdsvl   %0, #1"
        : "=r"(svl)
    );
    printf("svl: %ld\n", svl);
    enable_perf(fd);
    // enable_sme();
    // Start timing
    auto start = high_resolution_clock::now();
    int v = 0;
    // Start measuring cycles for multiple FMA operations
    // uint64_t start_cycles = read_cycle_counter();
    asm volatile (
            "smstart\n"
    );
    asm volatile (
        "smstart\n"
        "ptrue p0.b\n"                      // Predicate all elements active
        "ptrue p1.b\n"                      // Predicate all elements active
        "ptrue p2.b\n"                      // Predicate all elements active
        "ptrue p3.b\n"                      // Predicate all elements active
        "ptrue p4.b\n"                      // Predicate all elements active
        "ptrue p5.b\n"                      // Predicate all elements active
        "ptrue p6.b\n"                      // Predicate all elements active
        "ptrue p7.b\n"                      // Predicate all elements active
        "mov    x12, 0\n"

    :
    :
    : "x12"
    );
    // svbool_t pg = svptrue_b32();

    // double buffer[1024];
    float* buffer = new float[1024];
    buffer = (float*)(((uint64_t)buffer & (~0xff))+0xff);
    const long inst_count = TIMES * 4 ;
    using FP=double;
    test((long)TIMES/100);
//     #pragma GCC unroll 4
//     for (unsigned long i = 0; i < TIMES; i++) {
//         // gemm_8x8x8_x2_reorder_acle<1,1, 1, 0>((float*)buffer, (float*)buffer+64, nullptr);
//         // gemm_8x8x8_asm((float*)buffer, (float*)buffer+64);
//         asm volatile(
// // "        mov     x2, %0\n"
// // "        mov     x3, %1\n"
// // "        ptrue   p0.b\n"
// "    \n"
// "        ld1w    {z0.s}, p0/z, [%0]\n"
// "        ld1w    {z1.s}, p0/z, [%1]\n"
// "        ld1w    {z2.s}, p0/z, [%2]\n"
// "        ld1w    {z3.s}, p0/z, [%3]\n"
// "        ld1w    {z4.s}, p0/z, [%4]\n"
// // "        ld1w    {z5.s}, p0/z, [%5]\n"
// // "        ld1w    {z6.s}, p0/z, [%1]\n"
// // "    \n"
// // "        ld1w    {z7.s}, p0/z, [%0]\n"
// // "        ld1w    {z4.s}, p0/z, [%1]\n"
// // "        ld1w    {z5.s}, p0/z, [%1]\n"
// // "    \n"
// // "        ld1w    {z6.s}, p0/z, [%0]\n"
// // "        ld1w    {z7.s}, p0/z, [%1]\n"
// // "        ld1w    {z8.s}, p0/z, [%1]\n"
// // "    \n"
// // "        ld1w    {z9.s}, p0/z, [%0]\n"
// // "        ld1w    {z10.s}, p0/z, [%1]\n"
// // "        ld1w    {z11.s}, p0/z, [%1]\n"
// // "    \n"
// // "        ld1w    {z12.s}, p0/z, [%0]\n"
// // "        ld1w    {z13.s}, p0/z, [%1]\n"
// // "        ld1w    {z14.s}, p0/z, [%1]\n"
// // "    \n"
// // "        ld1w    {z15.s}, p0/z, [%0]\n"
// // "        ld1w    {z16.s}, p0/z, [%1]\n"
// // "        ld1w    {z17.s}, p0/z, [%1]\n"
// // "    \n"
// // "        ld1w    {z18.s}, p0/z, [%0]\n"
// // "        ld1w    {z19.s}, p0/z, [%1]\n"
// // "        ld1w    {z20.s}, p0/z, [%1]\n"
// // "    \n"
// // "        ld1w    {z21.s}, p0/z, [%0]\n"
// // "        ld1w    {z22.s}, p0/z, [%1]\n"
// // "        ld1w    {z23.s}, p0/z, [%1]\n"
// "    \n"
//         //  "fmad z3.s, p0/m, z3.s, z3.s\n"           // z3 = z3 + (z3 * z2)
//         //  "fmla z4.s, p0/m, z4.s, z4.s\n"           // z4 = z4 + (z4 * z2)
//         //  "fmla z5.s, p0/m, z5.s, z5.s\n"           // z5 = z5 + (z5 * z2)
// "        fmopa   za0.s, p0/m, p0/m, z0.s, z1.s\n"
// "        fmopa   za1.s, p0/m, p0/m, z0.s, z2.s\n"
//          "fmad z5.s, p0/m, z0.s, z0.s\n"           // z0 = z0 + (z1 * z2)
//          "fmad z6.s, p0/m, z1.s, z1.s\n"           // z1 = z1 + (z1 * z2)
//          "fmad z7.s, p0/m, z2.s, z2.s\n"           // z2 = z2 + (z2 * z2)

//         "st1w {z5.s}, p0, [%6]\n"
//         "st1w {z6.s}, p0, [%7]\n"
//         // "st1w   {za0h.s[w12,3]}, p0, [%8]\n"
//         // "st1w {z7.s}, p0, [%8]\n"
// // "        \n"
// // "        fmopa   za2.s, p0/m, p0/m, z3.s, z4.s\n"
// // "        fmopa   za3.s, p0/m, p0/m, z3.s, z5.s\n"
// // // "        \n"
// // "        fmopa   za0.s, p0/m, p0/m, z6.s, z7.s\n"
// // "        fmopa   za1.s, p0/m, p0/m, z6.s, z8.s\n"
// // "        \n"
// // "        fmopa   za0.s, p0/m, p0/m, z9.s, z10.s\n"
// // "        fmopa   za1.s, p0/m, p0/m, z9.s, z11.s\n"
// // "        \n"
// // "        fmopa   za0.s, p0/m, p0/m, z12.s, z13.s\n"
// // "        fmopa   za1.s, p0/m, p0/m, z12.s, z14.s\n"
// // "        \n"
// // "        fmopa   za0.s, p0/m, p0/m, z15.s, z16.s\n"
// // "        fmopa   za1.s, p0/m, p0/m, z15.s, z17.s\n"
// // "        \n"
// // "        fmopa   za0.s, p0/m, p0/m, z18.s, z19.s\n"
// // "        fmopa   za1.s, p0/m, p0/m, z18.s, z20.s\n"
// // "        \n"
// // "        fmopa   za0.s, p0/m, p0/m, z21.s, z22.s\n"
// // "        fmopa   za1.s, p0/m, p0/m, z21.s, z23.s\n"
// :
// : "r"((float*)buffer), "r"((float*)buffer+8), "r"((float*)buffer+16), "r"((float*)(buffer+24)), "r"((float*)(buffer+32)), "r"((float*)(buffer+40)), "r"((float*)(buffer+48)), "r"(buffer+56), "r"(buffer+64)
// : "x12"
//         );
//         // Repeat the FMA instruction across multiple registers and iterations
//         // asm volatile (
//         //     "fmla z0.s, p0/m, z0.s, z0.s\n"           // z0 = z0 + (z1 * z2)
//         //     "fmla z1.s, p1/m, z1.s, z1.s\n"           // z1 = z1 + (z1 * z2)
//         //     "fmla z2.s, p2/m, z2.s, z2.s\n"           // z2 = z2 + (z2 * z2)
//         //     "fmla z3.s, p3/m, z3.s, z3.s\n"           // z3 = z3 + (z3 * z2)
//         //     "fmla z4.s, p4/m, z4.s, z4.s\n"           // z4 = z4 + (z4 * z2)
//         //     "fmla z5.s, p5/m, z5.s, z5.s\n"           // z5 = z5 + (z5 * z2)
//         //     "fmla z6.s, p6/m, z6.s, z6.s\n"           // z6 = z6 + (z6 * z2)
//         //     "fmla z7.s, p7/m, z7.s, z7.s\n"           // z7 = z7 + (z7 * z2)
//         //     // "fmla z2.s, p2/m, z2.s, z2.s\n"           // z2 = z2 + (z1 * z2)
//         //     // "fmla z3.s, p3/m, z3.s, z3.s\n"           // z3 = z3 + (z1 * z2)
//         //     // "fmla z3.s, p4/m, z4.s, z4.s\n"           // z4 = z4 + (z1 * z2)
//         //     // "fmla z3.s, p5/m, z5.s, z5.s\n"           // z5 = z5 + (z1 * z2)
//         //     // "fmla z3.s, p6/m, z6.s, z6.s\n"           // z6 = z6 + (z1 * z2)
//         //     // "fmla z3.s, p7/m, z7.s, z7.s\n"           // z7 = z7 + (z1 * z2)
//         // );

//         // asm (
//             // "ld1b   {z0.b}, p1/z, [%0]\n"
//             // "ld1b   {z1.b}, p1/z, [%1]\n"
//             // "ld1b   {z2.b}, p1/z, [%2]\n"
//             // "ld1b   {z3.b}, p1/z, [%3]\n"
//             // "fmops za0.s, p0/m, p1/m, z0.s, z0.s\n"
//             // "fmops za1.s, p0/m, p1/m, z1.s, z1.s\n"
// //             "fmops za2.s, p0/m, p1/m, z2.s, z2.s\n"
// //             "fmops za3.s, p0/m, p1/m, z3.s, z3.s\n"
// //             "st1b   {z0.b}, p1, [%0]\n"
// //             "st1b   {z1.b}, p1, [%1]\n"
// //             "st1b   {z2.b}, p1, [%2]\n"
// //             "st1b   {z3.b}, p1, [%3]\n"
// //             // "fmops za1.s, p0/m, p1/m, z0.s, z0.s\n"
// //             // "fmops za0.s, p0/m, p1/m, z0.s, z0.s\n"
// //             // "fmops za3.s, p0/m, p1/m, z0.s, z0.s\n"
// //             // "fmops za3.s, p0/m, p1/m, z0.s, z0.s\n"
// // // SUMOPA <ZAda>.D, <Pn>/M, <Pm>/M, <Zn>.H, <Zm>.H
// //             // "sumopa za0.s, p0/m, p1/m, z0.b, z0.b\n"
// //             // "sumopa za1.s, p0/m, p1/m, z0.b, z0.b\n"
// // // LD1B { ZA0<HV>.B[<Ws>, <offs>] }, <Pg>/Z, [<Xn|SP>{, <Xm>}]
// //             // "ld1b   za0h.b[w12,0], p7/z, [sp]\n"
// //             // "ld1b   za0h.b[w12,1], p7/z, [sp]\n"
// //             // "ld1b   za0h.b[w12,2], p7/z, [spgg]\n"
// //             // "ld1b   za0h.b[w12,3], p7/z, [sp]\n"
// //             // "ld1b    {z0.b}, p7/z, [%0]\n"
// //             // "ld1b    {z1.b}, p7/z, [%0]\n"
// //             // "ld1b    {z2.b}, p7/z, [%0]\n"
// //             // "ld1b    {z3.b}, p7/z, [%0]\n"
//             // : 
//             // : "r" (buffer), "r"(buffer+8), "r"(buffer+16), "r"(buffer+24)
//             // "msr     s0_3_c4_c7_3, xzr\n"
//             // "msr     s0_3_c4_c7_3, xzr\n"
//             // "msr     s0_3_c4_c6_3, xzr\n"
//             // "fmops za1.s, p1/m, p1/m, z1.h, z1.h\n"
//             // "fmops za2.s, p1/m, p1/m, z1.h, z1.h\n"
//             // "fmops za3.s, p1/m, p1/m, z1.h, z1.h\n"
//             // "fmops za1.h, p1/m, p1/m, z1.h, z1.h\n"
//             // "fmops za2.h, p2/m, p1/m, z2.h, z2.h\n"
//             // "fmops za3.h, p3/m, p1/m, z3.h, z3.h\n"
//             // "fmops za4.s, p4/m, p1/m, z4.s, z4.s\n"
//             // "fmops za5.s, p5/m, p1/m, z5.s, z5.s\n"
//             // "fmops za6.s, p6/m, p1/m, z6.s, z6.s\n"
//             // "fmops za7.s, p7/m, p1/m, z7.s, z7.s\n"
//             // "msr	s0_3_c4_c6_3, xzr\n"
//             // "mova  za0h.d[w12, 0], p7/m, z0.d\n"
//         // );
//     }
    disable_perf(fd, &vals);
    /*
            "fmla z0.s, p0/m, z0.s, z0.s\n"           // z0 = z0 + (z1 * z2)
            "fmla z1.s, p1/m, z1.s, z1.s\n"           // z1 = z1 + (z1 * z2)
            "fmla z2.s, p2/m, z2.s, z2.s\n"           // z2 = z2 + (z1 * z2)
            "fmla z3.s, p3/m, z3.s, z3.s\n"           // z3 = z3 + (z1 * z2)
            "fmla z4.s, p4/m, z4.s, z4.s\n"           // z4 = z4 + (z1 * z2)
            "fmla z5.s, p5/m, z5.s, z5.s\n"           // z5 = z5 + (z1 * z2)
            "fmla z6.s, p6/m, z6.s, z6.s\n"           // z6 = z6 + (z1 * z2)
            "fmla z7.s, p7/m, z7.s, z7.s\n"           // z7 = z7 + (z1 * z2)
						      //
            "fmla z0.s, p0/m, z1.s, z2.s\n"           // z0 = z0 + (z1 * z2)
            "fmla z3.s, p0/m, z4.s, z5.s\n"           // z3 = z3 + (z4 * z5)
            "fmla z6.s, p0/m, z7.s, z1.s\n"           // z6 = z6 + (z7 * z1)
            "fmla z0.s, p0/m, z1.s, z2.s\n"           // z0 = z0 + (z1 * z2)
            "fmla z3.s, p0/m, z4.s, z5.s\n"           // Repeat FMA
            "fmla z6.s, p0/m, z7.s, z1.s\n"           // Repeat FMA
            "fmla z0.s, p0/m, z1.s, z2.s\n"           // Repeat FMA
            "fmla z3.s, p0/m, z4.s, z5.s\n"           // Repeat FMA
            "fmla z6.s, p0/m, z7.s, z1.s\n"           // Repeat FMA
            "fmla z0.s, p0/m, z1.s, z2.s\n"           // Repeat FMA
						      */

    // Stop measuring cycles
    // uint64_t end_cycles = read_cycle_counter();
    // Stop timing
    auto end = high_resolution_clock::now();
 
  // Read and print result
    // Calculate and print elapsed time
    auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Convert elapsed time to seconds
    double elapsed_time_sec = elapsed_time / 1e9;

    // Store the result back to memory (after timing)
    asm volatile (
        "st1w z0.s, p0, %[d]\n"             // Store the result in d[]
        :
        : [d] "m" (d)
        : "z0", "p0"
    );

    // Calculate the elapsed cycles
    uint64_t elapsed_cycles = vals.values[0];

    // Calculate frequency (cycles per second)
    double frequency = vals.values[0] / elapsed_time_sec;

    // double ipc = (double)inst_count / elapsed_cycles;
    double ipc = (double)vals.values[1] / elapsed_cycles;
    double gflops = (double)inst_count / 1e9 * (512/sizeof(FP)/8) * (512/sizeof(FP)/8) * 2 / elapsed_time_sec ;
    double bandwidth = (double)inst_count / 1e9 * (512/sizeof(FP)/8) * sizeof(FP) * 2 / elapsed_time_sec ;
    

    // printf("cycles : %lld\n", val);
    printf( "%15s\t%10lf %s\n", "Cycles: " , (double)elapsed_cycles/1e9 , " x 1e9");
    printf( "%15s\t%10lf %s\n", "Valid Inst: " , (double)inst_count/1e9 , " x 1e9");
    printf( "%15s\t%10lf %s\n", "Total Inst: " , (double)vals.values[1]/1e9 , " x 1e9");
    printf( "%15s\t%10lf %s\n", "Time: " , elapsed_time_sec , " seconds" );
    printf( "%15s\t%10lf %s\n", "IPC: " ,  ipc, ""  );
    printf( "%15s\t%10lf %s\n", "Frequency: " , frequency/1e9 , " GHz" );
    printf( "%15s\t%10lf %s\n", "GFLPOS: " , gflops, " GFLOPS" );
    printf( "%15s\t%10lf %s\n", "Bandwidth: " , bandwidth, " GB/s" );

    // Output results for validation
    std::cout << "Result of FMA operation (first 4 elements): ";
    for (int i = 0; i < 4; ++i) {
        std::cout << d[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

int init_perf()
{
    struct perf_event_attr  pe;
    int fd;
	uint64_t val; 
  // Configure the event to count
    memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(struct perf_event_attr);
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.read_format = PERF_FORMAT_GROUP;
    pe.disabled = 1;
    pe.exclude_kernel = 1;   // Do not measure instructions executed in the kernel
    pe.exclude_hv = 1;  // Do not measure instructions executed in a hypervisor
 
  // Create the event
    fd = perf_event_open(&pe, 0, -1, -1, 0);
    if (fd < 0)
    {
        fprintf(stderr, "failed to open fd\n");
    }
    memset(&pe,0,sizeof(struct perf_event_attr));
    pe.size=sizeof(struct perf_event_attr);
    //监测
    pe.type=PERF_TYPE_HARDWARE;
    //监测时钟周期数
    pe.config=PERF_COUNT_HW_INSTRUCTIONS;
    //初始状态为禁用
    pe.disabled=1;
    pe.exclude_kernel = 1;   // Do not measure instructions executed in the kernel
    pe.exclude_hv = 1;  // Do not measure instructions executed in a hypervisor
    //创建perf文件描述符
    int fd2=perf_event_open(&pe,0,-1,fd,0);
    if (fd2 < 0)
    {
        fprintf(stderr, "failed to open fd2\n");
    }
    return fd;
    
}
void enable_perf(int fd)
{
    ioctl(fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
//   ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fd,PERF_EVENT_IOC_ENABLE,PERF_IOC_FLAG_GROUP);

}
void disable_perf(int fd, read_format* vals)
{

    ioctl(fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    // read(fd, &val, sizeof(val));
    read(fd, vals, sizeof(*vals));
}