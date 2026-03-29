/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#ifndef SME_GEMM_ADAPTER_H
#define SME_GEMM_ADAPTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cp2k_smegemm_backend_ {
  CP2K_SMEGEMM_BACKEND_AUTO = 0,
  CP2K_SMEGEMM_BACKEND_SME = 1,
  CP2K_SMEGEMM_BACKEND_LIBXSMM = 2,
  CP2K_SMEGEMM_BACKEND_BLAS = 3
} cp2k_smegemm_backend;

cp2k_smegemm_backend cp2k_smegemm_backend_from_env(void);
const char *cp2k_smegemm_backend_name(cp2k_smegemm_backend backend);
int cp2k_smegemm_debug_enabled(void);

int cp2k_smegemm_dgemm_rowmajor(int transa, int transb, int m, int n, int k,
                                double alpha, const double *a, int lda,
                                const double *b, int ldb, double beta,
                                double *c, int ldc,
                                cp2k_smegemm_backend fallback_backend);

int cp2k_smegemm_sgemm_rowmajor(int transa, int transb, int m, int n, int k,
                                float alpha, const float *a, int lda,
                                const float *b, int ldb, float beta, float *c,
                                int ldc,
                                cp2k_smegemm_backend fallback_backend);

int cp2k_smegemm_dgemm_rowmajor_batch(int transa, int transb, int m, int n,
                                      int k, int64_t batch,
                                      const double *const *a_array,
                                      const double *const *b_array,
                                      double *const *c_array, double alpha,
                                      double beta,
                                      cp2k_smegemm_backend fallback_backend);

int cp2k_smegemm_sgemm_rowmajor_batch(int transa, int transb, int m, int n,
                                      int k, int64_t batch,
                                      const float *const *a_array,
                                      const float *const *b_array,
                                      float *const *c_array, float alpha,
                                      float beta,
                                      cp2k_smegemm_backend fallback_backend);

int cp2k_smegemm_dgemm_colmajor_batch(int transa, int transb, int m, int n,
                                      int k, int64_t batch,
                                      const double *const *a_array,
                                      const double *const *b_array,
                                      double *const *c_array, double alpha,
                                      double beta,
                                      cp2k_smegemm_backend fallback_backend);

int cp2k_smegemm_sgemm_colmajor_batch(int transa, int transb, int m, int n,
                                      int k, int64_t batch,
                                      const float *const *a_array,
                                      const float *const *b_array,
                                      float *const *c_array, float alpha,
                                      float beta,
                                      cp2k_smegemm_backend fallback_backend);

int cp2k_smegemm_dgemm_colmajor(int transa, int transb, int m, int n, int k,
                                double alpha, const double *a, int lda,
                                const double *b, int ldb, double beta,
                                double *c, int ldc,
                                cp2k_smegemm_backend fallback_backend);

int cp2k_smegemm_sgemm_colmajor(int transa, int transb, int m, int n, int k,
                                float alpha, const float *a, int lda,
                                const float *b, int ldb, float beta, float *c,
                                int ldc,
                                cp2k_smegemm_backend fallback_backend);

#ifdef __cplusplus
}
#endif

#endif
