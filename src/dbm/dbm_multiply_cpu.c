/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#include "dbm_multiply_cpu.h"
#include "dbm_hyperparams.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__SMEGEMM)
#include "../grid/dgemm/sme_gemm_adapter.h"
#endif

#if defined(__LIBXSMM)
#include <libxsmm.h>
#if !defined(DBM_LIBXSMM_PREFETCH)
// #define DBM_LIBXSMM_PREFETCH LIBXSMM_GEMM_PREFETCH_AL2_AHEAD
#define DBM_LIBXSMM_PREFETCH LIBXSMM_GEMM_PREFETCH_NONE
#endif
#if LIBXSMM_VERSION4(1, 17, 0, 3710) > LIBXSMM_VERSION_NUMBER
#define libxsmm_dispatch_gemm libxsmm_dispatch_gemm_v2
#endif
#endif

/*******************************************************************************
 * \brief Prototype for BLAS dgemm.
 * \author Ole Schuett
 ******************************************************************************/
void dgemm_(const char *transa, const char *transb, const int *m, const int *n,
            const int *k, const double *alpha, const double *a, const int *lda,
            const double *b, const int *ldb, const double *beta, double *c,
            const int *ldc);

/*******************************************************************************
 * \brief Private convenient wrapper to hide Fortran nature of dgemm_.
 * \author Ole Schuett
 ******************************************************************************/
static inline void dbm_dgemm(const char transa, const char transb, const int m,
                             const int n, const int k, const double alpha,
                             const double *a, const int lda, const double *b,
                             const int ldb, const double beta, double *c,
                             const int ldc) {

  dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c,
         &ldc);
}

/*******************************************************************************
 * \brief Private hash function based on Szudzik's elegant pairing.
 *        Using unsigned int to return a positive number even after overflow.
 *        https://en.wikipedia.org/wiki/Pairing_function#Other_pairing_functions
 *        https://stackoverflow.com/a/13871379
 *        http://szudzik.com/ElegantPairing.pdf
 * \author Ole Schuett
 ******************************************************************************/
static inline unsigned int hash(const dbm_task_t task) {
  const unsigned int m = task.m, n = task.n, k = task.k;
  const unsigned int mn = (m >= n) ? m * m + m + n : m + n * n;
  const unsigned int mnk = (mn >= k) ? mn * mn + mn + k : mn + k * k;
  return mnk;
}

#if defined(__SMEGEMM)
static inline int task_shape_compare(const dbm_task_t *lhs,
                                    const dbm_task_t *rhs) {
  if (lhs->m != rhs->m) {
    return lhs->m < rhs->m ? -1 : 1;
  }
  if (lhs->n != rhs->n) {
    return lhs->n < rhs->n ? -1 : 1;
  }
  if (lhs->k != rhs->k) {
    return lhs->k < rhs->k ? -1 : 1;
  }
  return 0;
}

static inline int dbm_sme_backend_enabled(void) {
  const cp2k_smegemm_backend backend = cp2k_smegemm_backend_from_env();
  return backend == CP2K_SMEGEMM_BACKEND_AUTO ||
         backend == CP2K_SMEGEMM_BACKEND_SME;
}

static inline int dbm_sme_debug_enabled(void) {
  const char *env = getenv("CP2K_GEMM_DEBUG");
  return env != NULL && env[0] != '\0' && strcmp(env, "0") != 0;
}

static void dbm_sme_debug(const char *message, const int ngroup,
                          const dbm_task_t *task) {
  if (!dbm_sme_debug_enabled()) {
    return;
  }
  if (task != NULL) {
    fprintf(stderr,
            "[DBM SME] %s: batch=%d shape=(m=%d n=%d k=%d) offsets=(%d,%d,%d)\n",
            message, ngroup, task->m, task->n, task->k, task->offset_a,
            task->offset_b, task->offset_c);
  } else {
    fprintf(stderr, "[DBM SME] %s: batch=%d\n", message, ngroup);
  }
  fflush(stderr);
}

static void dbm_add_scratch_to_c(const dbm_task_t *task,
                                 const double alpha, const double *scratch,
                                 dbm_shard_t *shard_c) {
  double *const data_c = shard_c->data + task->offset_c;
  const int elems = task->m * task->n;
  if (alpha == 1.0) {
    for (int i = 0; i < elems; ++i) {
      data_c[i] += scratch[i];
    }
  } else {
    for (int i = 0; i < elems; ++i) {
      data_c[i] += alpha * scratch[i];
    }
  }
}

static int dbm_multiply_cpu_process_sme_group(
    const int ngroup, const dbm_task_t group[ngroup], const double alpha,
    const dbm_pack_t *pack_a, const dbm_pack_t *pack_b, dbm_shard_t *shard_c,
    const int options) {
  if (ngroup <= 0) {
    return 1;
  }
  if (!dbm_sme_backend_enabled()) {
    return 0;
  }
  if (alpha == 0.0) {
    return 1;
  }
  const dbm_task_t task0 = group[0];
  for (int i = 1; i < ngroup; ++i) {
    if (group[i].m != task0.m || group[i].n != task0.n || group[i].k != task0.k) {
      return 0;
    }
  }
  // Extremely skinny micro-shapes are a poor fit for SME and, in practice on
  // the current macOS/M4 Pro runtime, they can fault before producing useful
  // throughput. Keep those on the fallback path and reserve SME for the
  // genuinely matrix-shaped batches we want to accelerate.
  if (task0.m < 4 || task0.n < 4 || task0.k < 4) {
    dbm_sme_debug("rejecting degenerate SME shape", ngroup, &task0);
    return 0;
  }

  // SME computes one compact output per task. We batch those outputs into a
  // temporary scratch buffer and then accumulate them back into the actual C
  // blocks so DBM keeps its original "beta-scaled C plus contributions"
  // semantics.
  const size_t scratch_elems =
      (size_t)ngroup * (size_t)task0.m * (size_t)task0.n;
  double *const scratch = malloc(scratch_elems * sizeof(double));
  const double **a_array = malloc((size_t)ngroup * sizeof(double *));
  const double **b_array = malloc((size_t)ngroup * sizeof(double *));
  double **c_array = malloc((size_t)ngroup * sizeof(double *));
  if (scratch == NULL || a_array == NULL || b_array == NULL || c_array == NULL) {
    free(scratch);
    free((void *)a_array);
    free((void *)b_array);
    free(c_array);
    return 0;
  }

  for (int i = 0; i < ngroup; ++i) {
    const dbm_task_t task = group[i];
    a_array[i] = pack_a->data + task.offset_a;
    b_array[i] = pack_b->data + task.offset_b;
    c_array[i] = scratch + ((size_t)i * (size_t)task0.m * (size_t)task0.n);
  }

  const cp2k_smegemm_backend fallback_backend =
#if defined(__LIBXSMM)
      CP2K_SMEGEMM_BACKEND_LIBXSMM;
#else
      CP2K_SMEGEMM_BACKEND_BLAS;
#endif

  dbm_sme_debug("calling cp2k_smegemm_dgemm_colmajor_batch", ngroup, &task0);
  const int ok = cp2k_smegemm_dgemm_colmajor_batch(
      'N', 'T', task0.m, task0.n, task0.k, ngroup, (const double *const *)a_array,
      (const double *const *)b_array, (double *const *)c_array, 1.0, 0.0,
      fallback_backend);
  if (!ok) {
    free(scratch);
    free((void *)a_array);
    free((void *)b_array);
    free(c_array);
    return 0;
  }
  dbm_sme_debug("cp2k_smegemm_dgemm_colmajor_batch returned", ngroup, &task0);

  for (int i = 0; i < ngroup; ++i) {
    dbm_add_scratch_to_c(&group[i], alpha,
                         scratch + ((size_t)i * (size_t)task0.m * (size_t)task0.n),
                         shard_c);
  }

  free(scratch);
  free((void *)a_array);
  free((void *)b_array);
  free(c_array);
  (void)options;
  return 1;
}
#endif

/*******************************************************************************
 * \brief Internal routine for executing the tasks in given batch on the CPU.
 * \author Ole Schuett
 ******************************************************************************/
static void dbm_multiply_cpu_process_batch_impl(
    int ntasks, const dbm_task_t batch[ntasks], double alpha,
    const dbm_pack_t *pack_a, const dbm_pack_t *pack_b, dbm_shard_t *shard_c,
    int options) {

  if (0 >= ntasks) { // nothing to do
    return;
  }
  dbm_shard_allocate_promised_blocks(shard_c);

  int batch_order[ntasks];
  if (DBM_MULTIPLY_TASK_REORDER & options) {
    // Sort tasks approximately by m,n,k via bucket sort.
    int buckets[DBM_BATCH_NUM_BUCKETS] = {0};
    for (int itask = 0; itask < ntasks; ++itask) {
      const int i = hash(batch[itask]) % DBM_BATCH_NUM_BUCKETS;
      ++buckets[i];
    }
    for (int i = 1; i < DBM_BATCH_NUM_BUCKETS; ++i) {
      buckets[i] += buckets[i - 1];
    }
    assert(buckets[DBM_BATCH_NUM_BUCKETS - 1] == ntasks);
    for (int itask = 0; itask < ntasks; ++itask) {
      const int i = hash(batch[itask]) % DBM_BATCH_NUM_BUCKETS;
      --buckets[i];
      batch_order[buckets[i]] = itask;
    }
  } else {
    for (int itask = 0; itask < ntasks; ++itask) {
      batch_order[itask] = itask;
    }
  }

#if defined(__LIBXSMM)
  // Prepare arguments for libxsmm's kernel-dispatch.
  const int flags = LIBXSMM_GEMM_FLAG_TRANS_B; // transa = "N", transb = "T"
  const int prefetch = DBM_LIBXSMM_PREFETCH;
  int kernel_m = 0, kernel_n = 0, kernel_k = 0;
#if (LIBXSMM_GEMM_PREFETCH_NONE != DBM_LIBXSMM_PREFETCH)
  double *data_a_next = NULL, *data_b_next = NULL, *data_c_next = NULL;
#endif
#if LIBXSMM_VERSION2(1, 17) < LIBXSMM_VERSION_NUMBER
  libxsmm_gemmfunction kernel_func = NULL;
#else
  libxsmm_dmmfunction kernel_func = NULL;
  const double beta = 1.0;
#endif
#endif

  // Loop over tasks.
  dbm_task_t task_next = batch[batch_order[0]];
  for (int itask = 0; itask < ntasks; ++itask) {
    const dbm_task_t task = task_next;
    task_next = batch[batch_order[(itask + 1) < ntasks ? (itask + 1) : itask]];

#if defined(__LIBXSMM)
    if (0 == (DBM_MULTIPLY_BLAS_LIBRARY & options) &&
        (task.m != kernel_m || task.n != kernel_n || task.k != kernel_k)) {
      if (LIBXSMM_SMM(task.m, task.n, task.m, 1 /*assume in-$, no RFO*/,
                      sizeof(double))) {
#if LIBXSMM_VERSION2(1, 17) < LIBXSMM_VERSION_NUMBER
        const libxsmm_gemm_shape shape = libxsmm_create_gemm_shape(
            task.m, task.n, task.k, task.m /*lda*/, task.n /*ldb*/,
            task.m /*ldc*/, LIBXSMM_DATATYPE_F64 /*aprec*/,
            LIBXSMM_DATATYPE_F64 /*bprec*/, LIBXSMM_DATATYPE_F64 /*cprec*/,
            LIBXSMM_DATATYPE_F64 /*calcp*/);
        kernel_func =
            (LIBXSMM_FEQ(1.0, alpha)
                 ? libxsmm_dispatch_gemm(shape, (libxsmm_bitfield)flags,
                                         (libxsmm_bitfield)prefetch)
                 : NULL);
#else
        kernel_func = libxsmm_dmmdispatch(task.m, task.n, task.k, NULL /*lda*/,
                                          NULL /*ldb*/, NULL /*ldc*/, &alpha,
                                          &beta, &flags, &prefetch);
#endif
      } else {
        kernel_func = NULL;
      }
      kernel_m = task.m;
      kernel_n = task.n;
      kernel_k = task.k;
    }
#endif
    // gemm_param wants non-const data even for A and B
    double *const data_a = pack_a->data + task.offset_a;
    double *const data_b = pack_b->data + task.offset_b;
    double *const data_c = shard_c->data + task.offset_c;

#if defined(__LIBXSMM)
    if (kernel_func != NULL) {
#if LIBXSMM_VERSION2(1, 17) < LIBXSMM_VERSION_NUMBER
      libxsmm_gemm_param gemm_param;
      gemm_param.a.primary = data_a;
      gemm_param.b.primary = data_b;
      gemm_param.c.primary = data_c;
#if (LIBXSMM_GEMM_PREFETCH_NONE != DBM_LIBXSMM_PREFETCH)
      gemm_param.a.quaternary = pack_a->data + task_next.offset_a;
      gemm_param.b.quaternary = pack_b->data + task_next.offset_b;
      gemm_param.c.quaternary = shard_c->data + task_next.offset_c;
#endif
      kernel_func(&gemm_param);
#elif (LIBXSMM_GEMM_PREFETCH_NONE != DBM_LIBXSMM_PREFETCH)
      kernel_func(data_a, data_b, data_c, pack_a->data + task_next.offset_a,
                  pack_b->data + task_next.offset_b,
                  shard_c->data + task_next.offset_c);
#else
      kernel_func(data_a, data_b, data_c);
#endif
    } else
#endif
    { // Fallback to BLAS when libxsmm is not available.
      dbm_dgemm('N', 'T', task.m, task.n, task.k, alpha, data_a, task.m, data_b,
                task.n, 1.0, data_c, task.m);
    }
  }
}

#if defined(__SMEGEMM)
/*******************************************************************************
 * \brief SME-first batch path for DBM tasks.
 * \author Ole Schuett
 ******************************************************************************/
static void dbm_multiply_cpu_process_batch_sme(
    int ntasks, const dbm_task_t batch[ntasks], double alpha,
    const dbm_pack_t *pack_a, const dbm_pack_t *pack_b, dbm_shard_t *shard_c,
    int options) {

  if (0 >= ntasks) {
    return;
  }
  dbm_shard_allocate_promised_blocks(shard_c);

  int batch_order[ntasks];
  if (DBM_MULTIPLY_TASK_REORDER & options) {
    int buckets[DBM_BATCH_NUM_BUCKETS] = {0};
    for (int itask = 0; itask < ntasks; ++itask) {
      const int i = hash(batch[itask]) % DBM_BATCH_NUM_BUCKETS;
      ++buckets[i];
    }
    for (int i = 1; i < DBM_BATCH_NUM_BUCKETS; ++i) {
      buckets[i] += buckets[i - 1];
    }
    assert(buckets[DBM_BATCH_NUM_BUCKETS - 1] == ntasks);
    for (int itask = 0; itask < ntasks; ++itask) {
      const int i = hash(batch[itask]) % DBM_BATCH_NUM_BUCKETS;
      --buckets[i];
      batch_order[buckets[i]] = itask;
    }
  } else {
    for (int itask = 0; itask < ntasks; ++itask) {
      batch_order[itask] = itask;
    }
  }

  // Stable exact-shape sort to maximize SME batch length.
  for (int i = 1; i < ntasks; ++i) {
    const int idx = batch_order[i];
    int j = i - 1;
    while (j >= 0 &&
           task_shape_compare(&batch[batch_order[j]], &batch[idx]) > 0) {
      batch_order[j + 1] = batch_order[j];
      --j;
    }
    batch_order[j + 1] = idx;
  }

  int group_start = 0;
  while (group_start < ntasks) {
    const dbm_task_t first = batch[batch_order[group_start]];
    int group_end = group_start + 1;
    while (group_end < ntasks) {
      const dbm_task_t next = batch[batch_order[group_end]];
      if (next.m != first.m || next.n != first.n || next.k != first.k) {
        break;
      }
      ++group_end;
    }

    const int ngroup = group_end - group_start;
    dbm_task_t *group = malloc((size_t)ngroup * sizeof(dbm_task_t));
    if (group == NULL) {
      for (int i = 0; i < ngroup; ++i) {
        const dbm_task_t single_task = batch[batch_order[group_start + i]];
        dbm_multiply_cpu_process_batch_impl(1, &single_task, alpha, pack_a,
                                            pack_b, shard_c, options);
      }
      group_start = group_end;
      continue;
    }
    for (int i = 0; i < ngroup; ++i) {
      group[i] = batch[batch_order[group_start + i]];
    }

    if (ngroup >= 4) {
      dbm_sme_debug("trying SME group", ngroup, &group[0]);
      if (dbm_multiply_cpu_process_sme_group(ngroup, group, alpha, pack_a,
                                             pack_b, shard_c, options)) {
        free(group);
        group_start = group_end;
        continue;
      }
      dbm_sme_debug("SME group fell back", ngroup, &group[0]);
    }

    // Fallback to the existing CPU path for this shape group.
    dbm_multiply_cpu_process_batch_impl(ngroup, group, alpha, pack_a, pack_b,
                                        shard_c, options);
    free(group);
    group_start = group_end;
  }
}
#endif

/*******************************************************************************
 * \brief Internal routine for executing the tasks in given batch on the CPU.
 * \author Ole Schuett
 ******************************************************************************/
void dbm_multiply_cpu_process_batch(int ntasks, const dbm_task_t batch[ntasks],
                                    double alpha, const dbm_pack_t *pack_a,
                                    const dbm_pack_t *pack_b,
                                    dbm_shard_t *shard_c, int options) {
#if defined(__SMEGEMM)
  if (dbm_sme_backend_enabled() &&
      0 == (options & DBM_MULTIPLY_BLAS_LIBRARY)) {
    dbm_multiply_cpu_process_batch_sme(ntasks, batch, alpha, pack_a, pack_b,
                                       shard_c, options);
    return;
  }
#endif
  dbm_multiply_cpu_process_batch_impl(ntasks, batch, alpha, pack_a, pack_b,
                                      shard_c, options);
}

// EOF
