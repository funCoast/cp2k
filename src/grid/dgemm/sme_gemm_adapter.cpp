/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "sme_gemm_adapter.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <string>
#include <type_traits>

#include "interface.h"

namespace {

constexpr std::int64_t kMinSmeBatch = 4;

std::mutex &log_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::once_flag &success_once_flag() {
  static std::once_flag flag;
  return flag;
}

bool debug_enabled() { return SMELT::is_debug_enabled(); }

bool backend_attempts_sme(const cp2k_smegemm_backend backend) {
  return backend == CP2K_SMEGEMM_BACKEND_AUTO ||
         backend == CP2K_SMEGEMM_BACKEND_SME;
}

bool exact_compact_rowmajor(const char transa, const char transb, const int m,
                            const int n, const int k, const int lda,
                            const int ldb, const int ldc) {
  const int expected_lda = (transa == 'N') ? k : m;
  const int expected_ldb = (transb == 'N') ? n : k;
  return lda == expected_lda && ldb == expected_ldb && ldc == n;
}

bool exact_compact_colmajor(const char transa, const char transb,
                            const int m, const int n, const int k,
                            const int lda, const int ldb, const int ldc) {
  const int expected_lda = (transa == 'N') ? m : k;
  const int expected_ldb = (transb == 'N') ? k : n;
  return lda == expected_lda && ldb == expected_ldb && ldc == m;
}

char normalize_trans(const int trans) {
  const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(trans)));
  return (c == 'N' || c == 'T') ? c : '\0';
}

std::string normalize_env_token(const char *value) {
  std::string token;
  if (value == nullptr) {
    return token;
  }
  for (const unsigned char ch : std::string(value)) {
    if (std::isspace(ch) || ch == '-' || ch == '_') {
      continue;
    }
    token.push_back(static_cast<char>(std::toupper(ch)));
  }
  return token;
}

void log_line(const std::string &line) {
  if (!debug_enabled()) {
    return;
  }
  std::lock_guard<std::mutex> lock(log_mutex());
  std::fprintf(stderr, "%s\n", line.c_str());
  std::fflush(stderr);
}

void log_success_once(const char *kind, const cp2k_smegemm_backend mode) {
  if (!debug_enabled()) {
    return;
  }
  std::call_once(success_once_flag(), [&]() {
    log_line(std::string("[SME-GEMM-dev]: 当前使用的后端是 SME-GEMM-dev, ") +
             kind + ", 成功 (env=" + cp2k_smegemm_backend_name(mode) + ")");
  });
}

void log_failure(const char *kind, const cp2k_smegemm_backend mode,
                 const char *reason,
                 const cp2k_smegemm_backend fallback_backend) {
  if (!debug_enabled()) {
    return;
  }
  log_line(std::string("[SME-GEMM-dev]: SME-GEMM-dev 失败 (") + kind +
           ", env=" + cp2k_smegemm_backend_name(mode) + "): " + reason +
           "，回退到 " + cp2k_smegemm_backend_name(fallback_backend));
}

class AutoContextSwitchGuard {
public:
  AutoContextSwitchGuard() : cfg_(SMELT::get_runtime_config()) {
    if (!cfg_.auto_context_switch) {
      cfg_.auto_context_switch = true;
      SMELT::set_runtime_config(cfg_);
      changed_ = true;
    }
  }

  ~AutoContextSwitchGuard() {
    if (changed_) {
      cfg_.auto_context_switch = false;
      SMELT::set_runtime_config(cfg_);
    }
  }

  AutoContextSwitchGuard(const AutoContextSwitchGuard &) = delete;
  AutoContextSwitchGuard &operator=(const AutoContextSwitchGuard &) = delete;

private:
  SMELT::RuntimeConfig cfg_;
  bool changed_ = false;
};

template <typename T>
bool check_common_constraints(const char *kind, const cp2k_smegemm_backend mode,
                              const int m, const int n, const int k,
                              const T alpha, const T beta, const T *a,
                              const T *b, T *c,
                              const cp2k_smegemm_backend fallback_backend) {
  if (!backend_attempts_sme(mode)) {
    return false;
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    log_failure(kind, mode, "m/n/k 必须为正", fallback_backend);
    return false;
  }
  if (a == nullptr || b == nullptr || c == nullptr) {
    log_failure(kind, mode, "矩阵指针不能为空", fallback_backend);
    return false;
  }
  if (alpha != static_cast<T>(1) || beta != static_cast<T>(0)) {
    log_failure(kind, mode, "仅支持 alpha=1 且 beta=0", fallback_backend);
    return false;
  }
  return true;
}

template <typename T>
int dgemm_colmajor_impl(int transa, int transb, int m, int n, int k,
                        T alpha, const T *a, int lda, const T *b, int ldb,
                        T beta, T *c, int ldc,
                        const cp2k_smegemm_backend fallback_backend,
                        const char *kind) {
  const auto mode = cp2k_smegemm_backend_from_env();
  if (backend_attempts_sme(mode)) {
    log_failure(kind, mode, "单次 GEMM 已禁用 SME；请使用 batch>=4 的批量路径",
                fallback_backend);
    return 0;
  }
  if (!check_common_constraints(kind, mode, m, n, k, alpha, beta, a, b, c,
                                fallback_backend)) {
    return 0;
  }

  const char ta = normalize_trans(transa);
  const char tb = normalize_trans(transb);
  if (ta == '\0' || tb == '\0') {
    log_failure(kind, mode, "仅支持 N/T 转置标志", fallback_backend);
    return 0;
  }

  if (!exact_compact_colmajor(ta, tb, m, n, k, lda, ldb, ldc)) {
    log_failure(kind, mode, "仅支持紧凑列主序布局", fallback_backend);
    return 0;
  }

  AutoContextSwitchGuard context_guard;
  try {
    if constexpr (std::is_same_v<T, double>) {
      SMELT::dgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
      SMELT::sgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    log_success_once(kind, mode);
    return 1;
  } catch (const std::exception &e) {
    log_failure(kind, mode, e.what(), fallback_backend);
    return 0;
  }
}

template <typename T>
int dgemm_rowmajor_impl(int transa, int transb, int m, int n, int k,
                        T alpha, const T *a, int lda, const T *b, int ldb,
                        T beta, T *c, int ldc,
                        const cp2k_smegemm_backend fallback_backend,
                        const char *kind) {
  const auto mode = cp2k_smegemm_backend_from_env();
  if (backend_attempts_sme(mode)) {
    log_failure(kind, mode, "单次 GEMM 已禁用 SME；请使用 batch>=4 的批量路径",
                fallback_backend);
    return 0;
  }
  if (!check_common_constraints(kind, mode, m, n, k, alpha, beta, a, b, c,
                                fallback_backend)) {
    return 0;
  }

  const char row_ta = normalize_trans(transa);
  const char row_tb = normalize_trans(transb);
  if (row_ta == '\0' || row_tb == '\0') {
    log_failure(kind, mode, "仅支持 N/T 转置标志", fallback_backend);
    return 0;
  }
  if (!exact_compact_rowmajor(row_ta, row_tb, m, n, k, lda, ldb, ldc)) {
    log_failure(kind, mode, "仅支持紧凑行主序布局", fallback_backend);
    return 0;
  }

  const char ta = normalize_trans(transa);
  const char tb = normalize_trans(transb);
  if (ta == '\0' || tb == '\0') {
    log_failure(kind, mode, "仅支持 N/T 转置标志", fallback_backend);
    return 0;
  }

  AutoContextSwitchGuard context_guard;
  try {
    if constexpr (std::is_same_v<T, double>) {
      SMELT::dgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
      SMELT::sgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    log_success_once(kind, mode);
    return 1;
  } catch (const std::exception &e) {
    log_failure(kind, mode, e.what(), fallback_backend);
    return 0;
  }
}

template <typename T>
int dgemm_rowmajor_batch_impl(int transa, int transb, int m, int n, int k,
                              const std::int64_t batch, const T *const *a_array,
                              const T *const *b_array, T *const *c_array,
                              T alpha, T beta,
                              const cp2k_smegemm_backend fallback_backend,
                              const char *kind) {
  const auto mode = cp2k_smegemm_backend_from_env();
  if (!backend_attempts_sme(mode)) {
    return 0;
  }
  if (batch <= 0) {
    log_failure(kind, mode, "batch 必须为正", fallback_backend);
    return 0;
  }
  if (batch < kMinSmeBatch) {
    log_failure(kind, mode, "batch 太小；SME 仅用于 batch>=4 的批量路径",
                fallback_backend);
    return 0;
  }
  if (a_array == nullptr || b_array == nullptr || c_array == nullptr) {
    log_failure(kind, mode, "batch 指针数组不能为空", fallback_backend);
    return 0;
  }
  for (std::int64_t i = 0; i < batch; ++i) {
    if (a_array[i] == nullptr || b_array[i] == nullptr || c_array[i] == nullptr) {
      log_failure(kind, mode, "batch 元素指针不能为空", fallback_backend);
      return 0;
    }
  }
  if (alpha != static_cast<T>(1) || beta != static_cast<T>(0)) {
    log_failure(kind, mode, "仅支持 alpha=1 且 beta=0", fallback_backend);
    return 0;
  }

  const char ta = normalize_trans(transa);
  const char tb = normalize_trans(transb);
  if (ta == '\0' || tb == '\0') {
    log_failure(kind, mode, "仅支持 N/T 转置标志", fallback_backend);
    return 0;
  }

  AutoContextSwitchGuard context_guard;
  try {
    if constexpr (std::is_same_v<T, double>) {
      SMELT::dgemm_batch(ta, tb, m, n, k, batch, a_array, b_array, c_array);
    } else {
      SMELT::sgemm_batch(ta, tb, m, n, k, batch, a_array, b_array, c_array);
    }
    log_success_once(kind, mode);
    return 1;
  } catch (const std::exception &e) {
    log_failure(kind, mode, e.what(), fallback_backend);
    return 0;
  }
}

template <typename T>
int dgemm_colmajor_batch_impl(int transa, int transb, int m, int n, int k,
                              const std::int64_t batch, const T *const *a_array,
                              const T *const *b_array, T *const *c_array,
                              T alpha, T beta,
                              const cp2k_smegemm_backend fallback_backend,
                              const char *kind) {
  const auto mode = cp2k_smegemm_backend_from_env();
  if (!backend_attempts_sme(mode)) {
    return 0;
  }
  if (batch <= 0) {
    log_failure(kind, mode, "batch 必须为正", fallback_backend);
    return 0;
  }
  if (batch < kMinSmeBatch) {
    log_failure(kind, mode, "batch 太小；SME 仅用于 batch>=4 的批量路径",
                fallback_backend);
    return 0;
  }
  if (a_array == nullptr || b_array == nullptr || c_array == nullptr) {
    log_failure(kind, mode, "batch 指针数组不能为空", fallback_backend);
    return 0;
  }
  for (std::int64_t i = 0; i < batch; ++i) {
    if (a_array[i] == nullptr || b_array[i] == nullptr || c_array[i] == nullptr) {
      log_failure(kind, mode, "batch 元素指针不能为空", fallback_backend);
      return 0;
    }
  }
  if (alpha != static_cast<T>(1) || beta != static_cast<T>(0)) {
    log_failure(kind, mode, "仅支持 alpha=1 且 beta=0", fallback_backend);
    return 0;
  }

  const char ta = normalize_trans(transa);
  const char tb = normalize_trans(transb);
  if (ta == '\0' || tb == '\0') {
    log_failure(kind, mode, "仅支持 N/T 转置标志", fallback_backend);
    return 0;
  }

  if (!exact_compact_colmajor(ta, tb, m, n, k, m, n, m)) {
    log_failure(kind, mode, "仅支持紧凑列主序布局", fallback_backend);
    return 0;
  }

  AutoContextSwitchGuard context_guard;
  try {
    if constexpr (std::is_same_v<T, double>) {
      SMELT::dgemm_batch_colmajor(ta, tb, m, n, k, batch, a_array, b_array,
                                  c_array);
    } else {
      SMELT::sgemm_batch_colmajor(ta, tb, m, n, k, batch, a_array, b_array,
                                  c_array);
    }
    log_success_once(kind, mode);
    return 1;
  } catch (const std::exception &e) {
    log_failure(kind, mode, e.what(), fallback_backend);
    return 0;
  }
}

cp2k_smegemm_backend parse_backend(const char *value) {
  const std::string token = normalize_env_token(value);
  if (token.empty() || token == "AUTO") {
    return CP2K_SMEGEMM_BACKEND_AUTO;
  }
  if (token == "SME" || token == "SMEGEMM" || token == "SMEGEMMDEV") {
    return CP2K_SMEGEMM_BACKEND_SME;
  }
  if (token == "LIBXSMM" || token == "XSMM") {
    return CP2K_SMEGEMM_BACKEND_LIBXSMM;
  }
  if (token == "BLAS") {
    return CP2K_SMEGEMM_BACKEND_BLAS;
  }
  return CP2K_SMEGEMM_BACKEND_AUTO;
}

} // namespace

extern "C" {

cp2k_smegemm_backend cp2k_smegemm_backend_from_env(void) {
  static std::once_flag once;
  static cp2k_smegemm_backend backend = CP2K_SMEGEMM_BACKEND_AUTO;
  std::call_once(once, []() {
    backend = parse_backend(std::getenv("CP2K_GEMM_BACKEND"));
  });
  return backend;
}

int cp2k_smegemm_debug_enabled(void) {
  return SMELT::is_debug_enabled() ? 1 : 0;
}

const char *cp2k_smegemm_backend_name(const cp2k_smegemm_backend backend) {
  switch (backend) {
  case CP2K_SMEGEMM_BACKEND_AUTO:
    return "AUTO";
  case CP2K_SMEGEMM_BACKEND_SME:
    return "SME-GEMM-dev";
  case CP2K_SMEGEMM_BACKEND_LIBXSMM:
    return "LIBXSMM";
  case CP2K_SMEGEMM_BACKEND_BLAS:
    return "BLAS";
  }
  return "AUTO";
}

int cp2k_smegemm_dgemm_rowmajor(int transa, int transb, int m, int n, int k,
                                double alpha, const double *a, int lda,
                                const double *b, int ldb, double beta,
                                double *c, int ldc,
                                cp2k_smegemm_backend fallback_backend) {
  return dgemm_rowmajor_impl(transa, transb, m, n, k, alpha, a, lda, b, ldb,
                             beta, c, ldc, fallback_backend,
                             "dgemm(row-major)");
}

int cp2k_smegemm_sgemm_rowmajor(int transa, int transb, int m, int n, int k,
                                float alpha, const float *a, int lda,
                                const float *b, int ldb, float beta, float *c,
                                int ldc,
                                cp2k_smegemm_backend fallback_backend) {
  return dgemm_rowmajor_impl(transa, transb, m, n, k, alpha, a, lda, b, ldb,
                             beta, c, ldc, fallback_backend,
                             "sgemm(row-major)");
}

int cp2k_smegemm_dgemm_rowmajor_batch(int transa, int transb, int m, int n,
                                      int k, std::int64_t batch,
                                      const double *const *a_array,
                                      const double *const *b_array,
                                      double *const *c_array, double alpha,
                                      double beta,
                                      cp2k_smegemm_backend fallback_backend) {
  return dgemm_rowmajor_batch_impl(transa, transb, m, n, k, batch, a_array,
                                   b_array, c_array, alpha, beta,
                                   fallback_backend, "dgemm_batch(row-major)");
}

int cp2k_smegemm_sgemm_rowmajor_batch(int transa, int transb, int m, int n,
                                      int k, std::int64_t batch,
                                      const float *const *a_array,
                                      const float *const *b_array,
                                      float *const *c_array, float alpha,
                                      float beta,
                                      cp2k_smegemm_backend fallback_backend) {
  return dgemm_rowmajor_batch_impl(transa, transb, m, n, k, batch, a_array,
                                   b_array, c_array, alpha, beta,
                                   fallback_backend, "sgemm_batch(row-major)");
}

int cp2k_smegemm_dgemm_colmajor_batch(int transa, int transb, int m, int n,
                                      int k, std::int64_t batch,
                                      const double *const *a_array,
                                      const double *const *b_array,
                                      double *const *c_array, double alpha,
                                      double beta,
                                      cp2k_smegemm_backend fallback_backend) {
  return dgemm_colmajor_batch_impl(transa, transb, m, n, k, batch, a_array,
                                   b_array, c_array, alpha, beta,
                                   fallback_backend, "dgemm_batch(col-major)");
}

int cp2k_smegemm_sgemm_colmajor_batch(int transa, int transb, int m, int n,
                                      int k, std::int64_t batch,
                                      const float *const *a_array,
                                      const float *const *b_array,
                                      float *const *c_array, float alpha,
                                      float beta,
                                      cp2k_smegemm_backend fallback_backend) {
  return dgemm_colmajor_batch_impl(transa, transb, m, n, k, batch, a_array,
                                   b_array, c_array, alpha, beta,
                                   fallback_backend, "sgemm_batch(col-major)");
}

int cp2k_smegemm_dgemm_colmajor(int transa, int transb, int m, int n, int k,
                                double alpha, const double *a, int lda,
                                const double *b, int ldb, double beta,
                                double *c, int ldc,
                                cp2k_smegemm_backend fallback_backend) {
  return dgemm_colmajor_impl(transa, transb, m, n, k, alpha, a, lda, b, ldb,
                             beta, c, ldc, fallback_backend,
                             "dgemm(col-major)");
}

int cp2k_smegemm_sgemm_colmajor(int transa, int transb, int m, int n, int k,
                                float alpha, const float *a, int lda,
                                const float *b, int ldb, float beta, float *c,
                                int ldc,
                                cp2k_smegemm_backend fallback_backend) {
  return dgemm_colmajor_impl(transa, transb, m, n, k, alpha, a, lda, b, ldb,
                             beta, c, ldc, fallback_backend,
                             "sgemm(col-major)");
}

} // extern "C"
