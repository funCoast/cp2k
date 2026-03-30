!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

MODULE sme_batch_bridge

   USE kinds, ONLY: dp
   USE, INTRINSIC :: ISO_C_BINDING, ONLY: c_double, &
                                            c_f_pointer, &
                                            c_int, &
                                            c_int64_t, &
                                            c_loc, &
                                            c_ptr

   IMPLICIT NONE

   PRIVATE

   INTEGER, PARAMETER, PUBLIC :: sme_min_batch = 4

   INTEGER(C_INT), PARAMETER, PRIVATE :: sme_backend_auto = 0_C_INT
   INTEGER(C_INT), PARAMETER, PRIVATE :: sme_backend_sme = 1_C_INT
   INTEGER(C_INT), PARAMETER, PRIVATE :: sme_backend_libxsmm = 2_C_INT
   INTEGER(C_INT), PARAMETER, PRIVATE :: sme_backend_blas = 3_C_INT

   PUBLIC :: sme_batch_try_dgemm_colmajor_strided
   PUBLIC :: sme_batch_wants_sme

   INTERFACE
      INTEGER(C_INT) FUNCTION cp2k_smegemm_backend_from_env() BIND(C, NAME="cp2k_smegemm_backend_from_env")
         IMPORT :: c_int
      END FUNCTION cp2k_smegemm_backend_from_env

      INTEGER(C_INT) FUNCTION cp2k_smegemm_dgemm_colmajor_batch( &
         transa, transb, m, n, k, batch, a_array, b_array, c_array, alpha, beta, fallback_backend) &
         BIND(C, NAME="cp2k_smegemm_dgemm_colmajor_batch")
         IMPORT :: c_double, c_int, c_int64_t, c_ptr
         INTEGER(C_INT), VALUE :: transa, transb, m, n, k, fallback_backend
         INTEGER(C_INT64_T), VALUE :: batch
         TYPE(C_PTR), INTENT(IN) :: a_array(*), b_array(*), c_array(*)
         REAL(C_DOUBLE), VALUE :: alpha, beta
      END FUNCTION cp2k_smegemm_dgemm_colmajor_batch
   END INTERFACE

CONTAINS

   LOGICAL FUNCTION sme_batch_wants_sme(batch)
      INTEGER, INTENT(IN) :: batch

      INTEGER(C_INT) :: backend

      sme_batch_wants_sme = .FALSE.
#if !defined(__SMEGEMM)
      IF (batch < 0) CONTINUE
#else
      IF (batch < sme_min_batch) RETURN
      backend = cp2k_smegemm_backend_from_env()
      sme_batch_wants_sme = (backend == sme_backend_auto .OR. backend == sme_backend_sme)
#endif
   END FUNCTION sme_batch_wants_sme

   SUBROUTINE sme_batch_try_dgemm_colmajor_strided(transa, transb, m, n, k, batch, &
                                                   a_ptrs, lda, b_ptrs, ldb, c_ptrs, ldc, alpha_scale, &
                                                   accumulate_output, success)
      CHARACTER(LEN=1), INTENT(IN)                      :: transa, transb
      INTEGER, INTENT(IN)                               :: m, n, k, batch, lda, ldb, ldc
      TYPE(C_PTR), DIMENSION(:), INTENT(IN)            :: a_ptrs, b_ptrs, c_ptrs
      REAL(dp), INTENT(IN)                              :: alpha_scale
      LOGICAL, INTENT(IN)                               :: accumulate_output
      LOGICAL, INTENT(OUT)                              :: success

      REAL(C_DOUBLE), ALLOCATABLE, TARGET               :: a_pack(:, :, :)
      REAL(C_DOUBLE), ALLOCATABLE, TARGET               :: b_pack(:, :, :)
      REAL(C_DOUBLE), ALLOCATABLE, TARGET               :: c_pack(:, :, :)
      REAL(C_DOUBLE), POINTER                           :: a_raw(:), b_raw(:), c_raw(:)
      TYPE(C_PTR), ALLOCATABLE                          :: a_compact(:), b_compact(:), c_compact(:)
      INTEGER(C_INT)                                    :: fallback_backend
      INTEGER                                           :: a_cols, a_rows, b_cols, b_rows, i, ia, ib, ic, j

      success = .FALSE.
#if !defined(__SMEGEMM)
      IF (batch < 0) CONTINUE
      RETURN
#else
      IF (.NOT. sme_batch_wants_sme(batch)) RETURN
      IF (SIZE(a_ptrs) < batch .OR. SIZE(b_ptrs) < batch .OR. SIZE(c_ptrs) < batch) RETURN

      a_rows = MERGE(m, k, transa == 'N' .OR. transa == 'n')
      a_cols = MERGE(k, m, transa == 'N' .OR. transa == 'n')
      b_rows = MERGE(k, n, transb == 'N' .OR. transb == 'n')
      b_cols = MERGE(n, k, transb == 'N' .OR. transb == 'n')

      ALLOCATE (a_pack(a_rows, a_cols, batch), b_pack(b_rows, b_cols, batch), c_pack(m, n, batch))
      ALLOCATE (a_compact(batch), b_compact(batch), c_compact(batch))

      DO i = 1, batch
         CALL c_f_pointer(a_ptrs(i), a_raw, [lda*a_cols])
         CALL c_f_pointer(b_ptrs(i), b_raw, [ldb*b_cols])
         DO j = 1, a_cols
            DO ia = 1, a_rows
               a_pack(ia, j, i) = a_raw((j - 1)*lda + ia)
            END DO
         END DO
         DO j = 1, b_cols
            DO ib = 1, b_rows
               b_pack(ib, j, i) = b_raw((j - 1)*ldb + ib)
            END DO
         END DO
         c_pack(:, :, i) = 0.0_dp
         a_compact(i) = c_loc(a_pack(1, 1, i))
         b_compact(i) = c_loc(b_pack(1, 1, i))
         c_compact(i) = c_loc(c_pack(1, 1, i))
      END DO

#if defined(__LIBXSMM)
      fallback_backend = sme_backend_libxsmm
#else
      fallback_backend = sme_backend_blas
#endif

      IF (cp2k_smegemm_dgemm_colmajor_batch(INT(IACHAR(transa), KIND=C_INT), INT(IACHAR(transb), KIND=C_INT), &
                                            m, n, k, INT(batch, KIND=C_INT64_T), a_compact, b_compact, c_compact, &
                                            1.0_C_DOUBLE, 0.0_C_DOUBLE, fallback_backend) /= 1_C_INT) THEN
         DEALLOCATE (a_pack, b_pack, c_pack, a_compact, b_compact, c_compact)
         RETURN
      END IF

      DO i = 1, batch
         CALL c_f_pointer(c_ptrs(i), c_raw, [ldc*n])
         DO j = 1, n
            DO ic = 1, m
               IF (accumulate_output) THEN
                  c_raw((j - 1)*ldc + ic) = c_raw((j - 1)*ldc + ic) + alpha_scale*c_pack(ic, j, i)
               ELSE
                  c_raw((j - 1)*ldc + ic) = alpha_scale*c_pack(ic, j, i)
               END IF
            END DO
         END DO
      END DO

      DEALLOCATE (a_pack, b_pack, c_pack, a_compact, b_compact, c_compact)
      success = .TRUE.
#endif
   END SUBROUTINE sme_batch_try_dgemm_colmajor_strided

END MODULE sme_batch_bridge
