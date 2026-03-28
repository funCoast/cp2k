include(CheckCXXCompilerFlag)

add_executable(m1_overlap_bench
    m1_overlap_bench.cpp
    ../test/state.S)

add_executable(m2_dgemm8x8_mopa_bench
    m2_dgemm8x8_mopa_bench.cpp
    ../test/state.S)

set(M1_OVERLAP_BASE_MARCH "-march=armv9-a+sve2+sme+sme-f64f64")
set(M1_OVERLAP_SME2_MARCH "-march=armv9-a+sve2+sme+sme2+sme-f64f64")
check_cxx_compiler_flag("${M1_OVERLAP_SME2_MARCH}" M1_OVERLAP_HAS_SME2_MARCH)

if(M1_OVERLAP_HAS_SME2_MARCH)
    set(M1_OVERLAP_MARCH "${M1_OVERLAP_SME2_MARCH}")
else()
    set(M1_OVERLAP_MARCH "${M1_OVERLAP_BASE_MARCH}")
endif()

target_compile_options(m1_overlap_bench PRIVATE
    "-O3"
    "-std=c++20"
    "${M1_OVERLAP_MARCH}")

target_compile_options(m2_dgemm8x8_mopa_bench PRIVATE
    "-O3"
    "-std=c++20"
    "-march=armv9-a+nosve+nosimd")
