include(CheckCXXCompilerFlag)

add_executable(overlap_bench
    overlap_bench.cpp
    ../test/state.S)

set(OVERLAP_BENCH_BASE_MARCH "-march=armv9-a+sve2+sme+sme-f64f64")
set(OVERLAP_BENCH_SME2_MARCH "-march=armv9-a+sve2+sme+sme2+sme-f64f64")
check_cxx_compiler_flag("${OVERLAP_BENCH_SME2_MARCH}" OVERLAP_BENCH_HAS_SME2_MARCH)

if(OVERLAP_BENCH_HAS_SME2_MARCH)
    set(OVERLAP_BENCH_MARCH "${OVERLAP_BENCH_SME2_MARCH}")
else()
    set(OVERLAP_BENCH_MARCH "${OVERLAP_BENCH_BASE_MARCH}")
endif()

target_compile_options(overlap_bench PRIVATE
    "-O3"
    "-std=c++20"
    "${OVERLAP_BENCH_MARCH}")
