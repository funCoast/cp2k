
find_package(argparse)
include(CheckCXXCompilerFlag)

add_executable(fmopa_gemm fmopa_gemm.cpp state.S)
target_compile_options(fmopa_gemm PRIVATE "-march=armv9-a+sme+sme2+sme-f64f64" "-w")
add_executable(IR2C_test IR2C_test.cpp)
target_link_libraries(IR2C_test PRIVATE IR)

add_executable(interface_test interface_test.cpp)
target_link_libraries(interface_test PRIVATE SMELT)

set(INTERFACE_TEST_BASE_MARCH "-march=armv9-a+sve2+sme+sme-f64f64")
set(INTERFACE_TEST_SME2_MARCH "-march=armv9-a+sve2+sme+sme2+sme-f64f64")
check_cxx_compiler_flag("${INTERFACE_TEST_SME2_MARCH}" INTERFACE_TEST_HAS_SME2_MARCH)

if(INTERFACE_TEST_HAS_SME2_MARCH)
    set(INTERFACE_TEST_MARCH "${INTERFACE_TEST_SME2_MARCH}")
else()
    set(INTERFACE_TEST_MARCH "${INTERFACE_TEST_BASE_MARCH}")
endif()

target_compile_options(interface_test PRIVATE "${INTERFACE_TEST_MARCH}")

# add_executable(IR2A_test IR2A_test.cpp)
# target_link_libraries(IR2A_test PRIVATE IR)

# add_executable(LLVMJITOverhead LLVMJITOverhead.cpp)
# target_link_libraries(LLVMJITOverhead PRIVATE
# LLVMCore
# LLVMSupport
# LLVMIRReader
# LLVMBitReader)
# target_include_directories(LLVMJITOverhead PRIVATE <llvm-include-dir>)
add_executable(frontend_test frontend_test.cpp)
target_link_libraries(frontend_test PRIVATE IR argparse::argparse)
