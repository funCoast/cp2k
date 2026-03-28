include(CheckCXXCompilerFlag)

add_library(IR STATIC
    IR.cpp
    IR2A.cpp
    IR2C.cpp

    # IR2LLVM.cpp
    primitive.cpp
    frontend.cpp
    block2Tile.cpp)

add_library(SMELT STATIC interface.cpp)
target_link_libraries(SMELT IR)

# SMELT is linked into the Python extension shared object, so both static
# archives must be position-independent to avoid relocation errors on AArch64.
set_target_properties(IR SMELT PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(SMELT_BASE_MARCH "-march=armv9-a+sve2+sme+sme-f64f64")
set(SMELT_SME2_MARCH "-march=armv9-a+sve2+sme+sme2+sme-f64f64")
check_cxx_compiler_flag("${SMELT_SME2_MARCH}" SMELT_HAS_SME2_MARCH)

if(SMELT_HAS_SME2_MARCH)
    set(SMELT_MARCH "${SMELT_SME2_MARCH}")
else()
    set(SMELT_MARCH "${SMELT_BASE_MARCH}")
endif()

target_compile_options(SMELT PRIVATE "${SMELT_MARCH}")