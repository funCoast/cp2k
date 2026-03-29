cmake_minimum_required(VERSION 3.12)
project(SME_GEMM LANGUAGES C CXX ASM)
enable_language(ASM)

option(ENABLE_MY_ASSERT "Enable custom assertions" OFF)
add_compile_definitions($<$<BOOL:${ENABLE_MY_ASSERT}>:ENABLE_MY_ASSERT=1>)

option(COSTMODEL_AUTOGEN "Auto-generate cost_config.h" ON)
set(COSTMODEL_BENCH_CSV "" CACHE FILEPATH "Benchmark CSV from microbench/kernel_cost_bench.py")
set(COSTMODEL_OVERRIDE_JSON "" CACHE FILEPATH "JSON overrides for generated cost config")
set(COSTMODEL_CUSTOM_CONFIG "" CACHE FILEPATH "Path to custom cost_config.h; overrides auto generation")
set(COSTMODEL_BENCH_ITERS "300000" CACHE STRING "Iterations passed to overlap_bench")
set(COSTMODEL_BENCH_REPEATS "3" CACHE STRING "Repeat count for overlap_bench collection")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CXX_STANDARD 20)
set(SME_GEMM_ROOT "${CMAKE_CURRENT_LIST_DIR}")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

option(SME_GEMM_BUILD_THIRDPARTY "Automatically build third-party dependencies during builds" ON)
option(SME_GEMM_BUILD_TESTS "Build SME-GEMM-dev tests" ON)

set(COSTMODEL_GEN_INCLUDE_DIR "${CMAKE_BINARY_DIR}/generated/include")
set(COSTMODEL_GEN_HEADER "${COSTMODEL_GEN_INCLUDE_DIR}/cost_config.h")
set(COSTMODEL_AUTO_CSV "${CMAKE_BINARY_DIR}/generated/overlap_metrics.csv")
set(SME_GEMM_THIRDPARTY_SCRIPT "${SME_GEMM_ROOT}/scripts/build_thirdparty.sh")
set(SME_GEMM_THIRDPARTY_FETCH_SCRIPT "${SME_GEMM_ROOT}/scripts/fetch_armpl.sh")
set(SME_GEMM_THIRDPARTY_STAMP_DIR "${CMAKE_BINARY_DIR}/thirdparty")
set(SME_GEMM_THIRDPARTY_STAMP "${SME_GEMM_THIRDPARTY_STAMP_DIR}/build.stamp")
set(SME_GEMM_BUILD_ENV_PATH "$ENV{PATH}")
set(SME_GEMM_BUILD_ENV_HOME "$ENV{HOME}")
file(MAKE_DIRECTORY "${COSTMODEL_GEN_INCLUDE_DIR}")

set(COSTMODEL_GENERATOR "${SME_GEMM_ROOT}/microbench/generate_cost_config.py")
set(COSTMODEL_KERNEL_BENCH "${SME_GEMM_ROOT}/microbench/kernel_cost_bench.py")
set(COSTMODEL_OVERRIDE_ARGS)
if(COSTMODEL_OVERRIDE_JSON)
	list(APPEND COSTMODEL_OVERRIDE_ARGS --override-json "${COSTMODEL_OVERRIDE_JSON}")
endif()

if(SME_GEMM_BUILD_THIRDPARTY)
	add_custom_command(
		OUTPUT "${SME_GEMM_THIRDPARTY_STAMP}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${SME_GEMM_THIRDPARTY_STAMP_DIR}"
		COMMAND ${CMAKE_COMMAND} -E env
			"PATH=${SME_GEMM_BUILD_ENV_PATH}"
			"HOME=${SME_GEMM_BUILD_ENV_HOME}"
			/usr/bin/env bash "${SME_GEMM_THIRDPARTY_SCRIPT}" build
		COMMAND ${CMAKE_COMMAND} -E touch "${SME_GEMM_THIRDPARTY_STAMP}"
		DEPENDS "${SME_GEMM_THIRDPARTY_SCRIPT}" "${SME_GEMM_THIRDPARTY_FETCH_SCRIPT}"
		WORKING_DIRECTORY "${SME_GEMM_ROOT}"
		COMMENT "Building third-party dependencies"
		VERBATIM)

	add_custom_target(thirdparty_build DEPENDS "${SME_GEMM_THIRDPARTY_STAMP}")

	add_custom_target(rebuild_thirdparty
		COMMAND ${CMAKE_COMMAND} -E rm -f "${SME_GEMM_THIRDPARTY_STAMP}"
		COMMAND ${CMAKE_COMMAND} -E env
			"PATH=${SME_GEMM_BUILD_ENV_PATH}"
			"HOME=${SME_GEMM_BUILD_ENV_HOME}"
			/usr/bin/env bash "${SME_GEMM_THIRDPARTY_SCRIPT}" rebuild
		COMMAND ${CMAKE_COMMAND} -E touch "${SME_GEMM_THIRDPARTY_STAMP}"
		WORKING_DIRECTORY "${SME_GEMM_ROOT}"
		COMMENT "Rebuilding third-party dependencies"
		VERBATIM)
else()
	add_custom_target(thirdparty_build
		COMMAND ${CMAKE_COMMAND} -E echo "SME_GEMM_BUILD_THIRDPARTY=OFF, skipping third-party build."
		VERBATIM)

	add_custom_target(rebuild_thirdparty
		COMMAND ${CMAKE_COMMAND} -E echo "SME_GEMM_BUILD_THIRDPARTY=OFF, skipping third-party rebuild."
		VERBATIM)
endif()

add_compile_options("-O3")
add_compile_options("-g")
add_compile_options("-std=c++20")
include_directories(BEFORE "${COSTMODEL_GEN_INCLUDE_DIR}")
include_directories(include)
add_subdirectory(src)
if(SME_GEMM_BUILD_TESTS)
	add_subdirectory(test)
endif()
add_subdirectory(microbench)
add_subdirectory(motivation)

if(COSTMODEL_CUSTOM_CONFIG)
	message(STATUS "Using custom cost config: ${COSTMODEL_CUSTOM_CONFIG}")
	add_custom_command(
		OUTPUT "${COSTMODEL_GEN_HEADER}"
		COMMAND ${CMAKE_COMMAND} -E copy_if_different "${COSTMODEL_CUSTOM_CONFIG}" "${COSTMODEL_GEN_HEADER}"
		DEPENDS "${COSTMODEL_CUSTOM_CONFIG}"
		COMMENT "Copying custom cost_config.h"
		VERBATIM)
elseif(COSTMODEL_AUTOGEN)
	if(COSTMODEL_BENCH_CSV)
		set(COSTMODEL_EFFECTIVE_CSV "${COSTMODEL_BENCH_CSV}")
		add_custom_command(
			OUTPUT "${COSTMODEL_GEN_HEADER}"
			COMMAND ${Python3_EXECUTABLE} "${COSTMODEL_GENERATOR}"
				--output "${COSTMODEL_GEN_HEADER}"
				--bench-csv "${COSTMODEL_EFFECTIVE_CSV}"
				${COSTMODEL_OVERRIDE_ARGS}
			DEPENDS "${COSTMODEL_GENERATOR}" "${COSTMODEL_EFFECTIVE_CSV}"
			COMMENT "Generating cost_config.h from benchmark CSV"
			VERBATIM)
	else()
		set(COSTMODEL_EFFECTIVE_CSV "${COSTMODEL_AUTO_CSV}")
		add_custom_command(
			OUTPUT "${COSTMODEL_GEN_HEADER}"
			COMMAND ${Python3_EXECUTABLE} "${COSTMODEL_KERNEL_BENCH}"
				--build-dir "${CMAKE_BINARY_DIR}"
				--iters "${COSTMODEL_BENCH_ITERS}"
				--repeats "${COSTMODEL_BENCH_REPEATS}"
				--csv-output "${COSTMODEL_EFFECTIVE_CSV}"
			COMMAND ${Python3_EXECUTABLE} "${COSTMODEL_GENERATOR}"
				--output "${COSTMODEL_GEN_HEADER}"
				--bench-csv "${COSTMODEL_EFFECTIVE_CSV}"
				${COSTMODEL_OVERRIDE_ARGS}
			DEPENDS overlap_bench "${COSTMODEL_KERNEL_BENCH}" "${COSTMODEL_GENERATOR}"
			COMMENT "Running overlap_bench and generating cost_config.h"
			VERBATIM)
	endif()
else()
	message(STATUS "COSTMODEL_AUTOGEN=OFF and no custom config, using checked-in default include/cost_config.h")
	add_custom_command(
		OUTPUT "${COSTMODEL_GEN_HEADER}"
		COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SME_GEMM_ROOT}/include/cost_config.h" "${COSTMODEL_GEN_HEADER}"
		DEPENDS "${SME_GEMM_ROOT}/include/cost_config.h"
		COMMENT "Copying default cost_config.h"
		VERBATIM)
endif()

add_custom_target(cost_config_gen ALL DEPENDS "${COSTMODEL_GEN_HEADER}")
add_custom_target(regen_cost_config DEPENDS cost_config_gen)
add_dependencies(IR cost_config_gen)

foreach(sme_gemm_target IN ITEMS
	IR
	SMELT
	fmopa_gemm
	IR2C_test
	interface_test
	frontend_test
	overlap_bench)
	if(TARGET "${sme_gemm_target}")
		add_dependencies("${sme_gemm_target}" thirdparty_build)
	endif()
endforeach()
