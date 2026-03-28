#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRDPARTY_DIR="${ROOT_DIR}/thirdparty"
BUILD_ROOT="${THIRDPARTY_DIR}/_build"
INSTALL_ROOT="${THIRDPARTY_DIR}/_install"

LIBXSMM_SRC_DIR="${THIRDPARTY_DIR}/libxsmm"
LIBXSMM_BUILD_DIR="${BUILD_ROOT}/libxsmm"
LIBXSMM_CMAKE_BUILD_DIR="${LIBXSMM_BUILD_DIR}/cmake-build"

OPENBLAS_SRC_DIR="${THIRDPARTY_DIR}/OpenBLAS"
OPENBLAS_INSTALL_DIR="${INSTALL_ROOT}/openblas"

ARMPL_FETCH_SCRIPT="${ROOT_DIR}/scripts/fetch_armpl.sh"
ARMPL_STAGING_DIR="${THIRDPARTY_DIR}/_staging/armpl"
ARMPL_INSTALL_DIR="${INSTALL_ROOT}/armpl"

OS="$(uname -s)"
ARCH="$(uname -m)"

NPROC=1
if command -v nproc >/dev/null 2>&1; then
    NPROC="$(nproc)"
elif command -v sysctl >/dev/null 2>&1; then
    NPROC="$(sysctl -n hw.ncpu)"
fi

log() {
    echo "[build_thirdparty] $*"
}

die() {
    echo "[build_thirdparty][ERROR] $*" >&2
    exit 1
}

ensure_dir() {
    [[ -d "$1" ]] || die "Directory not found: $1"
}

detect_fortran_compiler() {
    if command -v gfortran >/dev/null 2>&1; then
        command -v gfortran
    elif command -v flang >/dev/null 2>&1; then
        command -v flang
    else
        echo ""
    fi
}

detect_gfortran_runtime_dir() {
    local fc="$1"
    [[ -n "${fc}" ]] || return 0

    local dylib_path=""
    dylib_path="$("${fc}" -print-file-name=libgfortran.dylib 2>/dev/null || true)"
    if [[ -n "${dylib_path}" && "${dylib_path}" != "libgfortran.dylib" && -f "${dylib_path}" ]]; then
        dirname "${dylib_path}"
        return 0
    fi

    local static_path=""
    static_path="$("${fc}" -print-file-name=libgfortran.a 2>/dev/null || true)"
    if [[ -n "${static_path}" && "${static_path}" != "libgfortran.a" && -f "${static_path}" ]]; then
        dirname "${static_path}"
        return 0
    fi

    echo ""
}

append_env_flag() {
    local var_name="$1"
    local flag="$2"
    local current="${!var_name:-}"
    if [[ " ${current} " == *" ${flag} "* ]]; then
        return 0
    fi
    export "${var_name}=${current:+${current} }${flag}"
}

append_flag_value() {
    local current="${1:-}"
    local flag="$2"
    if [[ " ${current} " == *" ${flag} "* ]]; then
        printf '%s' "${current}"
    else
        printf '%s' "${current:+${current} }${flag}"
    fi
}

cmake_bin() {
    if [[ -n "${CMAKE:-}" ]] && command -v "${CMAKE}" >/dev/null 2>&1; then
        command -v "${CMAKE}"
        return 0
    fi
    if command -v cmake >/dev/null 2>&1; then
        command -v cmake
    else
        die "cmake not found in PATH; configure it in CMakeUserPresets.json or export CMAKE"
    fi
}

setup_macos_sdk_env() {
    [[ "${OS}" == "Darwin" ]] || return 0
    if ! command -v xcrun >/dev/null 2>&1; then
        return 0
    fi

    local sdkroot="${SDKROOT:-$(xcrun --show-sdk-path 2>/dev/null || true)}"
    if [[ -z "${sdkroot}" || ! -d "${sdkroot}" ]]; then
        return 0
    fi

    export SDKROOT="${sdkroot}"
    append_env_flag CPPFLAGS "-isysroot ${SDKROOT}"
    append_env_flag CFLAGS "-isysroot ${SDKROOT}"
    append_env_flag CXXFLAGS "-isysroot ${SDKROOT}"
    append_env_flag LDFLAGS "-isysroot ${SDKROOT}"
}

detect_macos_tool() {
    local tool_name="$1"
    if [[ "${OS}" != "Darwin" ]]; then
        echo ""
        return 0
    fi

    if command -v xcrun >/dev/null 2>&1; then
        local xcrun_path=""
        xcrun_path="$(xcrun -f "${tool_name}" 2>/dev/null || true)"
        if [[ -n "${xcrun_path}" && -x "${xcrun_path}" ]]; then
            echo "${xcrun_path}"
            return 0
        fi
    fi

    if command -v "${tool_name}" >/dev/null 2>&1; then
        command -v "${tool_name}"
        return 0
    fi

    echo ""
}

# --------------------------------------------------
# libxsmm
# 改用 upstream 自带的 CMake 入口，统一输出到 thirdparty/_build/libxsmm/
# --------------------------------------------------
build_libxsmm() {
    log "========== Building libxsmm =========="
    ensure_dir "${LIBXSMM_SRC_DIR}"
    mkdir -p "${LIBXSMM_BUILD_DIR}" "${LIBXSMM_BUILD_DIR}/lib"
    setup_macos_sdk_env
    local cmake
    cmake="$(cmake_bin)"
    [[ -x "${cmake}" ]] || die "cmake not found: ${cmake}"

    local args=()
    args+=("-S" "${LIBXSMM_SRC_DIR}")
    args+=("-B" "${LIBXSMM_CMAKE_BUILD_DIR}")
    args+=("-DCMAKE_BUILD_TYPE=Release")
    if [[ "${LIBXSMM_STATIC:-1}" == "1" ]]; then
        args+=("-DXSMM_STATIC=ON")
    else
        args+=("-DXSMM_STATIC=OFF")
    fi
    args+=("-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=${LIBXSMM_BUILD_DIR}/lib")
    args+=("-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${LIBXSMM_BUILD_DIR}/lib")
    args+=("-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${LIBXSMM_BUILD_DIR}/bin")

    if [[ -n "${CC:-}" ]]; then
        args+=("-DCMAKE_C_COMPILER=${CC}")
    fi
    if [[ -n "${CXX:-}" ]]; then
        args+=("-DCMAKE_CXX_COMPILER=${CXX}")
    fi
    if [[ -n "${CFLAGS:-}" ]]; then
        args+=("-DCMAKE_C_FLAGS=${CFLAGS}")
    fi
    if [[ -n "${CXXFLAGS:-}" ]]; then
        args+=("-DCMAKE_CXX_FLAGS=${CXXFLAGS}")
    fi
    if [[ -n "${LDFLAGS:-}" ]]; then
        args+=("-DCMAKE_EXE_LINKER_FLAGS=${LDFLAGS}")
        args+=("-DCMAKE_SHARED_LINKER_FLAGS=${LDFLAGS}")
    fi

    if [[ "${LIBXSMM_DBG:-0}" != "0" || "${LIBXSMM_TRACE:-0}" != "0" || "${LIBXSMM_COMPATIBLE:-0}" != "0" ]]; then
        log "Note: LIBXSMM_DBG/LIBXSMM_TRACE/LIBXSMM_COMPATIBLE are make-only knobs and are ignored in CMake mode."
    fi

    log "OS=${OS}, ARCH=${ARCH}"
    log "Detected macOS SDK: ${SDKROOT:-<none>}"
    log "Detected C compiler: ${CC:-<cmake default>}"
    log "Detected CXX compiler: ${CXX:-<cmake default>}"
    log "libxsmm source dir : ${LIBXSMM_SRC_DIR}"
    log "libxsmm build dir  : ${LIBXSMM_BUILD_DIR}"
    log "libxsmm cmake dir  : ${LIBXSMM_CMAKE_BUILD_DIR}"
    log "libxsmm cmake args : ${args[*]}"

    (
        "${cmake}" "${args[@]}"
        "${cmake}" --build "${LIBXSMM_CMAKE_BUILD_DIR}" --target xsmm -j"${NPROC}"
    )

    rm -rf "${LIBXSMM_BUILD_DIR}/include"
    mkdir -p "${LIBXSMM_BUILD_DIR}/include"
    cp -R "${LIBXSMM_SRC_DIR}/include/." "${LIBXSMM_BUILD_DIR}/include/"

    if [[ "${LIBXSMM_STATIC:-1}" == "1" ]]; then
        [[ -f "${LIBXSMM_BUILD_DIR}/lib/libxsmm.a" ]] || die "Expected libxsmm static library not found under ${LIBXSMM_BUILD_DIR}/lib"
    fi

    log "libxsmm build done."
    log "Artifacts expected under: ${LIBXSMM_BUILD_DIR}"
}

clean_libxsmm() {
    log "========== Cleaning libxsmm =========="
    rm -rf "${LIBXSMM_BUILD_DIR}"
}

# --------------------------------------------------
# OpenBLAS
# 目标：先把库构建/安装出来，不强求 tests 全部通过
# --------------------------------------------------
build_openblas() {
    log "========== Building OpenBLAS =========="
    ensure_dir "${OPENBLAS_SRC_DIR}"
    cd "${OPENBLAS_SRC_DIR}"

    [[ -f Makefile || -f GNUmakefile ]] || die "No Makefile found in ${OPENBLAS_SRC_DIR}"

    local fc
    fc="$(detect_fortran_compiler)"

    local gfortran_runtime_dir=""
    if [[ -n "${fc}" ]]; then
        gfortran_runtime_dir="$(detect_gfortran_runtime_dir "${fc}")"
    fi

    local args=()
    args+=("-j${NPROC}")

    local ar_tool=""
    local nm_tool=""
    local ranlib_tool=""
    ar_tool="$(detect_macos_tool ar)"
    nm_tool="$(detect_macos_tool nm)"
    ranlib_tool="$(detect_macos_tool ranlib)"

    if [[ -n "${ar_tool}" ]]; then
        args+=("AR=${ar_tool}")
    fi
    if [[ -n "${nm_tool}" ]]; then
        args+=("NM=${nm_tool}")
    fi
    if [[ -n "${ranlib_tool}" ]]; then
        args+=("RANLIB=${ranlib_tool}")
    fi

    if [[ -n "${OPENBLAS_TARGET:-}" ]]; then
        args+=("TARGET=${OPENBLAS_TARGET}")
    fi

    if [[ "${OPENBLAS_DYNAMIC_ARCH:-0}" == "1" ]]; then
        args+=("DYNAMIC_ARCH=1")
    fi

    if [[ "${OPENBLAS_DEBUG:-0}" == "1" ]]; then
        args+=("DEBUG=1")
    fi

    # 若显式关闭 Fortran，则不使用 FC
    if [[ "${OPENBLAS_NOFORTRAN:-0}" == "1" ]]; then
        args+=("NOFORTRAN=1")
    else
        if [[ -n "${fc}" ]]; then
            args+=("FC=${fc}")
        else
            args+=("NOFORTRAN=1")
        fi
    fi

    # macOS 下为 gfortran runtime 补链接参数
    if [[ "${OPENBLAS_NOFORTRAN:-0}" != "1" && -n "${gfortran_runtime_dir}" ]]; then
        args+=("FEXTRALIB=-L${gfortran_runtime_dir} -lgfortran -lm")
    fi

    log "OS=${OS}, ARCH=${ARCH}"
    log "Detected Fortran compiler: ${fc:-<none>}"
    log "Detected libgfortran dir: ${gfortran_runtime_dir:-<none>}"
    log "Detected ar       : ${ar_tool:-<default>}"
    log "Detected nm       : ${nm_tool:-<default>}"
    log "Detected ranlib   : ${ranlib_tool:-<default>}"
    log "OpenBLAS source dir: ${OPENBLAS_SRC_DIR}"
    log "OpenBLAS make args : ${args[*]}"

    make shared "${args[@]}"

    log "OpenBLAS build done."
    log "Artifacts expected under source tree: ${OPENBLAS_SRC_DIR}"
}

install_openblas() {
    log "========== Installing OpenBLAS =========="
    ensure_dir "${OPENBLAS_SRC_DIR}"
    mkdir -p "${OPENBLAS_INSTALL_DIR}"
    cd "${OPENBLAS_SRC_DIR}"

    local fc
    fc="$(detect_fortran_compiler)"

    local gfortran_runtime_dir=""
    if [[ -n "${fc}" ]]; then
        gfortran_runtime_dir="$(detect_gfortran_runtime_dir "${fc}")"
    fi

    local args=()
    args+=("PREFIX=${OPENBLAS_INSTALL_DIR}")

    local ar_tool=""
    local nm_tool=""
    local ranlib_tool=""
    ar_tool="$(detect_macos_tool ar)"
    nm_tool="$(detect_macos_tool nm)"
    ranlib_tool="$(detect_macos_tool ranlib)"

    if [[ -n "${ar_tool}" ]]; then
        args+=("AR=${ar_tool}")
    fi
    if [[ -n "${nm_tool}" ]]; then
        args+=("NM=${nm_tool}")
    fi
    if [[ -n "${ranlib_tool}" ]]; then
        args+=("RANLIB=${ranlib_tool}")
    fi

    if [[ -n "${OPENBLAS_TARGET:-}" ]]; then
        args+=("TARGET=${OPENBLAS_TARGET}")
    fi

    if [[ "${OPENBLAS_DYNAMIC_ARCH:-0}" == "1" ]]; then
        args+=("DYNAMIC_ARCH=1")
    fi

    if [[ "${OPENBLAS_DEBUG:-0}" == "1" ]]; then
        args+=("DEBUG=1")
    fi

    if [[ "${OPENBLAS_NOFORTRAN:-0}" == "1" ]]; then
        args+=("NOFORTRAN=1")
    else
        if [[ -n "${fc}" ]]; then
            args+=("FC=${fc}")
        else
            args+=("NOFORTRAN=1")
        fi
    fi

    if [[ "${OPENBLAS_NOFORTRAN:-0}" != "1" && -n "${gfortran_runtime_dir}" ]]; then
        args+=("FEXTRALIB=-L${gfortran_runtime_dir} -lgfortran -lm")
    fi

    log "Detected Fortran compiler: ${fc:-<none>}"
    log "Detected libgfortran dir: ${gfortran_runtime_dir:-<none>}"
    log "Detected ar         : ${ar_tool:-<default>}"
    log "Detected nm         : ${nm_tool:-<default>}"
    log "Detected ranlib     : ${ranlib_tool:-<default>}"
    log "OpenBLAS install args: ${args[*]}"

    make install "${args[@]}"

    log "OpenBLAS installed to: ${OPENBLAS_INSTALL_DIR}"
}

clean_openblas() {
    log "========== Cleaning OpenBLAS =========="
    if [[ -d "${OPENBLAS_SRC_DIR}" ]]; then
        cd "${OPENBLAS_SRC_DIR}"
        make clean || true
    fi
    rm -rf "${OPENBLAS_INSTALL_DIR}"
}

# --------------------------------------------------
# ArmPL
# 不 build，不 make，只准备对应平台的软件包
# --------------------------------------------------
prepare_armpl() {
    log "========== Preparing ArmPL =========="
    [[ -x "${ARMPL_FETCH_SCRIPT}" ]] || die "fetch_armpl.sh not found or not executable: ${ARMPL_FETCH_SCRIPT}"

    log "OS=${OS}, ARCH=${ARCH}"
    bash "${ARMPL_FETCH_SCRIPT}"

    log "ArmPL staged under: ${ARMPL_STAGING_DIR}"
    log "Repo-local ArmPL install target: ${ARMPL_INSTALL_DIR}"
}

clean_armpl() {
    log "========== Cleaning ArmPL =========="
    rm -rf "${ARMPL_STAGING_DIR}" "${ARMPL_INSTALL_DIR}"
}

# --------------------------------------------------
# Combined
# --------------------------------------------------
build_all() {
    build_libxsmm
    build_openblas
    prepare_armpl
}

install_all() {
    install_openblas
}

clean_all() {
    clean_libxsmm
    clean_openblas
    clean_armpl
}

usage() {
    cat <<EOF
Usage:
  $0 build             Build libxsmm + OpenBLAS, and prepare ArmPL
  $0 install           Install OpenBLAS to thirdparty/_install/openblas
  $0 rebuild           Clean, then build all
  $0 clean             Clean all

  $0 libxsmm           Build only libxsmm
  $0 openblas          Build only OpenBLAS
  $0 armpl             Prepare only ArmPL

  $0 install-openblas  Install only OpenBLAS

Environment variables:
  LIBXSMM_STATIC=1         Build static libxsmm with CMake (default: 1)
  CC=...                   Optional C compiler override for libxsmm CMake configure
  CXX=...                  Optional C++ compiler override for libxsmm CMake configure

  OPENBLAS_TARGET=...      e.g. ARMV8 / NEHALEM
  OPENBLAS_DYNAMIC_ARCH=1  Enable runtime multi-target support
  OPENBLAS_DEBUG=1         Build debug OpenBLAS
  OPENBLAS_NOFORTRAN=1     Disable Fortran explicitly
EOF
}

cmd="${1:-build}"

case "${cmd}" in
    build)
        build_all
        ;;
    install)
        install_all
        ;;
    rebuild)
        clean_all
        build_all
        ;;
    clean)
        clean_all
        ;;
    libxsmm)
        build_libxsmm
        ;;
    openblas)
        build_openblas
        ;;
    armpl)
        prepare_armpl
        ;;
    install-openblas)
        install_openblas
        ;;
    *)
        usage
        exit 1
        ;;
esac

log "========== Done =========="
