#!/bin/bash -e

# TODO: Review and if possible fix shellcheck errors.
# shellcheck disable=all

[ "${BASH_SOURCE[0]}" ] && SCRIPT_NAME="${BASH_SOURCE[0]}" || SCRIPT_NAME=$0
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_NAME}")/.." && pwd -P)"

source ${SCRIPT_DIR}/common_vars.sh
source ${SCRIPT_DIR}/tool_kit.sh
source ${SCRIPT_DIR}/signal_trap.sh
source ${INSTALLDIR}/toolchain.conf
source ${INSTALLDIR}/toolchain.env

for ii in $tool_list; do
  load "${BUILDDIR}/setup_${ii}"
done

# ------------------------------------------------------------------------
# Install or compile packages using newly installed tools
# ------------------------------------------------------------------------

# Setup compiler flags, leading to nice stack traces on crashes but still optimised
if [ "${with_intel}" != "__DONTUSE__" ]; then
  CFLAGS="-O2 -fPIC -fp-model=precise -funroll-loops -g -qopenmp -qopenmp-simd -traceback"
  if [ "${TARGET_CPU}" = "native" ]; then
    CFLAGS="${CFLAGS} -xHost"
  else
    CFLAGS="${CFLAGS} -mtune=${TARGET_CPU}"
  fi
  FFLAGS="${CFLAGS}"
elif [ "${with_amd}" != "__DONTUSE__" ]; then
  CFLAGS="-O2 -fPIC -fopenmp -g -mtune=${TARGET_CPU}"
  FFLAGS="${CFLAGS}"
else
  CFLAGS="-O2 -fPIC -fno-omit-frame-pointer -fopenmp -g -mtune=${TARGET_CPU} ${TSANFLAGS}"
  FFLAGS="${CFLAGS} -fbacktrace"
fi
CXXFLAGS="${CFLAGS}"
F77FLAGS="${FFLAGS}"
F90FLAGS="${FFLAGS}"
FCFLAGS="${FFLAGS}"

if [ "${with_intel}" == "__DONTUSE__" ] && [ "${with_amd}" == "__DONTUSE__" ]; then
  export CFLAGS="$(allowed_gcc_flags ${CFLAGS})"
  export FFLAGS="$(allowed_gfortran_flags ${FFLAGS})"
  export F77FLAGS="$(allowed_gfortran_flags ${F77FLAGS})"
  export F90FLAGS="$(allowed_gfortran_flags ${F90FLAGS})"
  export FCFLAGS="$(allowed_gfortran_flags ${FCFLAGS})"
  export CXXFLAGS="$(allowed_gxx_flags ${CXXFLAGS})"
else
  # TODO Check functions for allowed Intel or AMD compiler flags
  export CFLAGS
  export FFLAGS
  export F77FLAGS
  export F90FLAGS
  export FCFLAGS
  export CXXFLAGS
fi

# macOS toolchains often route MPI wrapper compilers through clang.
# Strip a few GNU-specific flags that are harmless for direct gcc builds
# but can make mpicc/mpic++ fail their configure-time compiler checks.
if [ "$(uname -s)" = "Darwin" ]; then
  CFLAGS="${CFLAGS//-Wa,-q/}"
  CXXFLAGS="${CXXFLAGS//-Wa,-q/}"
  FFLAGS="${FFLAGS//-Wa,-q/}"
  F77FLAGS="${F77FLAGS//-Wa,-q/}"
  F90FLAGS="${F90FLAGS//-Wa,-q/}"
  FCFLAGS="${FCFLAGS//-Wa,-q/}"
  CFLAGS="${CFLAGS//-Wl,-no_compact_unwind/}"
  CXXFLAGS="${CXXFLAGS//-Wl,-no_compact_unwind/}"
  FFLAGS="${FFLAGS//-Wl,-no_compact_unwind/}"
  F77FLAGS="${F77FLAGS//-Wl,-no_compact_unwind/}"
  F90FLAGS="${F90FLAGS//-Wl,-no_compact_unwind/}"
  FCFLAGS="${FCFLAGS//-Wl,-no_compact_unwind/}"
fi

export LDFLAGS="${TSANFLAGS}"

# get system arch information using OpenBLAS prebuild
${SCRIPTDIR}/get_openblas_arch.sh
load "${BUILDDIR}/openblas_arch"

write_toolchain_env "${INSTALLDIR}"

#EOF
