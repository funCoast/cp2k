#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_bin_dir="${1:-$root_dir/build-sme/bin}"
cp2k_version="${2:-psmp}"
log_root="${3:-$root_dir/bench-logs/$(date +%Y%m%d-%H%M%S)}"

cd "$root_dir"

cp2k_exe="$build_bin_dir/cp2k.${cp2k_version}"
if [[ ! -x "$cp2k_exe" ]]; then
  echo "Missing executable: $cp2k_exe" >&2
  echo "Usage: $0 [build-bin-dir] [version] [log-dir]" >&2
  exit 1
fi

# Reuse the installed toolchain and keep the runtime environment stable.
. tools/toolchain/install/setup >/dev/null 2>&1
export PATH="/opt/homebrew/bin:$PATH"
export CP2K_DATA_DIR="$root_dir/data"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export CP2K_GEMM_DEBUG="${CP2K_GEMM_DEBUG:-0}"

mkdir -p "$log_root"

run_suite() {
  local backend="$1"
  local log_file="$log_root/${backend}.log"

  echo "== Running backend: ${backend} =="
  export CP2K_GEMM_BACKEND="$backend"

  # Skip unit tests here; the point of this script is application/regression coverage.
  python3 tests/do_regtest.py "$build_bin_dir" "$cp2k_version" --skip_unittests \
    | tee "$log_file"
}

run_suite LIBXSMM
run_suite SME

cat <<EOF

Done.
Logs:
  $log_root/LIBXSMM.log
  $log_root/SME.log
EOF
