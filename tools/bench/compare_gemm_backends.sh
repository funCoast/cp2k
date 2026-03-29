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
  local start_ts end_ts elapsed rc
  local status total_tests correct_tests failed_tests wrong_tests

  echo "== Running backend: ${backend} =="
  export CP2K_GEMM_BACKEND="$backend"

  start_ts="$(python3 - <<'PY'
import time
print(f"{time.perf_counter():.9f}")
PY
)"
  set +e
  python3 tests/do_regtest.py "$build_bin_dir" "$cp2k_version" | tee "$log_file"
  rc=${PIPESTATUS[0]}
  set -e
  end_ts="$(python3 - <<'PY'
import time
print(f"{time.perf_counter():.9f}")
PY
)"
  elapsed="$(python3 - <<PY
start = float(${start_ts})
end = float(${end_ts})
print(f"{end - start:.2f}")
PY
)"

  status="$(awk -F': ' '/^Status:/ {print $2; exit}' "$log_file")"
  total_tests="$(awk '/^Total number of[[:space:]]+tests/ {print $NF; exit}' "$log_file")"
  correct_tests="$(awk '/^Number of[[:space:]]+CORRECT[[:space:]]+tests/ {print $NF; exit}' "$log_file")"
  failed_tests="$(awk '/^Number of[[:space:]]+FAILED[[:space:]]+tests/ {print $NF; exit}' "$log_file")"
  wrong_tests="$(awk '/^Number of[[:space:]]+WRONG[[:space:]]+tests/ {print $NF; exit}' "$log_file")"

  printf '%s\n' \
    "Summary for ${backend}:" \
    "  status:        ${status:-unknown}" \
    "  total tests:   ${total_tests:-unknown}" \
    "  correct tests: ${correct_tests:-unknown}" \
    "  failed tests:  ${failed_tests:-unknown}" \
    "  wrong tests:   ${wrong_tests:-unknown}" \
    "  wall time:     ${elapsed}s"

  if [[ $rc -ne 0 ]]; then
    echo "  exit code:     $rc"
  fi

  export "RESULT_${backend}_SECONDS=$elapsed"
  export "RESULT_${backend}_STATUS=${status:-unknown}"
  export "RESULT_${backend}_TOTAL=${total_tests:-unknown}"
  export "RESULT_${backend}_CORRECT=${correct_tests:-unknown}"
  export "RESULT_${backend}_FAILED=${failed_tests:-unknown}"
  export "RESULT_${backend}_WRONG=${wrong_tests:-unknown}"
  return "$rc"
}

overall_rc=0
run_suite LIBXSMM || overall_rc=1
run_suite SME || overall_rc=1

if [[ -n "${RESULT_LIBXSMM_SECONDS:-}" && -n "${RESULT_SME_SECONDS:-}" ]]; then
  comparison="$(python3 - <<PY
libxsmm = float("${RESULT_LIBXSMM_SECONDS}")
sme = float("${RESULT_SME_SECONDS}")
delta = sme - libxsmm
pct = (delta / libxsmm * 100.0) if libxsmm else 0.0
if delta < 0:
    print(f"SME is faster by {-delta:.2f}s ({-pct:.1f}%)")
elif delta > 0:
    print(f"SME is slower by {delta:.2f}s ({pct:.1f}%)")
else:
    print("SME and LIBXSMM have identical wall time")
PY
)"
  echo
  echo "Comparison:"
  echo "  LIBXSMM: ${RESULT_LIBXSMM_SECONDS}s"
  echo "  SME:     ${RESULT_SME_SECONDS}s"
  echo "  -> ${comparison}"
fi

cat <<EOF

Done.
Logs:
  $log_root/LIBXSMM.log
  $log_root/SME.log
EOF

exit "$overall_rc"
