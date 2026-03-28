#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from gemm_bench_common import (
    BENCH_OUT_DIR,
    BUILD_DIR,
    DEFAULT_BATCH,
    DEFAULT_ITERS,
    DEFAULT_SIZES,
    DEFAULT_WARMUP,
    VERIFY_ARTIFACT_DIR,
    cmake_env,
    ensure_cmake_target,
    print_summary,
    write_rows,
)
LOG_PATTERN = re.compile(
    r"gemm_kernel\s+Time\s*=\s*([0-9.eE+-]+) ns\s+GFLOPS\s*=\s*([0-9.eE+-]+)\s+Kernel Time\s*=\s*([0-9.eE+-]+) ns",
    re.MULTILINE,
)


def parse_sizes(text: str) -> list[int]:
    values = [int(part) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("sizes must not be empty")
    return values


def sanitize_artifact_component(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def layout_note_name(layout: str) -> str:
    return layout.upper()


def artifact_stem(strategy: str, layout: str, dtype: str, m: int, n: int, k: int, batch: int) -> str:
    return (
        f"{sanitize_artifact_component(strategy)}__ir2c__{sanitize_artifact_component(layout)}__"
        f"{sanitize_artifact_component(dtype)}__M{m}_N{n}_K{k}_B{batch}"
    )


def frontend_test_bin() -> Path:
    return BUILD_DIR / "test" / "frontend_test"


def run_frontend_test(
    size: int, batch: int, iters: int, strategy: str, *, layout: str, dtype: str, verify_results: bool
) -> Path:
    binary = frontend_test_bin()
    env = cmake_env()
    env["TIMES_OVERRIDE"] = str(iters)
    if not verify_results:
        env["NO_CHECK"] = "1"
    else:
        env.pop("NO_CHECK", None)

    cmd = [
        str(binary),
        str(size),
        str(size),
        str(size),
        str(batch),
        "--layout",
        layout,
        "--type",
        dtype,
        "--strategy",
        strategy,
        "--backend",
        "ir2c",
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=binary.parent, env=env, check=True)

    stem = artifact_stem(strategy, layout, dtype, size, size, size, batch)
    return VERIFY_ARTIFACT_DIR / f"{stem}.verify.log"


def parse_verify_log(path: Path, m: int, n: int, k: int, batch: int, iters: int, verify_results: bool) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = LOG_PATTERN.findall(text)
    if not matches:
        raise RuntimeError(f"failed to parse gemm_kernel timing from {path}")
    if verify_results and "[PASS] gemm_kernel" not in text:
        raise RuntimeError(f"gemm_kernel validation did not pass in {path}")

    # verify.cpp currently runs TEST_GROUP(...) twice; use the last gemm_kernel timing so
    # the CSV matches the final frontend_test/verify printout instead of averaging both passes.
    total_ns = float(matches[-1][0])
    avg_ns = total_ns / float(iters * batch)
    if avg_ns <= 0.0:
        raise RuntimeError(f"invalid average timing parsed from {path}")

    gflops = (2.0 * float(m) * float(n) * float(k)) / avg_ns
    return avg_ns, gflops


def measure_case(
    size: int,
    batch: int,
    iters: int,
    strategy: str,
    *,
    layout: str,
    dtype: str,
    verify_results: bool,
    repeats: int,
) -> tuple[float, float]:
    best: tuple[float, float] | None = None
    for _ in range(repeats):
        log_path = run_frontend_test(
            size,
            batch,
            iters,
            strategy,
            layout=layout,
            dtype=dtype,
            verify_results=verify_results,
        )
        result = parse_verify_log(log_path, size, size, size, batch, iters, verify_results)
        if best is None or result[0] < best[0]:
            best = result
    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the IR2C backend GEMM benchmark via frontend_test + verify.")
    parser.add_argument("--sizes", default=DEFAULT_SIZES)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--strategy", default="costmodel")
    parser.add_argument("--layout", default="tn", choices=["nn", "nt", "tn", "tt"])
    parser.add_argument("--dtype", default="fp64", choices=["fp64", "fp32"])
    parser.add_argument("--repeats", type=int, default=1, help="repeat each size this many times and keep the best timing")
    parser.add_argument("--verify-results", action="store_true", help="keep verify.cpp correctness checks enabled")
    parser.add_argument(
        "--output",
        default=str(BENCH_OUT_DIR / "ir2c_gemm_perf.csv"),
        help="CSV output path",
    )
    args = parser.parse_args()

    if args.warmup != 10:
        raise SystemExit("run_ir2c_gemm_bench.py currently relies on verify.cpp's fixed 10-iteration warmup")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    ensure_cmake_target("frontend_test")
    VERIFY_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for size in parse_sizes(args.sizes):
        avg_ns, gflops = measure_case(
            size,
            args.batch,
            args.iters,
            args.strategy,
            layout=args.layout,
            dtype=args.dtype,
            verify_results=args.verify_results,
            repeats=args.repeats,
        )
        rows.append(
            {
                "backend": "ir2c",
                "m": str(size),
                "n": str(size),
                "k": str(size),
                "batch": str(args.batch),
                "warmup": str(args.warmup),
                "iters": str(args.iters),
                "avg_ns": f"{avg_ns:.3f}",
                "gflops": f"{gflops:.6f}",
                "checksum": "",
                "status": "ok",
                "note": (
                    f"backend=ir2c;strategy={args.strategy};layout={args.layout};dtype={args.dtype};"
                    f"storage=rowmajor;trans={layout_note_name(args.layout)};verify_harness=1;warmup_fixed=10;"
                    f"verify={'on' if args.verify_results else 'off'};avg_ns_per_gemm=from_verify_total_time;"
                    f"repeats={args.repeats};selection={'best_of_repeats' if args.repeats > 1 else 'single_run'}"
                ),
            }
        )

    output = Path(args.output)
    write_rows(output, rows)
    print_summary(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
