#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from gemm_bench_common import (
    BENCH_BIN_DIR,
    BENCH_OUT_DIR,
    BUILD_DIR,
    DEFAULT_BATCH,
    DEFAULT_ITERS,
    DEFAULT_SIZES,
    DEFAULT_WARMUP,
    REPO_ROOT,
    compile_cpp,
    ensure_cmake_target,
    print_summary,
    read_rows,
    run_binary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SMELT GEMM performance benchmark.")
    parser.add_argument("--sizes", default=DEFAULT_SIZES)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--strategy", default="AUTO")
    parser.add_argument("--layout", default="tn", choices=["nn", "nt", "tn", "tt"])
    parser.add_argument("--dtype", default="fp64", choices=["fp64", "fp32"])
    parser.add_argument("--verify-results", action="store_true", help="run a correctness check after timing")
    parser.add_argument(
        "--output",
        default=str(BENCH_OUT_DIR / "smelt_gemm_perf.csv"),
        help="CSV output path",
    )
    args = parser.parse_args()

    ensure_cmake_target("SMELT")

    source = REPO_ROOT / "test" / "perf" / "smelt_gemm_bench.cpp"
    binary = BENCH_BIN_DIR / "smelt_gemm_bench"
    compile_cpp(
        binary,
        source,
        [
            "-I",
            str(REPO_ROOT / "include"),
            str(BUILD_DIR / "src" / "libSMELT.a"),
            str(BUILD_DIR / "src" / "libIR.a"),
            "-march=armv8+nosve"
        ],
        require_sme=True,
        opt_level="-O3",
    )

    output = Path(args.output)
    run_binary(
        binary,
        output,
        sizes=args.sizes,
        batch=args.batch,
        warmup=args.warmup,
        iters=args.iters,
        extra_args=["--strategy", args.strategy, "--layout", args.layout, "--dtype", args.dtype]
        + (["--verify-results"] if args.verify_results else []),
    )
    rows = read_rows(output)
    print_summary(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
