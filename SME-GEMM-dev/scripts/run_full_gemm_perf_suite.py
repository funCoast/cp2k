#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from gemm_bench_common import (
    BENCH_OUT_DIR,
    DEFAULT_BATCH,
    DEFAULT_ITERS,
    DEFAULT_SIZES,
    DEFAULT_WARMUP,
    python_exe,
    python_with_module,
    script_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full GEMM performance suite, merge benchmark CSVs, and generate the grouped bar chart."
    )
    parser.add_argument("--sizes", default=DEFAULT_SIZES)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--verify-results", action="store_true", help="enable correctness checks in all benchmarks")
    parser.add_argument("--smelt-strategy", default="COSTMODEL")
    parser.add_argument("--direct-strategy", default="costmodel")
    parser.add_argument("--direct-repeats", type=int, default=1)
    parser.add_argument("--direct-backends", default="ir2a-asm,ir2a-bin,ir2c")
    parser.add_argument("--backends", default="openblas,libxsmm,armpl")
    parser.add_argument("--csv-output", default=str(BENCH_OUT_DIR / "gemm_perf_compare.csv"))
    parser.add_argument("--plot-output", default=str(BENCH_OUT_DIR / "gemm_perf_compare.png"))
    parser.add_argument("--title", default="SME-GEMM Small Matrix Performance Comparison")
    parser.add_argument("--skip-plot", action="store_true", help="only run benchmarks and write the merged CSV")
    args = parser.parse_args()

    compare_cmd = [
        python_exe(),
        script_path("run_gemm_perf_compare.py"),
        "--sizes",
        args.sizes,
        "--batch",
        str(args.batch),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--smelt-strategy",
        args.smelt_strategy,
        "--direct-strategy",
        args.direct_strategy,
        "--direct-repeats",
        str(args.direct_repeats),
        "--direct-backends",
        args.direct_backends,
        "--backends",
        args.backends,
        "--output",
        args.csv_output,
    ]
    if args.verify_results:
        compare_cmd.append("--verify-results")
    print("+", " ".join(compare_cmd))
    subprocess.run(compare_cmd, check=True)

    if args.skip_plot:
        print(f"skipped plot generation; merged CSV is at {args.csv_output}")
        return

    plot_python = python_with_module("matplotlib")
    plot_cmd = [
        plot_python,
        script_path("generate_gemm_perf_plot.py"),
        "--csv",
        args.csv_output,
        "--output",
        args.plot_output,
        "--title",
        args.title,
    ]
    print("+", " ".join(plot_cmd))
    subprocess.run(plot_cmd, check=True)

    print(f"wrote merged CSV: {Path(args.csv_output)}")
    print(f"wrote plot: {Path(args.plot_output)}")
    print(f"wrote plot: {Path(args.plot_output).with_suffix('.svg')}")


if __name__ == "__main__":
    main()
