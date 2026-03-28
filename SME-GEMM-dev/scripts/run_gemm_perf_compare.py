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
    print_summary,
    python_exe,
    read_rows,
    script_path,
    write_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local and third-party GEMM benchmarks, then merge results.")
    parser.add_argument("--sizes", default=DEFAULT_SIZES)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--layout", default="tn", choices=["nn", "nt", "tn", "tt"])
    parser.add_argument("--dtype", default="fp64", choices=["fp64", "fp32"])
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--verify-results", action="store_true", help="enable correctness checks in all benchmarks")
    parser.add_argument("--smelt-strategy", default="COSTMODEL", help="strategy passed to run_smelt_gemm_bench.py")
    parser.add_argument(
        "--direct-strategy",
        default="costmodel",
        help="strategy passed to run_ir2a_gemm_bench.py and run_ir2c_gemm_bench.py",
    )
    parser.add_argument(
        "--direct-repeats",
        type=int,
        default=1,
        help="repeat each direct-kernel size this many times and keep the best timing",
    )
    parser.add_argument("--strategy", dest="smelt_strategy", help=argparse.SUPPRESS)
    parser.add_argument("--direct-backends", default="ir2a-asm,ir2a-bin,ir2c")
    parser.add_argument("--backends", default="openblas,libxsmm,armpl")
    parser.add_argument(
        "--output",
        default=str(BENCH_OUT_DIR / "gemm_perf_compare.csv"),
        help="merged CSV output path",
    )
    args = parser.parse_args()

    local_csv = BENCH_OUT_DIR / "smelt_gemm_perf.csv"
    direct_csvs: list[Path] = []
    thirdparty_csv = BENCH_OUT_DIR / "thirdparty_gemm_perf.csv"

    local_cmd = [
        python_exe(),
        script_path("run_smelt_gemm_bench.py"),
        "--sizes",
        args.sizes,
        "--batch",
        str(args.batch),
        "--layout",
        args.layout,
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--strategy",
        args.smelt_strategy,
        "--output",
        str(local_csv),
    ]
    if args.verify_results:
        local_cmd.append("--verify-results")
    subprocess.run(local_cmd, check=True)

    direct_backends = [backend.strip() for backend in args.direct_backends.split(",") if backend.strip()]
    for backend in direct_backends:
        if backend == "ir2c":
            output = BENCH_OUT_DIR / "ir2c_gemm_perf.csv"
            cmd = [
                python_exe(),
                script_path("run_ir2c_gemm_bench.py"),
                "--sizes",
                args.sizes,
                "--batch",
                str(args.batch),
                "--layout",
                args.layout,
                "--dtype",
                args.dtype,
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--strategy",
                args.direct_strategy,
                "--repeats",
                str(args.direct_repeats),
                "--output",
                str(output),
            ]
        elif backend in {"ir2a-asm", "ir2a-bin"}:
            output = BENCH_OUT_DIR / f"{backend.replace('-', '_')}_gemm_perf.csv"
            cmd = [
                python_exe(),
                script_path("run_ir2a_gemm_bench.py"),
                "--sizes",
                args.sizes,
                "--batch",
                str(args.batch),
                "--layout",
                args.layout,
                "--dtype",
                args.dtype,
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--strategy",
                args.direct_strategy,
                "--repeats",
                str(args.direct_repeats),
                "--backend",
                backend,
                "--output",
                str(output),
            ]
        else:
            raise SystemExit(f"unsupported direct backend: {backend}")

        if args.verify_results:
            cmd.append("--verify-results")
        subprocess.run(cmd, check=True)
        direct_csvs.append(output)

    thirdparty_cmd = [
        python_exe(),
        script_path("run_thirdparty_gemm_bench.py"),
        "--sizes",
        args.sizes,
        "--batch",
        str(args.batch),
        "--layout",
        args.layout,
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--backends",
        args.backends,
        "--output",
        str(thirdparty_csv),
    ]
    if args.verify_results:
        thirdparty_cmd.append("--verify-results")
    subprocess.run(thirdparty_cmd, check=True)

    merged = read_rows(local_csv)
    for path in direct_csvs:
        merged.extend(read_rows(path))
    merged.extend(read_rows(thirdparty_csv))
    output = Path(args.output)
    write_rows(output, merged)
    print_summary(merged)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
