#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

from gemm_bench_common import (
    BENCH_OUT_DIR,
    DEFAULT_ITERS,
    DEFAULT_SIZES,
    DEFAULT_WARMUP,
    python_exe,
    python_with_module,
    script_path,
)


def parse_dtypes(text: str) -> list[str]:
    values = [part.strip().lower() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("dtypes must not be empty")
    for value in values:
        if value not in {"fp32", "fp64"}:
            raise ValueError(f"unsupported dtype: {value}")
    return values


def parse_layouts(text: str) -> list[str]:
    layouts = [part.strip().lower() for part in text.split(",") if part.strip()]
    if not layouts:
        raise ValueError("layouts must not be empty")
    for layout in layouts:
        if layout not in {"nn", "nt", "tn", "tt"}:
            raise ValueError(f"unsupported layout: {layout}")
    return layouts


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dtype", "batch", "layout", "csv", "png", "svg", "title"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full GEMM comparison suite for fp32/fp64 across nn/nt/tn/tt and generate per-mode plots."
    )
    parser.add_argument("--sizes", default=DEFAULT_SIZES)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dtypes", default="fp32,fp64")
    parser.add_argument("--layouts", default="nn,nt,tn,tt")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--verify-results", action="store_true")
    parser.add_argument("--smelt-strategy", default="COSTMODEL")
    parser.add_argument("--direct-strategy", default="costmodel")
    parser.add_argument("--direct-repeats", type=int, default=10)
    parser.add_argument("--direct-backends", default="ir2a-asm,ir2a-bin,ir2c")
    parser.add_argument("--backends", default="openblas,libxsmm,armpl")
    parser.add_argument("--output-dir", default=str(BENCH_OUT_DIR / "mode_matrix_suite"))
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtypes = parse_dtypes(args.dtypes)
    layouts = parse_layouts(args.layouts)
    plot_python = python_with_module("matplotlib") if not args.skip_plot else ""

    manifest_rows: list[dict[str, str]] = []
    for dtype in dtypes:
        for layout in layouts:
            mode_tag = f"{dtype}_{layout}"
            mode_dir = output_dir / mode_tag
            mode_dir.mkdir(parents=True, exist_ok=True)
            csv_output = mode_dir / "gemm_perf_compare.csv"
            plot_output = mode_dir / "gemm_perf_compare.png"
            title = f"SME-GEMM Small Matrix Performance Comparison ({dtype.upper()}, {layout.upper()}, batch={args.batch})"

            compare_cmd = [
                python_exe(),
                script_path("run_gemm_perf_compare.py"),
                "--sizes",
                args.sizes,
                "--batch",
                str(args.batch),
                "--dtype",
                dtype,
                "--layout",
                layout,
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
                str(csv_output),
            ]
            if args.verify_results:
                compare_cmd.append("--verify-results")
            print("+", " ".join(compare_cmd))
            subprocess.run(compare_cmd, check=True)

            if not args.skip_plot:
                plot_cmd = [
                    plot_python,
                    script_path("generate_gemm_perf_plot.py"),
                    "--csv",
                    str(csv_output),
                    "--output",
                    str(plot_output),
                    "--title",
                    title,
                ]
                print("+", " ".join(plot_cmd))
                subprocess.run(plot_cmd, check=True)

            manifest_rows.append(
                {
                    "dtype": dtype,
                    "batch": str(args.batch),
                    "layout": layout,
                    "csv": str(csv_output),
                    "png": str(plot_output),
                    "svg": str(plot_output.with_suffix(".svg")),
                    "title": title,
                }
            )

    manifest_path = output_dir / "suite_manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
