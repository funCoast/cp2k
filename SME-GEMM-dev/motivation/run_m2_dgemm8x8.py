#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M2 8x8 MOPA DGEMM benchmark and draw figure.")
    parser.add_argument("--build-dir", default="out/build/default", help="CMake build directory")
    parser.add_argument("--k-list", default="8,16,32,64,128,256,512")
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--csv", default="motivation/m2_dgemm8x8_results.csv")
    parser.add_argument("--plot", default="motivation/m2_dgemm8x8_results.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    build_dir = (repo_root / args.build_dir).resolve()
    exe = build_dir / "motivation" / "m2_dgemm8x8_mopa_bench"
    csv_path = (repo_root / args.csv).resolve()
    plot_path = (repo_root / args.plot).resolve()

    if not exe.exists():
        raise SystemExit(
            "m2_dgemm8x8_mopa_bench not found. Build it first with VS Code CMake Tools "
            f"(expected: {exe})"
        )

    subprocess.run(
        [
            str(exe),
            "--k-list",
            args.k_list,
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--csv",
            str(csv_path),
        ],
        check=True,
    )

    subprocess.run(
        [
            "python3",
            str(repo_root / "motivation" / "plot_m2_dgemm8x8.py"),
            "--csv",
            str(csv_path),
            "--output",
            str(plot_path),
        ],
        check=True,
    )

    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
