#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M1 overlap benchmark and generate bar chart.")
    parser.add_argument("--build-dir", default="out/build/default", help="CMake build directory")
    parser.add_argument("--iters", type=int, default=300000)
    parser.add_argument("--warmup", type=int, default=6000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--csv", default="motivation/m1_overlap_results.csv")
    parser.add_argument("--plot", default="motivation/m1_overlap_results.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    build_dir = (repo_root / args.build_dir).resolve()
    exe = build_dir / "motivation" / "m1_overlap_bench"
    csv_path = (repo_root / args.csv).resolve()
    plot_path = (repo_root / args.plot).resolve()

    if not exe.exists():
        raise SystemExit(
            "m1_overlap_bench not found. Build it first with VS Code CMake Tools "
            f"(expected: {exe})"
        )

    if csv_path.exists():
        csv_path.unlink()

    common = [
        str(exe),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--csv",
        str(csv_path),
    ]

    subprocess.run(common + ["--target", "sve_mla"], check=True)
    subprocess.run(common + ["--target", "za_mova", "--append"], check=True)

    plot_script = repo_root / "motivation" / "plot_m1_overlap.py"
    subprocess.run(
        [
            "python3",
            str(plot_script),
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
