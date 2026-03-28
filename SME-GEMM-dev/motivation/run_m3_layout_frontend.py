#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path

LAYOUTS = ["tt", "nn", "tn", "nt"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M3 layout transformation experiment via frontend_test and collect GFLOPS."
    )
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--strategy", default="mopa")
    parser.add_argument("--backend", default="ir2c")
    parser.add_argument("--layouts", default=",".join(LAYOUTS), help="Comma-separated layouts, e.g. tt,nn,tn,nt")
    parser.add_argument("--csv", default="motivation/m3_layout_ranking_results.csv")
    return parser.parse_args()


def parse_layouts(text: str) -> list[str]:
    layouts = [x.strip().lower() for x in text.split(",") if x.strip()]
    for layout in layouts:
        if layout not in LAYOUTS:
            raise ValueError(f"unsupported layout: {layout}")
    if not layouts:
        raise ValueError("no layouts provided")
    return layouts


def run_frontend_test(repo_root: Path, args: argparse.Namespace, layout: str) -> Path:
    cmd = [
        str(repo_root / "out" / "build" / "default" / "test" / "frontend_test"),
        str(args.m),
        str(args.n),
        str(args.k),
        "-b",
        args.backend,
        "-s",
        args.strategy,
        "-l",
        layout,
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)

    log_name = f"{args.strategy}__{args.backend}__{layout}__fp64__M{args.m}_N{args.n}_K{args.k}_B8.verify.log"
    log_path = repo_root / "test" / "verify" / "artifacts" / log_name
    if not log_path.exists():
        raise FileNotFoundError(f"verify log not found: {log_path}")
    return log_path


def extract_gflops_from_verify_log(log_path: Path) -> float:
    # Select GFLOPS in the gemm_kernel section (not batch_dgemm_* section).
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r"gemm_kernel\s+Time\s*=\s*[^\n]+\nGFLOPS\s*=\s*([0-9.eE+-]+)", re.MULTILINE)
    matches = pattern.findall(text)
    if not matches:
        raise RuntimeError(f"failed to parse gemm_kernel GFLOPS from {log_path}")
    values = [float(x) for x in matches]
    return max(values)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["layout", "gflops", "m", "n", "k", "backend", "strategy"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    layouts = parse_layouts(args.layouts)
    repo_root = Path(__file__).resolve().parent.parent

    rows: list[dict[str, str]] = []
    for layout in layouts:
        log_path = run_frontend_test(repo_root, args, layout)
        gflops = extract_gflops_from_verify_log(log_path)
        rows.append(
            {
                "layout": layout,
                "gflops": f"{gflops:.6f}",
                "m": str(args.m),
                "n": str(args.n),
                "k": str(args.k),
                "backend": args.backend,
                "strategy": args.strategy,
            }
        )
        print(f"layout={layout}, gflops={gflops:.6f}, log={log_path}")

    csv_path = (repo_root / args.csv).resolve()
    write_csv(csv_path, rows)
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
