#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


FONT_SIZE = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot M3 layout ranking by GFLOPS.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="M3: Layout transformation cost changes kernel ranking")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def layout_to_gemm(layout: str) -> str:
    if layout == 'tt':
        return r'$A^TB^T$'
    elif layout == 'tn':
        return r'$A^TB$'
    elif layout == 'nt':
        return r'$AB^T$'
    elif layout == 'nn':
        return r'$AB$'
    else:
        raise ValueError(f"unknown layout: {layout}")

def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.output)

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")

    layouts = [layout_to_gemm(r["layout"]) for r in rows]
    gflops = [float(r["gflops"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    fig.patch.set_facecolor("#f6f4ec")

    colors = ["#005f73", "#0a9396", "#ca6702", "#bb3e03"]
    bars = ax.bar(layouts, gflops, color=colors[: len(layouts)], edgecolor="white", linewidth=1.0, alpha=0.94)

    for bar, value in zip(bars, gflops):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE,
        )

    ax.set_ylabel("GFLOPS", fontsize=FONT_SIZE)
    ax.set_xlabel("Layout", fontsize=FONT_SIZE)
    # ax.set_title(args.title, fontweight="bold", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")

    print(f"wrote {out_path}")
    print(f"wrote {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
