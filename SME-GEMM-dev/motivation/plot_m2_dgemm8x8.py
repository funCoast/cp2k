#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


FONT_SIZE = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot M2 K-sweep GFLOPS and mova ratio.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="M2: 16x16 MOPA DGEMM (A col, B row, C row)")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.output)

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")

    ks = [int(r["K"]) for r in rows]
    gflops_full = [float(r["gflops_full"]) for r in rows]
    gflops_compute = [float(r["gflops_compute"]) for r in rows]
    mova_ratio = [100.0 * float(r["mova_ratio"]) for r in rows]
    x = list(range(len(ks)))

    fig, ax1 = plt.subplots(figsize=(6, 5.5))
    fig.patch.set_facecolor("#f9f5ec")

    ax1.bar(x, gflops_full, width=0.78, color="#005f73", alpha=0.85, label="GFLOPS")
    # ax1.plot(ks, gflops_compute, marker="s", linewidth=2.2, color="#ca6702", label="GFLOPS (compute-only)")
    ax1.set_xlabel("K", fontsize=FONT_SIZE)
    ax1.set_ylabel("GFLOPS", fontsize=FONT_SIZE)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(k) for k in ks])
    ax1.grid(True, linestyle="--", linewidth=0.8, alpha=0.3)
    ax1.tick_params(axis="both", labelsize=FONT_SIZE)

    ax2 = ax1.twinx()
    ax2.plot(x, mova_ratio, marker="^", linewidth=2.0, color="#2a9d8f", label="MOVA Share (%)")
    ax2.set_ylabel("MOVA Share (%)", fontsize=FONT_SIZE)
    ax2.tick_params(axis="y", labelsize=FONT_SIZE)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False, fontsize=FONT_SIZE)

    # ax1.set_title(args.title, fontweight="bold", fontsize=FONT_SIZE)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")

    print(f"wrote {out_path}")
    print(f"wrote {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
