#!/usr/bin/env python3
import argparse
import csv
import json
import os
import statistics
from typing import Dict, List, Tuple

DEFAULTS: Dict[str, object] = {
    "SVE_FMLA_COST": 1.00,
    "SME_MOPA_COST": 1.05,
    "SME2_FMLA_COST": 1.20,
    "SVE_LOAD_COST": 0.30,
    "SVE_CONCAT_COST": 0.20,
    "SVE_STORE_COST": 0.35,
    "SME_MOVA_COST": 0.45,
    "SCALAR_LOAD_COST": 0.40,
    "SCALAR_STORE_COST": 0.45,
    "SCALAR_COMPUTE_COST": 1.60,
    "SVE_INSNS_COVERED_PER_MOPA": 3,
    "SME_MOVE_INSNS_COVERED_PER_MOPA": 1,
    "SVE_LOAD_INSNS_COVERED_PER_MOPA": 1,
    "SVE_STORE_INSNS_COVERED_PER_MOPA": 1,
    "SVE_CONCAT_INSNS_COVERED_PER_MOPA": 1,
    "MOPA_CHAIN_WIDTH": 8,
    "MOPA_CHAIN_HEAD": 2,
    "MEM_COMPUTE_OVERLAP": 0.25,
    "ENABLE_SME2": 0,
}

FLOAT_KEYS = {
    "SVE_FMLA_COST",
    "SME_MOPA_COST",
    "SME2_FMLA_COST",
    "SVE_LOAD_COST",
    "SVE_CONCAT_COST",
    "SVE_STORE_COST",
    "SME_MOVA_COST",
    "SCALAR_LOAD_COST",
    "SCALAR_STORE_COST",
    "SCALAR_COMPUTE_COST",
    "MEM_COMPUTE_OVERLAP",
}

INT_KEYS = {
    "SVE_INSNS_COVERED_PER_MOPA",
    "SME_MOVE_INSNS_COVERED_PER_MOPA",
    "SVE_LOAD_INSNS_COVERED_PER_MOPA",
    "SVE_STORE_INSNS_COVERED_PER_MOPA",
    "SVE_CONCAT_INSNS_COVERED_PER_MOPA",
    "MOPA_CHAIN_WIDTH",
    "MOPA_CHAIN_HEAD",
    "ENABLE_SME2",
}


def read_strategy_csv(path: str) -> Dict[str, List[float]]:
    by_strategy: Dict[str, List[float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strategy = (row.get("strategy") or "").strip().lower()
            kernel_time = row.get("kernel_time_ns")
            if not strategy or kernel_time is None:
                continue
            try:
                t = float(kernel_time)
            except ValueError:
                continue
            if t <= 0:
                continue
            by_strategy.setdefault(strategy, []).append(t)
    return by_strategy


def read_metric_csv(path: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("metric") or "").strip()
            value = row.get("value")
            if not key or value is None:
                continue
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
    return metrics


def detect_csv_format(path: str) -> Tuple[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().lower()

    if header == "case,strategy,kernel_time_ns":
        return ("strategy", read_strategy_csv(path))
    if header == "metric,value":
        return ("metric", read_metric_csv(path))
    raise ValueError(f"Unsupported benchmark CSV format: {path}")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def derive_from_bench(values: Dict[str, object], by_strategy: Dict[str, List[float]]) -> None:
    sve = by_strategy.get("sve", [])
    mopa = by_strategy.get("mopa", [])
    sme2 = by_strategy.get("sme2", [])

    if sve and mopa:
        sve_med = statistics.median(sve)
        mopa_med = statistics.median(mopa)
        ratio = mopa_med / sve_med if sve_med > 0 else DEFAULTS["SME_MOPA_COST"]
        values["SME_MOPA_COST"] = round(clamp(ratio, 0.1, 10.0), 4)

    if sve and sme2:
        sve_med = statistics.median(sve)
        sme2_med = statistics.median(sme2)
        ratio = sme2_med / sve_med if sve_med > 0 else DEFAULTS["SME2_FMLA_COST"]
        values["SME2_FMLA_COST"] = round(clamp(ratio, 0.1, 10.0), 4)
        values["ENABLE_SME2"] = 1


def derive_from_overlap_metrics(values: Dict[str, object], metrics: Dict[str, float]) -> None:
    sve_candidates = []
    mopa_candidates = []

    for n in (1, 2, 4, 8):
        key = f"sve{n}_ns"
        if key in metrics and metrics[key] > 0:
            sve_candidates.append(metrics[key] / n)

    for n in (1, 2, 4):
        key = f"mopa{n}_ns"
        if key in metrics and metrics[key] > 0:
            mopa_candidates.append(metrics[key] / n)

    if sve_candidates and mopa_candidates:
        sve_unit = statistics.median(sve_candidates)
        mopa_unit = statistics.median(mopa_candidates)
        ratio = mopa_unit / sve_unit if sve_unit > 0 else DEFAULTS["SME_MOPA_COST"]
        values["SVE_FMLA_COST"] = 1.0
        values["SME_MOPA_COST"] = round(clamp(ratio, 0.1, 10.0), 4)

    covered_estimates = []
    for n in (1, 2, 4, 8):
        key = f"hidden_sve_ratio_n{n}"
        if key not in metrics:
            continue
        ratio = clamp(metrics[key], 0.0, 1.0)
        covered_estimates.append(n * ratio)
    if covered_estimates:
        covered = statistics.mean(covered_estimates)
        values["SVE_INSNS_COVERED_PER_MOPA"] = int(round(clamp(covered, 0.0, 8.0)))

    hidden_mova = metrics.get("hidden_mova_write_ratio")
    if hidden_mova is not None:
        values["SME_MOVE_INSNS_COVERED_PER_MOPA"] = int(round(clamp(hidden_mova, 0.0, 1.0) * 2.0))

    hidden_ld = metrics.get("hidden_ld1_ratio")
    if hidden_ld is not None:
        values["SVE_LOAD_INSNS_COVERED_PER_MOPA"] = int(round(clamp(hidden_ld, 0.0, 1.0) * 2.0))

    hidden_st = metrics.get("hidden_st1_ratio")
    if hidden_st is not None:
        values["SVE_STORE_INSNS_COVERED_PER_MOPA"] = int(round(clamp(hidden_st, 0.0, 1.0) * 2.0))

    hidden_sel = metrics.get("hidden_sel_ratio")
    if hidden_sel is not None:
        values["SVE_CONCAT_INSNS_COVERED_PER_MOPA"] = int(round(clamp(hidden_sel, 0.0, 1.0) * 2.0))

    if metrics.get("sme2_enabled", 0.0) > 0.5:
        sme2_mla1 = metrics.get("sme2_mla1_ns")
        sve1 = metrics.get("sve1_ns")
        if sme2_mla1 is not None and sve1 is not None and sve1 > 0.0:
            values["SME2_FMLA_COST"] = round(clamp(sme2_mla1 / sve1, 0.1, 10.0), 4)
            values["ENABLE_SME2"] = 1

    speedup4 = metrics.get("mopa_chain_speedup_4")
    if speedup4 is not None and speedup4 > 1.0:
        hidden_slots = 4.0 - (4.0 / speedup4)
        if hidden_slots > 0.1:
            width = int(round(4.0 / hidden_slots))
            values["MOPA_CHAIN_WIDTH"] = max(1, min(32, width))


def normalize_key(raw_key: str) -> str:
    key = raw_key.strip()
    if key.startswith("COSTCFG_"):
        key = key[len("COSTCFG_") :]
    return key.upper()


def apply_overrides(values: Dict[str, object], path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Override JSON must be an object")

    for raw_key, raw_value in data.items():
        key = normalize_key(raw_key)
        if key not in values:
            raise KeyError(f"Unknown cost config key: {raw_key}")
        if key in FLOAT_KEYS:
            values[key] = float(raw_value)
        else:
            values[key] = int(raw_value)


def write_header(values: Dict[str, object], out_path: str, source_csv: str, source_json: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lines = [
        "#pragma once",
        "",
        "// Auto-generated by microbench/generate_cost_config.py",
    ]
    if source_csv:
        lines.append(f"// Derived benchmark CSV: {source_csv}")
    if source_json:
        lines.append(f"// Applied override JSON: {source_json}")
    lines.append("")

    ordered_keys = [
        "SVE_FMLA_COST",
        "SME_MOPA_COST",
        "SME2_FMLA_COST",
        "SVE_LOAD_COST",
        "SVE_CONCAT_COST",
        "SVE_STORE_COST",
        "SME_MOVA_COST",
        "SCALAR_LOAD_COST",
        "SCALAR_STORE_COST",
        "SCALAR_COMPUTE_COST",
        "SVE_INSNS_COVERED_PER_MOPA",
        "SME_MOVE_INSNS_COVERED_PER_MOPA",
        "SVE_LOAD_INSNS_COVERED_PER_MOPA",
        "SVE_STORE_INSNS_COVERED_PER_MOPA",
        "SVE_CONCAT_INSNS_COVERED_PER_MOPA",
        "MOPA_CHAIN_WIDTH",
        "MOPA_CHAIN_HEAD",
        "MEM_COMPUTE_OVERLAP",
        "ENABLE_SME2",
    ]

    for key in ordered_keys:
        value = values[key]
        if key in FLOAT_KEYS:
            lines.append(f"#define COSTCFG_{key} {float(value):.6f}")
        else:
            lines.append(f"#define COSTCFG_{key} {int(value)}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate include/cost_config.h from benchmark data")
    parser.add_argument("--output", required=True, help="Output header path")
    parser.add_argument("--bench-csv", default="", help="CSV from kernel_cost_bench.py")
    parser.add_argument("--override-json", default="", help="JSON overrides for any field")
    args = parser.parse_args()

    values: Dict[str, object] = dict(DEFAULTS)

    if args.bench_csv:
        fmt, payload = detect_csv_format(args.bench_csv)
        if fmt == "strategy":
            derive_from_bench(values, payload)
        else:
            derive_from_overlap_metrics(values, payload)

    if args.override_json:
        apply_overrides(values, args.override_json)

    write_header(values, args.output, args.bench_csv, args.override_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
