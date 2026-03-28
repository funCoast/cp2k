#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import statistics

METRIC_RE = re.compile(r"^([A-Za-z0-9_]+)\s*=\s*([0-9eE+\-.]+)\s*$")


def run_overlap_bench(bin_path: str, iters: int, env: dict) -> dict[str, float]:
    cmd = [bin_path, str(iters)]
    proc = subprocess.run(
        cmd,
        cwd=os.path.dirname(bin_path),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    out = proc.stdout
    if proc.returncode != 0:
        raise RuntimeError(f"overlap_bench failed\n{out}\n{proc.stderr}")

    metrics: dict[str, float] = {}
    for line in out.splitlines():
        m = METRIC_RE.match(line.strip())
        if not m:
            continue
        metrics[m.group(1)] = float(m.group(2))

    if "sve1_ns" not in metrics or "mopa1_ns" not in metrics:
        raise RuntimeError(f"Failed to parse overlap_bench output\n{out}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark overlap_bench for cost model calibration")
    parser.add_argument("--build-dir", default="out/build/default")
    parser.add_argument("--iters", type=int, default=300000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--csv-output", default="", help="Optional path to save CSV output")
    args = parser.parse_args()

    bench_bin = os.path.join(args.build_dir, "microbench", "overlap_bench")
    if not os.path.exists(bench_bin):
        raise FileNotFoundError(f"overlap_bench not found: {bench_bin}")

    env = os.environ.copy()
    old_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = "/home/dsk/anaconda3/envs/sme/lib" + (":" + old_ld if old_ld else "")

    all_runs: dict[str, list[float]] = {}
    for _ in range(max(1, args.repeats)):
        metrics = run_overlap_bench(bench_bin, max(1, args.iters), env)
        for key, value in metrics.items():
            all_runs.setdefault(key, []).append(value)

    merged = {k: statistics.median(vs) for k, vs in all_runs.items()}
    rows = ["metric,value"] + [f"{k},{merged[k]:.6f}" for k in sorted(merged.keys())]

    csv_text = "\n".join(rows)
    print(csv_text)

    if args.csv_output:
        out_path = os.path.abspath(args.csv_output)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(csv_text + "\n")
        print(f"Saved CSV to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
