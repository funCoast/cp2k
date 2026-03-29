#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


BACKENDS = ("LIBXSMM", "SME")


@dataclass
class BackendReport:
    name: str
    log_path: Path
    exists: bool = False
    state: str = "missing"
    launched_dirs: Optional[int] = None
    completed_dirs: Optional[int] = None
    last_completed_workdir: Optional[str] = None
    status: Optional[str] = None
    total_tests: Optional[int] = None
    correct_tests: Optional[int] = None
    failed_tests: Optional[int] = None
    wrong_tests: Optional[int] = None
    wall_time: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize CP2K benchmark/regtest logs without running anything."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        help=(
            "Run directory, bench-logs directory, or a backend log file. "
            "If omitted, the latest bench-logs/run-* under the repo root is used."
        ),
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep refreshing the summary until both backends finish.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Refresh interval in seconds when --watch is used.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_run_dir(arg: Optional[str]) -> Path:
    root = repo_root()
    bench_root = root / "bench-logs"

    if arg:
        candidate = Path(arg).expanduser().resolve()
        if candidate.is_file():
            candidate = candidate.parent
        if candidate.name == "bench-logs":
            bench_root = candidate
        elif candidate.name.startswith("run-") and candidate.is_dir():
            return candidate
        elif candidate.is_dir():
            bench_root = candidate

    if not bench_root.exists():
        raise FileNotFoundError(f"No bench-logs directory found at {bench_root}")

    run_dirs = [p for p in bench_root.glob("run-*") if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run-* directories found under {bench_root}")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def parse_summary_file(summary_path: Path) -> dict[str, dict[str, str] | str]:
    parsed: dict[str, dict[str, str] | str] = {}
    if not summary_path.exists():
        return parsed

    current_section: Optional[str] = None
    comparison_lines: list[str] = []
    for raw_line in summary_path.read_text(encoding="utf8", errors="replace").splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("Run directory:"):
            parsed["run_directory"] = line.split(":", 1)[1].strip()
            current_section = None
            continue
        if line in (*BACKENDS, "Comparison"):
            current_section = line
            parsed.setdefault(current_section, {})
            continue
        if current_section == "Comparison" and line.startswith("  "):
            comparison_lines.append(line.strip())
            continue
        if current_section and line.startswith("  ") and ":" in line:
            key, value = line.strip().split(":", 1)
            section = parsed.setdefault(current_section, {})
            assert isinstance(section, dict)
            section[key.strip()] = value.strip()
            continue
    if comparison_lines:
        parsed["Comparison"] = " ".join(comparison_lines)
    return parsed


def parse_backend_log(log_path: Path) -> BackendReport:
    report = BackendReport(name=log_path.stem, log_path=log_path)
    if not log_path.exists():
        return report

    report.exists = True
    text = log_path.read_text(encoding="utf8", errors="replace")
    lines = text.splitlines()

    launch_re = re.compile(r"^Launched (\d+) test directories")
    done_re = re.compile(r"^<<< (.+) \((\d+) of (\d+)\) done in ([0-9.]+) sec$")
    status_re = re.compile(r"^Status:\s+(.*)$")
    total_re = re.compile(r"^Total number of\s+tests\s+(\d+)$")
    correct_re = re.compile(r"^Number of\s+CORRECT\s+tests\s+(\d+)$")
    failed_re = re.compile(r"^Number of\s+FAILED\s+tests\s+(\d+)$")
    wrong_re = re.compile(r"^Number of\s+WRONG\s+tests\s+(\d+)$")
    ended_marker = "*************************** Testing ended ******************************"

    for line in lines:
        if m := launch_re.match(line):
            report.launched_dirs = int(m.group(1))
            report.state = "running"
        elif m := done_re.match(line):
            report.completed_dirs = int(m.group(2))
            report.last_completed_workdir = m.group(1)
            report.state = "running"
        elif m := status_re.match(line):
            report.status = m.group(1).strip()
        elif m := total_re.match(line):
            report.total_tests = int(m.group(1))
        elif m := correct_re.match(line):
            report.correct_tests = int(m.group(1))
        elif m := failed_re.match(line):
            report.failed_tests = int(m.group(1))
        elif m := wrong_re.match(line):
            report.wrong_tests = int(m.group(1))

    if ended_marker in text:
        report.state = "completed"
    elif report.state != "missing":
        report.state = "running" if report.launched_dirs is not None else "starting"

    return report


def apply_summary(report: BackendReport, summary: dict[str, dict[str, str] | str]) -> BackendReport:
    section = summary.get(report.name)
    if isinstance(section, dict):
        report.status = section.get("status", report.status)
        report.total_tests = int(section["total tests"]) if section.get("total tests", "").isdigit() else report.total_tests
        report.correct_tests = int(section["correct tests"]) if section.get("correct tests", "").isdigit() else report.correct_tests
        report.failed_tests = int(section["failed tests"]) if section.get("failed tests", "").isdigit() else report.failed_tests
        report.wrong_tests = int(section["wrong tests"]) if section.get("wrong tests", "").isdigit() else report.wrong_tests
        wall_time = section.get("wall time")
        if wall_time:
            report.wall_time = wall_time
        if report.status and report.status.upper() in {"OK", "FAILED"}:
            report.state = "completed"
    return report


def find_summary_text(summary: dict[str, dict[str, str] | str]) -> Optional[str]:
    section = summary.get("Comparison")
    if isinstance(section, str) and section.strip():
        return section.strip()
    if isinstance(section, dict):
        return " ".join(f"{k}: {v}" for k, v in section.items())
    return None


def print_snapshot(run_dir: Path) -> bool:
    summary_path = run_dir / "summary.txt"
    summary = parse_summary_file(summary_path)

    reports = []
    for backend in BACKENDS:
        report = parse_backend_log(run_dir / f"{backend}.log")
        report = apply_summary(report, summary)
        reports.append(report)

    both_done = all(r.state == "completed" for r in reports)

    print(f"Run directory: {run_dir}")
    print(f"Summary file: {summary_path if summary_path.exists() else 'not yet written'}")
    print()

    for report in reports:
        print(f"{report.name}:")
        print(f"  state:        {report.state}")
        print(f"  log:          {report.log_path}")
        if report.launched_dirs is not None:
            print(f"  launched dirs:{report.launched_dirs:>8}")
        if report.completed_dirs is not None:
            total = report.launched_dirs if report.launched_dirs is not None else "?"
            print(f"  completed dirs:{report.completed_dirs:>8} / {total}")
        if report.last_completed_workdir:
            print(f"  last done:    {report.last_completed_workdir}")
        print(f"  status:       {report.status or 'unknown'}")
        print(f"  total tests:  {report.total_tests if report.total_tests is not None else 'unknown'}")
        print(f"  correct:      {report.correct_tests if report.correct_tests is not None else 'unknown'}")
        print(f"  failed:       {report.failed_tests if report.failed_tests is not None else 'unknown'}")
        print(f"  wrong:        {report.wrong_tests if report.wrong_tests is not None else 'unknown'}")
        print(f"  wall time:    {report.wall_time or 'unknown'}")
        print()

    comparison = find_summary_text(summary)
    if comparison:
        print(f"Comparison: {comparison}")
    else:
        libxsmm = next((r for r in reports if r.name == "LIBXSMM"), None)
        sme = next((r for r in reports if r.name == "SME"), None)
        if libxsmm and sme and libxsmm.wall_time and sme.wall_time:
            try:
                lib_time = float(libxsmm.wall_time.rstrip("s"))
                sme_time = float(sme.wall_time.rstrip("s"))
                delta = sme_time - lib_time
                pct = (delta / lib_time * 100.0) if lib_time else 0.0
                if delta < 0:
                    comparison = f"SME is faster by {-delta:.2f}s ({-pct:.1f}%)"
                elif delta > 0:
                    comparison = f"SME is slower by {delta:.2f}s ({pct:.1f}%)"
                else:
                    comparison = "SME and LIBXSMM have identical wall time"
            except ValueError:
                comparison = None
        if comparison:
            print(f"Comparison: {comparison}")
        else:
            print("Comparison: not available yet")

    return both_done


def main() -> int:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)

    if not args.watch:
        print_snapshot(run_dir)
        return 0

    try:
        while True:
            print("\033[2J\033[H", end="")
            print_snapshot(run_dir)
            if all((run_dir / f"{backend}.log").exists() for backend in BACKENDS):
                # Keep watching until both logs contain a terminal status.
                reports = [parse_backend_log(run_dir / f"{backend}.log") for backend in BACKENDS]
                reports = [apply_summary(r, parse_summary_file(run_dir / "summary.txt")) for r in reports]
                if all(r.state == "completed" for r in reports):
                    return 0
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
