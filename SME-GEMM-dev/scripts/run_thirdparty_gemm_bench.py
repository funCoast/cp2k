#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from gemm_bench_common import (
    BENCH_BIN_DIR,
    BENCH_OUT_DIR,
    DEFAULT_BATCH,
    DEFAULT_ITERS,
    DEFAULT_SIZES,
    DEFAULT_WARMUP,
    REPO_ROOT,
    armpl_link_args,
    compile_cpp,
    detect_gfortran_runtime_dir,
    ensure_cmake_target,
    find_armpl_config,
    print_summary,
    read_rows,
    run_binary,
    skipped_rows,
    write_rows,
)


def detect_openblas_library() -> Path | None:
    root = REPO_ROOT / "thirdparty" / "OpenBLAS"
    actual_dylibs = [
        path
        for path in sorted(root.glob("libopenblas*.dylib"))
        if path.name not in {"libopenblas.dylib", "libopenblas.0.dylib"}
    ]
    if actual_dylibs:
        for alias in (root / "libopenblas.0.dylib", root / "libopenblas.dylib"):
            if not alias.exists():
                if alias.is_symlink():
                    alias.unlink()
                alias.symlink_to(actual_dylibs[0].name)

    candidates = [root / "libopenblas.dylib"]
    candidates.extend(sorted(root.glob("libopenblas*.dylib")))
    candidates.append(root / "libopenblas.a")
    candidates.extend(sorted(root.glob("libopenblas*.a")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_and_run_openblas(
    output: Path, sizes: str, batch: int, warmup: int, iters: int, verify_results: bool, layout: str, dtype: str
) -> list[dict[str, str]]:
    runtime_dir = detect_gfortran_runtime_dir()
    if runtime_dir is None:
        return skipped_rows("openblas", sizes, batch, warmup, iters, "gfortran runtime not found")
    openblas_lib = detect_openblas_library()
    if openblas_lib is None:
        return skipped_rows("openblas", sizes, batch, warmup, iters, "OpenBLAS library artifact not found")

    source = REPO_ROOT / "test" / "perf" / "thirdparty_gemm_bench.cpp"
    binary = BENCH_BIN_DIR / "openblas_gemm_bench"
    link_args = [
        "-DBENCH_BACKEND_OPENBLAS=1",
        "-I",
        str(REPO_ROOT / "thirdparty" / "OpenBLAS"),
        str(openblas_lib),
    ]
    if openblas_lib.suffix == ".dylib":
        link_args.extend([f"-Wl,-rpath,{openblas_lib.parent}"])
    link_args.extend(
        [
            f"-L{runtime_dir}",
            f"-Wl,-rpath,{runtime_dir}",
            "-lgfortran",
            "-lm",
            "-lpthread",
        ]
    )
    compile_cpp(
        binary,
        source,
        link_args,
    )
    run_binary(
        binary,
        output,
        sizes=sizes,
        batch=batch,
        warmup=warmup,
        iters=iters,
        extra_args=["--layout", layout, "--dtype", dtype] + (["--verify-results"] if verify_results else []),
    )
    return read_rows(output)


def build_and_run_libxsmm(
    output: Path, sizes: str, batch: int, warmup: int, iters: int, verify_results: bool, layout: str, dtype: str
) -> list[dict[str, str]]:
    source = REPO_ROOT / "test" / "perf" / "thirdparty_gemm_bench.cpp"
    binary = BENCH_BIN_DIR / "libxsmm_gemm_bench"
    compile_cpp(
        binary,
        source,
        [
            "-DBENCH_BACKEND_LIBXSMM=1",
            "-I",
            str(REPO_ROOT / "thirdparty" / "_build" / "libxsmm" / "include"),
            "-I",
            str(REPO_ROOT / "thirdparty" / "libxsmm/src/template"),
            str(REPO_ROOT / "thirdparty" / "_build" / "libxsmm" / "lib" / "libxsmm.a"),
            "-lm",
            "-lpthread",
        ],
    )
    run_binary(
        binary,
        output,
        sizes=sizes,
        batch=batch,
        warmup=warmup,
        iters=iters,
        extra_args=["--layout", layout, "--dtype", dtype] + (["--verify-results"] if verify_results else []),
    )
    return read_rows(output)


def build_and_run_armpl(
    output: Path, sizes: str, batch: int, warmup: int, iters: int, verify_results: bool, layout: str, dtype: str
) -> list[dict[str, str]]:
    armpl_config = find_armpl_config()
    if armpl_config is None:
        note = "ArmPL not installed; run scripts/install_armpl.sh -y or set ARMPL_DIR"
        return skipped_rows("armpl", sizes, batch, warmup, iters, note)

    source = REPO_ROOT / "test" / "perf" / "thirdparty_gemm_bench.cpp"
    binary = BENCH_BIN_DIR / "armpl_gemm_bench"
    compile_cpp(
        binary,
        source,
        [
            "-DBENCH_BACKEND_ARMPL=1",
            *armpl_link_args(armpl_config),
        ],
    )
    run_binary(
        binary,
        output,
        sizes=sizes,
        batch=batch,
        warmup=warmup,
        iters=iters,
        extra_args=["--layout", layout, "--dtype", dtype] + (["--verify-results"] if verify_results else []),
    )
    return read_rows(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run third-party GEMM performance benchmarks.")
    parser.add_argument("--sizes", default=DEFAULT_SIZES)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--layout", default="tn", choices=["nn", "nt", "tn", "tt"])
    parser.add_argument("--dtype", default="fp64", choices=["fp64", "fp32"])
    parser.add_argument("--verify-results", action="store_true", help="run a correctness check after timing")
    parser.add_argument(
        "--backends",
        default="openblas,libxsmm,armpl",
        help="comma-separated backend list",
    )
    parser.add_argument(
        "--output",
        default=str(BENCH_OUT_DIR / "thirdparty_gemm_perf.csv"),
        help="merged CSV output path",
    )
    args = parser.parse_args()

    ensure_cmake_target("thirdparty_build")

    requested = [item.strip() for item in args.backends.split(",") if item.strip()]
    merged_rows: list[dict[str, str]] = []

    if "openblas" in requested:
        merged_rows.extend(
            build_and_run_openblas(
                BENCH_OUT_DIR / "openblas_gemm_perf.csv",
                args.sizes,
                args.batch,
                args.warmup,
                args.iters,
                args.verify_results,
                args.layout,
                args.dtype,
            )
        )
    if "libxsmm" in requested:
        merged_rows.extend(
            build_and_run_libxsmm(
                BENCH_OUT_DIR / "libxsmm_gemm_perf.csv",
                args.sizes,
                args.batch,
                args.warmup,
                args.iters,
                args.verify_results,
                args.layout,
                args.dtype,
            )
        )
    if "armpl" in requested:
        merged_rows.extend(
            build_and_run_armpl(
                BENCH_OUT_DIR / "armpl_gemm_perf.csv",
                args.sizes,
                args.batch,
                args.warmup,
                args.iters,
                args.verify_results,
                args.layout,
                args.dtype,
            )
        )

    output = Path(args.output)
    write_rows(output, merged_rows)
    print_summary(merged_rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
