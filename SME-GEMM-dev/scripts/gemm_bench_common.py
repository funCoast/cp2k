#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(os.environ.get("SME_GEMM_REPO_ROOT", Path(__file__).resolve().parents[1]))
CONFIGURE_PRESET = os.environ.get("SME_GEMM_CONFIGURE_PRESET", "default")
BUILD_DIR = Path(os.environ.get("SME_GEMM_BUILD_DIR", REPO_ROOT / "out" / "build" / CONFIGURE_PRESET))
BENCH_OUT_DIR = REPO_ROOT / "out" / "bench"
BENCH_BIN_DIR = BENCH_OUT_DIR / "bin"
VERIFY_ARTIFACT_DIR = REPO_ROOT / "test" / "verify" / "artifacts"

DEFAULT_SIZES = ",".join(str(v) for v in range(2, 22, 2))
DEFAULT_BATCH = 8
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 1000
CMAKE_CACHE_RESET_HINT = "require your cache to be deleted"


@lru_cache(maxsize=1)
def preset_environment() -> dict[str, str]:
    preset_file = REPO_ROOT / "CMakeUserPresets.json"
    if not preset_file.is_file():
        return {}

    data = json.loads(preset_file.read_text(encoding="utf-8"))
    for preset in data.get("configurePresets", []):
        if preset.get("name") != CONFIGURE_PRESET:
            continue

        resolved: dict[str, str] = {}
        for key, raw_value in preset.get("environment", {}).items():
            value = str(raw_value)
            for match in re.finditer(r"\$penv\{([^}]+)\}", value):
                name = match.group(1)
                value = value.replace(match.group(0), os.environ.get(name, ""))
            for match in re.finditer(r"\$env\{([^}]+)\}", value):
                name = match.group(1)
                replacement = resolved.get(name, os.environ.get(name, ""))
                value = value.replace(match.group(0), replacement)
            resolved[key] = value
        return resolved

    return {}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run(cmd: List[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd or REPO_ROOT, env=env, check=True)


def captured(cmd: List[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(cmd, cwd=cwd or REPO_ROOT, env=env, text=True).strip()


def cmake_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(preset_environment())
    env.setdefault("VCPKG_FORCE_SYSTEM_BINARIES", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    return env


def cmake_bin() -> str:
    if os.environ.get("SME_GEMM_CMAKE"):
        return os.environ["SME_GEMM_CMAKE"]
    resolved = shutil.which("cmake", path=cmake_env().get("PATH"))
    return resolved or "cmake"


def ensure_cmake_target(target: str) -> None:
    return
    env = cmake_env()
    configure_cmd = [cmake_bin(), "--preset", CONFIGURE_PRESET]
    print("+", " ".join(configure_cmd))
    configure = subprocess.run(
        configure_cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    if configure.stdout:
        print(configure.stdout, end="")
    if configure.stderr:
        print(configure.stderr, end="", file=sys.stderr)

    combined_output = f"{configure.stdout}\n{configure.stderr}"
    if CMAKE_CACHE_RESET_HINT in combined_output:
        fresh_cmd = configure_cmd + ["--fresh"]
        run(fresh_cmd, env=env)
    elif configure.returncode != 0:
        raise subprocess.CalledProcessError(configure.returncode, configure_cmd)

    run([cmake_bin(), "--build", str(BUILD_DIR), "--target", target, "-j"], env=env)


def compiler() -> str:
    cache = BUILD_DIR / "CMakeCache.txt"
    if cache.exists():
        cache_text = cache.read_text(encoding="utf-8", errors="ignore")
        match = re.search(
            r"^CMAKE_CXX_COMPILER:FILEPATH=(.+)$",
            cache_text,
            re.MULTILINE,
        )
        if not match:
            match = re.search(r"^CMAKE_CXX_COMPILER:UNINITIALIZED=(.+)$", cache_text, re.MULTILINE)
        if match:
            candidate = match.group(1)
            if Path(candidate).exists():
                return candidate
            resolved = shutil.which(candidate, path=cmake_env().get("PATH"))
            if resolved:
                return resolved
            tool_match = re.search(r"^CMAKE_CXX_COMPILER_AR:FILEPATH=(.+)$", cache_text, re.MULTILINE)
            if tool_match:
                sibling = Path(tool_match.group(1)).parent / Path(candidate).name
                if sibling.exists():
                    return str(sibling)
    cxx_env = os.environ.get("CXX")
    if cxx_env:
        resolved = shutil.which(cxx_env, path=cmake_env().get("PATH"))
        if resolved:
            return resolved
        if Path(cxx_env).exists():
            return cxx_env
    for candidate in ("clang++", "g++", "c++"):
        resolved = shutil.which(candidate, path=cmake_env().get("PATH"))
        if resolved:
            return resolved
    return "c++"


def detect_gfortran_runtime_dir() -> Path | None:
    path_env = cmake_env().get("PATH")
    gfortran = os.environ.get("FC") or shutil.which("gfortran", path=path_env)
    if not gfortran:
        return None
    for name in ("libgfortran.so", "libgfortran.dylib", "libgfortran.a"):
        path = captured([gfortran, f"-print-file-name={name}"])
        if path and path != name and Path(path).exists():
            return Path(path).parent
    return None


def compile_cpp(
    output: Path,
    source: Path,
    extra_args: Iterable[str],
    *,
    require_sme: bool = False,
    opt_level: str = "-O3",
) -> None:
    ensure_dir(output.parent)
    cxx = compiler()
    common = [cxx, "-std=c++20", opt_level, "-g", str(source), "-o", str(output)]
    extra = list(extra_args)
    if not require_sme:
        run(common + extra, env=cmake_env())
        return

    march_candidates = [
        "-march=armv9-a+sve2+sme+sme2+sme-f64f64",
        "-march=armv9-a+sve2+sme+sme-f64f64",
    ]
    errors: list[str] = []
    for march in march_candidates:
        try:
            run(common + [march] + extra, env=cmake_env())
            return
        except subprocess.CalledProcessError as exc:
            errors.append(f"{march}: exit={exc.returncode}")
    raise RuntimeError("failed to compile SME benchmark source with candidate flags: " + "; ".join(errors))


def run_binary(
    binary: Path,
    output_csv: Path,
    *,
    sizes: str,
    batch: int,
    warmup: int,
    iters: int,
    extra_args: Iterable[str] = (),
    env_overrides: dict[str, str] | None = None,
) -> None:
    ensure_dir(output_csv.parent)
    cmd = [
        str(binary),
        "--sizes",
        sizes,
        "--batch",
        str(batch),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--csv-output",
        str(output_csv),
    ]
    cmd.extend(extra_args)
    env = cmake_env()
    if env_overrides:
        env.update(env_overrides)
    run(cmd, env=env)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "backend",
        "m",
        "n",
        "k",
        "batch",
        "warmup",
        "iters",
        "avg_ns",
        "gflops",
        "checksum",
        "status",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def size_values(size_expr: str) -> list[int]:
    return [int(part) for part in size_expr.split(",") if part.strip()]


def skipped_rows(backend: str, size_expr: str, batch: int, warmup: int, iters: int, note: str) -> list[dict[str, str]]:
    rows = []
    for value in size_values(size_expr):
        rows.append(
            {
                "backend": backend,
                "m": str(value),
                "n": str(value),
                "k": str(value),
                "batch": str(batch),
                "warmup": str(warmup),
                "iters": str(iters),
                "avg_ns": "",
                "gflops": "",
                "checksum": "",
                "status": "skipped",
                "note": note,
            }
        )
    return rows


def print_summary(rows: list[dict[str, str]]) -> None:
    by_size: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["m"], row["n"], row["k"])
        by_size.setdefault(key, []).append(row)

    for key in sorted(by_size, key=lambda item: int(item[0])):
        batch = by_size[key][0].get("batch", "-")
        print(f"m=n=k={key[0]} batch={batch}")
        for row in sorted(by_size[key], key=lambda item: item["backend"]):
            avg = row["avg_ns"] or "-"
            gflops = row["gflops"] or "-"
            status = row["status"]
            note = row["note"]
            tail = f" ({note})" if note else ""
            print(f"  {row['backend']:<8} avg_ns={avg:<12} gflops={gflops:<10} status={status}{tail}")


def dedup_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def armpl_candidate_roots() -> list[Path]:
    roots: list[Path] = []

    env_dir = os.environ.get("ARMPL_DIR")
    if env_dir:
        roots.append(Path(env_dir).expanduser())

    search_bases = [
        REPO_ROOT / "thirdparty" / "_install" / "armpl",
        REPO_ROOT / "thirdparty" / "_staging" / "armpl",
        REPO_ROOT / "thirdparty" / "armpl-install",
        REPO_ROOT / "thirdparty" / "armpl",
    ]
    if os.environ.get("ARMPL_USE_SYSTEM_INSTALL") == "1":
        search_bases.append(Path("/opt/arm"))
    for base in search_bases:
        if not base.exists():
            continue
        if (base / "armpl_env_vars.sh").is_file() or (base / "include").is_dir() or (base / "lib").is_dir():
            roots.append(base)
        roots.extend(sorted(path for path in base.glob("armpl_*") if path.is_dir()))

    return dedup_paths(roots)


def parse_armpl_env_script(script: Path) -> dict[str, str]:
    exports: dict[str, str] = {}
    pattern = re.compile(r"^export\s+([A-Z0-9_]+)=(.*)$")
    for raw_line in script.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        match = pattern.match(line)
        if not match:
            continue
        key, value = match.groups()
        value = value.strip().strip('"').strip("'")
        for other_key, other_value in exports.items():
            value = value.replace(f"${other_key}", other_value)
            value = value.replace("${" + other_key + "}", other_value)
        exports[key] = value.rstrip("/")
    return exports


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def detect_openmp_library(lib_dir: Path) -> Path | None:
    candidates = [
        lib_dir / "libomp.dylib",
        lib_dir / "libomp.so",
        Path("/usr/local/lib/libomp.dylib"),
        Path("/usr/local/lib/libomp.so"),
        Path("/usr/lib64/libomp.so"),
        Path("/usr/lib/aarch64-linux-gnu/libomp.so"),
        Path("/usr/lib/libomp.so"),
    ]
    return first_existing(candidates)


def detect_flang_runtime(lib_dir: Path) -> Path | None:
    candidates = [
        lib_dir / "libflang_rt.runtime.dylib",
        lib_dir / "libflang_rt.runtime.so",
    ]
    return first_existing(candidates)


def find_armpl_config() -> dict[str, Path | str] | None:
    for root in armpl_candidate_roots():
        env_script = root / "armpl_env_vars.sh"
        exports = parse_armpl_env_script(env_script) if env_script.is_file() else {}

        resolved_root = Path(exports.get("ARMPL_DIR", str(root))).expanduser()
        include_dir = Path(exports.get("ARMPL_INCLUDES", str(resolved_root / "include"))).expanduser()
        lib_dir = Path(exports.get("ARMPL_LIBRARIES", str(resolved_root / "lib"))).expanduser()

        # Older repo-local installs may be moved during layout cleanups while the
        # generated env script still points at the legacy path. Fall back to the
        # discovered root so existing installs remain usable.
        if not include_dir.is_dir() or not lib_dir.is_dir():
            resolved_root = root
            include_dir = root / "include"
            lib_dir = root / "lib"

        if not include_dir.is_dir() or not lib_dir.is_dir():
            continue

        openmp_lib = detect_openmp_library(lib_dir)
        flang_runtime = detect_flang_runtime(lib_dir)
        return {
            "root": resolved_root,
            "include_dir": include_dir,
            "lib_dir": lib_dir,
            "env_script": env_script if env_script.is_file() else "",
            "openmp_lib": openmp_lib or "",
            "flang_runtime": flang_runtime or "",
        }
    return None


def find_armpl_root() -> Path | None:
    config = find_armpl_config()
    if config is None:
        return None
    return Path(config["root"])


def armpl_library(root: Path) -> Path | None:
    lib_dir = root / "lib"
    if not lib_dir.is_dir():
        return None

    preferred_names = [
        "libarmpl_lp64.dylib",
        "libarmpl_lp64.so",
        "libarmpl_lp64.a",
        "libarmpl.dylib",
        "libarmpl.so",
        "libarmpl.a",
    ]
    for name in preferred_names:
        candidate = lib_dir / name
        if candidate.exists():
            return candidate
    return None


def armpl_link_args(config: dict[str, Path | str]) -> list[str]:
    root = Path(config["root"])
    include_dir = Path(config["include_dir"])
    lib_dir = Path(config["lib_dir"])
    library = armpl_library(root)
    if library is None:
        raise RuntimeError(f"no ArmPL library found under {lib_dir}")

    args = [
        "-I",
        str(include_dir),
        str(library),
    ]

    if library.suffix in (".dylib", ".so"):
        args.append(f"-Wl,-rpath,{lib_dir}")

    flang_runtime_value = config["flang_runtime"]
    if flang_runtime_value:
        flang_runtime = Path(flang_runtime_value)
        args.append(str(flang_runtime))
        if flang_runtime.suffix in (".dylib", ".so"):
            args.append(f"-Wl,-rpath,{flang_runtime.parent}")

    args.extend(["-lm", "-lpthread"])
    return args


def script_path(name: str) -> str:
    return str(REPO_ROOT / "scripts" / name)


def python_exe() -> str:
    return os.environ.get("SME_GEMM_PYTHON") or sys.executable


def python_with_module(module: str) -> str:
    candidates = [
        os.environ.get("PLOT_PYTHON"),
        os.environ.get("SME_GEMM_PYTHON"),
        sys.executable,
        shutil.which("python3"),
        "/usr/bin/python3",
        shutil.which("python"),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        path = Path(candidate)
        if not path.exists():
            continue
        probe = subprocess.run(
            [candidate, "-c", f"import {module}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            return candidate
    raise RuntimeError(f"no Python interpreter with module '{module}' found")
