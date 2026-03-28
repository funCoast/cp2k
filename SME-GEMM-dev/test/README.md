# Test And Benchmark Guide

This directory contains three different kinds of validation: build targets, the verify harness, and the GEMM performance comparison scripts.

## 1. Build the normal test targets

Configure once:

```bash
cmake --preset default
```

Machine-specific paths such as the LLVM toolchain location, `VCPKG_ROOT`, and whether third-party auto-build should be skipped belong in `CMakeUserPresets.json`, not in the tracked repo presets. A typical local setup is:

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "default",
      "inherits": ["llvm"],
      "environment": {
        "VCPKG_ROOT": "/path/to/vcpkg"
      },
      "cacheVariables": {
        "SME_GEMM_BUILD_THIRDPARTY": "OFF"
      }
    }
  ]
}
```

On another Linux machine, keep the same `default` preset name but replace the machine-specific values, for example your Linux `VCPKG_ROOT` and any compiler path entries you want in `PATH`.

Build the common test executables:

```bash
cmake --build out/build/default --target interface_test frontend_test IR2C_test fmopa_gemm -j
```

The resulting binaries are placed under `out/build/default/test/`.

## 2. Run the local interface test

`interface_test.cpp` is the closest reference for directly generating a kernel and executing it from this project. Build it first, then run:

```bash
out/build/default/test/interface_test
```

## 3. Run the verify harness

`test/verify/` is a small standalone harness for checking emitted kernels against generated `shape.h` and kernel sources.

Typical usage:

```bash
cd test/verify
make
./verify
```

The harness writes temporary objects and generated artifacts inside `test/verify/`; those files are intentionally ignored by Git.

## 4. Run the GEMM performance comparison

The recommended user-facing entrypoint is:

```bash
python3 scripts/run_mode_matrix_perf_suite.py
```

Treat this as the only normal benchmark entrypoint. The lower-level scripts remain in `scripts/` for debugging, but the intended workflow is to drive everything from the top-level suite script.

Install repo-local ArmPL once if you want ArmPL results included:

```bash
bash scripts/install_armpl.sh -y
```

The suite compares:

- `smelt`: current runtime interface path
- `ir2a-asm`: direct generated IR2A assembly kernel, bypassing `SMELT::dgemm_batch`
- `ir2a-bin`: direct generated IR2A raw binary kernel, bypassing `SMELT::dgemm_batch`
- `ir2c`: direct generated IR2C kernel, bypassing `SMELT::dgemm_batch`
- `openblas`, `libxsmm`, `armpl`: reference implementations

The suite now runs the real 8-mode matrix:

- dtypes: `fp32`, `fp64`
- transpose modes: `NN`, `NT`, `TN`, `TT`
- fixed batch: `8`
- total modes: `8`

For each mode it writes:

- one merged comparison CSV
- one grouped bar chart as `.png`
- one grouped bar chart as `.svg`
- one row in `out/bench/mode_matrix_suite/suite_manifest.csv`

The per-mode outputs land under:

```bash
out/bench/mode_matrix_suite/fp32_nn/
out/bench/mode_matrix_suite/fp32_nt/
out/bench/mode_matrix_suite/fp32_tn/
out/bench/mode_matrix_suite/fp32_tt/
out/bench/mode_matrix_suite/fp64_nn/
out/bench/mode_matrix_suite/fp64_nt/
out/bench/mode_matrix_suite/fp64_tn/
out/bench/mode_matrix_suite/fp64_tt/
```

Run the full suite:

```bash
python3 scripts/run_mode_matrix_perf_suite.py
```

Run the same suite in the background:

```bash
mkdir -p out/bench/mode_matrix_suite
nohup python3 scripts/run_mode_matrix_perf_suite.py > out/bench/mode_matrix_suite/run.log 2>&1 & echo $!
```

Tail the background log:

```bash
tail -f out/bench/mode_matrix_suite/run.log
```

Important: run only one benchmark suite at a time. The direct-kernel paths share `test/verify/` as a build-and-run workspace, so launching multiple benchmark scripts in parallel can corrupt each other's temporary objects.

Append `--verify-results` if you want a post-run correctness check. The default is verification off so the benchmark loop stays focused on performance.

If you want the strategy selection to line up more closely across the local runtime path and the direct offline-generated paths, pass both strategy flags explicitly:

```bash
python3 scripts/run_mode_matrix_perf_suite.py --smelt-strategy COSTMODEL --direct-strategy costmodel
```

If a tiny direct-kernel case such as `4x4x4` is too noisy on your machine, use the top-level repeat knob so direct-kernel paths keep the best result:

```bash
python3 scripts/run_mode_matrix_perf_suite.py --direct-repeats 10
```

Outputs are written under `out/bench/`.

## Timing scope and consistency

The current scripts do not all time the exact same scope, and that difference matters when you compare `smelt` against `ir2a-*` or `ir2c`.

- `smelt` times repeated calls to `SMELT::dgemm_batch(...)` or `SMELT::sgemm_batch(...)` from `test/perf/smelt_gemm_bench.cpp`.
- That timing includes the runtime interface work in `src/interface.cpp`, such as transpose parsing, current-strategy lookup, JIT-cache lookup, batch pointer validation, and the final call wrapper.
- The warmup call removes one-time compilation cost from the steady-state numbers, but the per-call runtime dispatch overhead is still part of the `smelt` result.
- `ir2a-asm`, `ir2a-bin`, and `ir2c` time the verify harness path in `test/verify/verify.cpp`, where the loop directly calls the generated kernel entry.
- Those direct-kernel timings intentionally bypass `SMELT::dgemm_batch`, so they are useful for isolating kernel-body performance, not for measuring full runtime API overhead.
- `frontend_test` also prints a separate `Time taken: ... ms` line. That number is lowering/codegen/setup time before `make && ./verify`, not the GEMM runtime throughput.
- `verify.cpp` currently runs `TEST_GROUP(...)` twice for each case. The direct benchmark scripts now record the last `gemm_kernel` pass so the CSV matches the final GFLOPS shown in the verify log instead of averaging both passes together.
- Very small direct-kernel cases can still fluctuate a lot between runs because the measured region is only on the order of sub-microseconds per GEMM. Use `--repeats 10` and keep the best result when you want a more stable peak number for those sizes.

There is also a strategy-selection difference unless you align it yourself:

- `run_smelt_gemm_bench.py` defaults to `AUTO` when you run it directly.
- `run_ir2a_gemm_bench.py` and `run_ir2c_gemm_bench.py` default to `costmodel`.
- `run_gemm_perf_compare.py` now defaults to `smelt=COSTMODEL` and `direct=costmodel` so the merged compare is strategy-aligned by default.
- `frontend_test` does not support an `auto` strategy, so the direct-kernel benchmarks cannot exactly mirror the runtime-side `AUTO` behavior.
- For `fp32 + NN`, `frontend_test` and the runtime path may fall back from `costmodel` to a safer IR2A strategy when the original choice runs out of vector registers. That fallback is there to keep the 8-mode suite runnable.

Because of that, a large gap between `smelt` and `ir2a-*` does not automatically mean the generated IR2A kernel itself is slow or fast. It can also reflect runtime dispatch overhead and strategy mismatch.

## Benchmark defaults

The current benchmark defaults are:

- matrix sizes: `m = n = k = 2, 4, 6, ..., 20`
- dtypes: `fp32`, `fp64`
- storage/layout: row-major inputs
- transpose modes: `nn`, `nt`, `tn`, `tt`
- fixed batch: `8`
- warmup: `10` iterations
- measured iterations: `1000`
- local SMELT benchmark context switching: `auto_context_switch = off`
- full-compare script default strategies: `smelt=COSTMODEL`, `direct=costmodel`
- direct-kernel repeat count: `1` by default, optionally raise to `10` for noisy tiny sizes
- result verification: off by default, opt in with `--verify-results`
- compiler used by verify and benchmark helpers: whatever `cmake --preset default` resolved via `CMakeUserPresets.json` or your `PATH`

The top-level `run_mode_matrix_perf_suite.py` adds these mode-matrix defaults on top:

- dtypes: `fp32,fp64`
- layouts: `nn,nt,tn,tt`
- batch: `8`
- output root: `out/bench/mode_matrix_suite`
- direct-kernel repeats: `10`

That SMELT setup matches the current `interface_test.cpp` execution style more closely than the unstable `batch=1` path.
