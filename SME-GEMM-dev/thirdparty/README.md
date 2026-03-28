# Third-Party Layout

`thirdparty/` keeps checked-in vendor sources separate from repo-local generated artifacts.

## Checked-in source trees

- `OpenBLAS/`: OpenBLAS submodule source.
- `libxsmm/`: LIBXSMM submodule source.

These directories are the only third-party trees we expect to review or version.

## Repo-local generated directories

- `_build/`: build outputs produced by `scripts/build_thirdparty.sh`.
  - `_build/libxsmm/`: LIBXSMM benchmark artifacts.
    - `cmake-build/`: internal CMake build tree.
    - `include/`: headers copied for the benchmark compile path.
    - `lib/`: built library artifacts such as `libxsmm.a`.
- `_install/`: local install prefixes created by project scripts.
  - `_install/openblas/`: optional OpenBLAS install prefix.
  - `_install/armpl/`: repo-local ArmPL installation created by `scripts/install_armpl.sh`.
- `_downloads/`: cached third-party archives downloaded by our helper scripts.
  - `_downloads/armpl/`: cached ArmPL archives.
- `_staging/`: extracted packages before installation.
  - `_staging/armpl/`: staged ArmPL payloads such as the macOS DMG contents.

Everything under the underscored directories is machine-local and ignored by Git.

## Typical workflow

- Run `bash scripts/build_thirdparty.sh build` to build LIBXSMM and OpenBLAS and to stage ArmPL.
- Run `bash scripts/install_armpl.sh -y` to install ArmPL into `_install/armpl/`.
- Run `python3 scripts/run_gemm_perf_compare.py` to benchmark the local kernel and the third-party backends.

## Notes

- We intentionally keep vendor source directories in place to avoid changing submodule paths.
- LIBXSMM is built through its upstream `CMakeLists.txt`, but we still normalize the output into `_build/libxsmm/include` and `_build/libxsmm/lib` so the benchmark scripts can keep a stable path convention.
- If a third-party package leaves extra files in its source tree, treat those as upstream build artifacts rather than project-owned files.
