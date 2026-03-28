#!/usr/bin/env python3
"""
Automated test runner for SMELT frontend with different layouts, shapes, and strategies.
Easily add or remove test cases and get detailed failure reports.
"""

import subprocess
import sys
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import os
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# Test Configuration
# ============================================================================

class Layout(Enum):
    """Matrix layout types: (trans_A, trans_B)"""
    NN = "nn"  # C = A * B
    NT = "nt"  # C = A * B^T
    TN = "tn"  # C = A^T * B
    TT = "tt"  # C = A^T * B^T

class DataType(Enum):
    """Supported data types"""
    FP64 = "fp64"
    FP32 = "fp32"

class Strategy(Enum):
    """Compilation strategies"""
    COSTMODEL = "costmodel"
    SCALAR = "scalar"
    SVE = "sve"
    FUSE_SVE = "fuse_sve"
    SME2 = "sme2"
    FUSE_SME2 = "fuse_sme2"
    MOPA = "mopa"
    FUSE_MOPA = "fuse_mopa"
    STRATEGY1 = "strategy1"

class Backend(Enum):
    """Backend lowering path"""
    IR2C = "ir2c"
    IR2A_ASM = "ir2a-asm"
    IR2A_BIN = "ir2a-bin"

@dataclass
class MatrixShape:
    """Matrix dimensions"""
    M: int
    N: int
    K: int


@dataclass
class FuseExpectation:
    """Expected fuse behavior in generated kernel.cpp."""
    expected_batch_step: Optional[int] = None
    min_batch_step: Optional[int] = None
    required_patterns: Optional[List[str]] = None
    forbidden_patterns: Optional[List[str]] = None
    note: str = ""

@dataclass
class TestCase:
    """A single test case configuration"""
    shape: MatrixShape
    layout: Layout
    dtype: DataType
    strategy: Strategy
    batch: int = 8  # 8 means use default
    fuse_expectation: Optional[FuseExpectation] = None
    expect_success: bool = True
    expected_error_patterns: Optional[List[str]] = None
    backend: Backend = Backend.IR2C
    
    def to_args(self) -> List[str]:
        """Convert test case to command-line arguments"""
        args = [str(self.shape.M), str(self.shape.N), str(self.shape.K)]
        if self.batch != -1:
            args.append(str(self.batch))
        args.extend([
            "-l", self.layout.value,
            "-t", self.dtype.value,
            "-s", self.strategy.value,
            "-b", self.backend.value,
        ])
        return args
    
    def __str__(self) -> str:
        batch_str = f"batch={self.batch}" if self.batch != -1 else ""
        expect_str = "" if self.expect_success else " expect_fail"
        return (f"Shape({self.shape.M}x{self.shape.N}x{self.shape.K}) "
                f"Layout={self.layout.value} Type={self.dtype.value} "
                f"Strategy={self.strategy.value} Backend={self.backend.value} {batch_str}{expect_str}").rstrip()

@dataclass
class TestResult:
    """Result of a single test case"""
    test_case: TestCase
    passed: bool
    duration_ms: float
    error_output: str = ""
    fuse_check_output: str = ""
    
    def summary(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status} | {self.test_case} | {self.duration_ms:.2f}ms"

# ============================================================================
# TEST SUITE CONFIGURATION - EDIT HERE TO ADD/REMOVE TESTS
# ============================================================================

def get_test_cases() -> List[TestCase]:
    """
    Configure all test cases here.
    Easily add new test cases by appending to the list.
    """
    test_cases = []
    
    # Example test cases - modify or add more as needed
    
    # Small matrices with different layouts
    for layout in [Layout.NN, Layout.NT, Layout.TN, Layout.TT]:
        for dtype in [DataType.FP64, DataType.FP32]:
            for strategy in [Strategy.FUSE_SVE, Strategy.SME2, Strategy.FUSE_SME2, Strategy.COSTMODEL, Strategy.MOPA, Strategy.FUSE_MOPA]:
                # for shape in [MatrixShape(11, 12, 10), MatrixShape(14, 13, 16)]:
                for shape in [MatrixShape(12, 12, 10), MatrixShape(14, 14, 16), MatrixShape(10, 10, 10), MatrixShape(17, 9, 16)]: 
                    for backend in [Backend.IR2C, Backend.IR2A_ASM, Backend.IR2A_BIN]:
                        fuse_expect = None
                        # Example: assert fused execution for known shape/layout/strategy combinations.
                        # if strategy == Strategy.FUSE_SVE and dtype == DataType.FP64:
                        #     if layout in [Layout.NN, Layout.TT] and shape == MatrixShape(10, 10, 10):
                        #         fuse_expect = FuseExpectation(
                        #             min_batch_step=2,
                        #             required_patterns=[r"A\[b \+ 1\]"],
                        #             note="Expect batch fusion for 10x10 FP64 in fuse_sve",
                        #         )
                        test_cases.append(TestCase(
                            shape=shape,
                            layout=layout,
                            dtype=dtype,
                            strategy=strategy,
                            fuse_expectation=fuse_expect,
                            backend=backend,
                        ))

    # Explicit fuse behavior checks (easy to extend)
    # test_cases.append(TestCase(
    #     shape=MatrixShape(10, 10, 10),
    #     layout=Layout.NN,
    #     dtype=DataType.FP64,
    #     strategy=Strategy.FUSE_SVE,
    #     batch=8,
    #     fuse_expectation=FuseExpectation(
    #         expected_batch_step=4,
    #         required_patterns=[r"A\[b \+ 1\]", r"A\[b \+ 2\]", r"A\[b \+ 3\]"],
    #         note="NN 10x10 should fuse by 4 batches",
    #     ),
    # ))
    # test_cases.append(TestCase(
    #     shape=MatrixShape(17, 9, 16),
    #     layout=Layout.TN,
    #     dtype=DataType.FP64,
    #     strategy=Strategy.FUSE_SVE,
    #     batch=8,
    #     fuse_expectation=FuseExpectation(
    #         expected_batch_step=8,
    #         required_patterns=[r"A\[b \+ 1\]"],
    #         note="TN 17x9 should fuse by 8 batches",
    #     ),
    # ))

    
    return test_cases


# ============================================================================
# Test Execution
# ============================================================================

class TestRunner:
    def __init__(self, binary_path: str, work_dir: Optional[str] = None, parallel_mode: bool = False):
        # Convert to absolute path for the binary
        self.binary_path = os.path.abspath(binary_path)
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.work_dir = work_dir or self.repo_root
        self.parallel_mode = parallel_mode
        self.results: List[TestResult] = []

    def build_frontend_test(self):
        """Rebuild frontend_test so test cases always run against current sources."""
        build_dir = os.path.join(self.repo_root, "out", "build", "default")
        cmd = ["cmake", "--build", build_dir, "--target", "frontend_test", "-j"]
        print(f"Rebuilding frontend_test: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=self.repo_root, check=True, timeout=300)

    def _artifact_tag(self, test_case: TestCase) -> str:
        def sanitize(value: str) -> str:
            return "".join(ch if ch.isalnum() else "_" for ch in value)

        effective_batch = test_case.batch if test_case.batch != -1 else 8

        return (
            f"{sanitize(test_case.strategy.value)}__{sanitize(test_case.backend.value)}__"
            f"{sanitize(test_case.layout.value)}__{sanitize(test_case.dtype.value)}__"
            f"M{test_case.shape.M}_N{test_case.shape.N}_K{test_case.shape.K}_B{effective_batch}"
        )

    def _load_generated_kernel(self, test_case: TestCase) -> Optional[str]:
        artifact_tag = self._artifact_tag(test_case)
        ext = ".kernel.cpp" if test_case.backend == Backend.IR2C else ".kernel.s"
        kernel_path = os.path.join(self.repo_root, "test", "verify", "artifacts", artifact_tag + ext)
        if not os.path.exists(kernel_path):
            return None
        with open(kernel_path, "r", encoding="utf-8") as f:
            return f.read()

    def _check_fuse_expectation(self, test_case: TestCase) -> str:
        expect = test_case.fuse_expectation
        if expect is None:
            return ""

        kernel_code = self._load_generated_kernel(test_case)
        if kernel_code is None:
            return "FuseCheck: artifact kernel file not found"

        msgs = []
        m = re.search(r"for\s*\(int64_t\s+b\s*=\s*0;\s*b\s*<\s*batch;\s*b\s*\+=\s*(\d+)\)", kernel_code)
        if m is None:
            msgs.append("FuseCheck: cannot find outer batch loop")
        else:
            step = int(m.group(1))
            if expect.expected_batch_step is not None and step != expect.expected_batch_step:
                msgs.append(f"FuseCheck: expected batch step {expect.expected_batch_step}, got {step}")
            if expect.min_batch_step is not None and step < expect.min_batch_step:
                msgs.append(f"FuseCheck: expected batch step >= {expect.min_batch_step}, got {step}")

        for pattern in (expect.required_patterns or []):
            if re.search(pattern, kernel_code) is None:
                msgs.append(f"FuseCheck: missing required pattern: {pattern}")

        for pattern in (expect.forbidden_patterns or []):
            if re.search(pattern, kernel_code) is not None:
                msgs.append(f"FuseCheck: found forbidden pattern: {pattern}")

        if msgs:
            if expect.note:
                msgs.append(f"FuseCheck note: {expect.note}")
            return "\n".join(msgs)
        return ""
    
    def run_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case"""
        args = [self.binary_path] + test_case.to_args()
        if self.parallel_mode:
            args.append("-p")
        
        try:
            env = os.environ.copy()
            env['TIMES_OVERRIDE'] = '1'  
            st = datetime.now()
            result = subprocess.run(
                args,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                env=env,
                timeout=30  # 30 seconds timeout for each test case
            )
            et = datetime.now()
            duration_ms = (et - st).total_seconds() * 1000
            
            combined_output = "\n".join([x for x in [result.stderr, result.stdout] if x])
            error_output = ""
            fuse_check_output = ""

            if test_case.expect_success:
                passed = result.returncode == 0
                error_output = combined_output if not passed else ""
                fuse_check_output = self._check_fuse_expectation(test_case)
                if fuse_check_output:
                    passed = False
                    if error_output:
                        error_output += "\n\n"
                    error_output += fuse_check_output
            else:
                passed = result.returncode != 0
                if not passed:
                    error_output = "Expected failure but command succeeded"
                else:
                    missing_patterns = []
                    for pattern in (test_case.expected_error_patterns or []):
                        if re.search(pattern, combined_output) is None:
                            missing_patterns.append(pattern)
                    if missing_patterns:
                        passed = False
                        error_output = "Expected error patterns not found:\n" + "\n".join(missing_patterns)
            
            test_result = TestResult(
                test_case=test_case,
                passed=passed,
                duration_ms=duration_ms,
                error_output=error_output,
                fuse_check_output=fuse_check_output,
            )
            self.results.append(test_result)
            return test_result
            
        except subprocess.TimeoutExpired:
            test_result = TestResult(
                test_case=test_case,
                passed=False,
                duration_ms=30000,
                error_output="Test timed out after 30 seconds",
                fuse_check_output="",
            )
            self.results.append(test_result)
            return test_result
        except Exception as e:
            test_result = TestResult(
                test_case=test_case,
                passed=False,
                duration_ms=0,
                error_output=str(e),
                fuse_check_output="",
            )
            self.results.append(test_result)
            return test_result
    
    def run_all(self, test_cases: List[TestCase], verbose: bool = False, jobs: int = 1):
        """Run all test cases"""
        total = len(test_cases)
        print(f"\n{'='*80}")
        print(f"Running {total} test cases...")
        print(f"{'='*80}\n")

        if jobs <= 1:
            for idx, test_case in enumerate(test_cases, 1):
                print(f"[{idx:3d}/{total}] Running: {test_case}")
                result = self.run_test(test_case)

                if result.passed:
                    print(f"       ✓ PASS ({result.duration_ms:.2f}ms)")
                else:
                    print(f"       ✗ FAIL ({result.duration_ms:.2f}ms)")
                    if verbose:
                        print(f"       Error: {result.error_output[:200]}")
                    if result.fuse_check_output:
                        print("       Fuse expectation check failed")
                print()
            return

        print(f"Parallel mode enabled, jobs={jobs}")
        completed = 0
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            future_to_case = {executor.submit(self.run_test, tc): tc for tc in test_cases}
            for future in as_completed(future_to_case):
                tc = future_to_case[future]
                completed += 1
                try:
                    result = future.result()
                except Exception as e:
                    result = TestResult(test_case=tc, passed=False, duration_ms=0, error_output=str(e))
                    self.results.append(result)

                print(f"[{completed:3d}/{total}] Finished: {tc}")
                if result.passed:
                    print(f"       ✓ PASS ({result.duration_ms:.2f}ms)")
                else:
                    print(f"       ✗ FAIL ({result.duration_ms:.2f}ms)")
                    if verbose:
                        print(f"       Error: {result.error_output[:200]}")
                    if result.fuse_check_output:
                        print("       Fuse expectation check failed")
                print()
    
    def print_summary(self):
        """Print test summary"""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        
        print(f"\n{'='*80}")
        print(f"TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total:  {total}")
        print(f"Passed: {passed} ({100*passed/total:.1f}%)" if total > 0 else "Passed: 0")
        print(f"Failed: {failed} ({100*failed/total:.1f}%)" if total > 0 else "Failed: 0")
        print(f"{'='*80}\n")
        
        # Print failed tests with details
        if failed > 0:
            print(f"\nFAILED TESTS ({failed}):\n")
            for result in self.results:
                if not result.passed:
                    print(f"✗ {result.test_case}")
                    if result.error_output:
                        error_lines = [line for line in result.error_output.split('\n') if line.strip()]
                        preview_lines = error_lines[-12:] if len(error_lines) > 12 else error_lines
                        for line in preview_lines:
                            if line.strip():
                                print(f"  > {line}")
                        if len(error_lines) > len(preview_lines):
                            print(f"  ... ({len(error_lines)-len(preview_lines)} earlier lines omitted)")
                    print()
    
    def save_report(self, filename: str = "test_report.json"):
        """Save detailed test report to JSON"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "tests": []
        }
        
        for result in self.results:
            report["tests"].append({
                "test": str(result.test_case),
                "passed": result.passed,
                "duration_ms": result.duration_ms,
                "error": result.error_output[:500] if result.error_output else None,
                "fuse_check": result.fuse_check_output if result.fuse_check_output else None,
            })
        
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {filename}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run frontend_test test matrix")
    parser.add_argument("-v", "--verbose", action="store_true", help="show failure snippets")
    parser.add_argument("-r", "--report", action="store_true", help="save JSON report")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="number of parallel jobs")
    args = parser.parse_args()

    # Find binary
    binary_options = [
        os.path.join(os.path.dirname(__file__), "..", "out", "build", "default", "test", "frontend_test"),
    ]
    
    binary_path = None
    for opt in binary_options:
        if os.path.exists(opt):
            binary_path = opt
            break
    
    if binary_path is None:
        print("Error: Could not find frontend_test binary")
        print(f"Searched: {binary_options}")
        print("\nPlease build the project first using:")
        print("  cmake -S . -B out/build/default --preset=default")
        print("  cmake --build out/build/default")
        print("Or alternatively:")
        print("  cmake -S . -B gcc_build")
        print("  cmake --build gcc_build")
        sys.exit(1)
    
    print(f"Using binary: {binary_path}")

    verbose = args.verbose
    save_report = args.report
    jobs = max(1, args.jobs)
    
    # Run tests
    runner = TestRunner(binary_path, parallel_mode=(jobs > 1))
    try:
        runner.build_frontend_test()
    except Exception as exc:
        print(f"Error: failed to rebuild frontend_test: {exc}")
        return 1
    test_cases = get_test_cases()
    
    runner.run_all(test_cases, verbose=verbose, jobs=jobs)
    runner.print_summary()
    
    if save_report:
        runner.save_report()
    
    # Return appropriate exit code
    failed = sum(1 for r in runner.results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())