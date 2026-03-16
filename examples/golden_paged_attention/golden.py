#!/usr/bin/env python3
"""
golden.py — Golden reference test framework for pypto-serving.

Injects requests via TestPath, retrieves engine output, and compares
against golden reference values.

Phase 0–4 (stub mode): validates pipeline plumbing with deterministic
stub outputs.  Phase 4+ (real kernel): validates against computed golden
reference values with RTOL/ATOL tolerance.

Usage:
    python examples/golden_paged_attention/golden.py [--case NAME] [--all] [--trace]
    python examples/golden_paged_attention/golden.py --list
"""

import argparse
import json
import os
import sys

# Add test_path binding to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "frontend"))
from test_path import TestPathClient


# =========================================================================
# Golden test case definitions
# =========================================================================

class GoldenCase:
    """A single golden test case."""
    def __init__(self, name, token_ids, max_tokens=4, vocab_size=32,
                 kv_size=64, kv_step=64, temperature=1.0, top_p=1.0,
                 stop_token=-1,
                 expected_n_tokens=None,
                 expected_tokens=None,
                 description=""):
        self.name = name
        self.token_ids = token_ids
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
        self.kv_size = kv_size
        self.kv_step = kv_step
        self.temperature = temperature
        self.top_p = top_p
        self.stop_token = stop_token
        self.expected_n_tokens = expected_n_tokens
        self.expected_tokens = expected_tokens
        self.description = description


# Default test cases for stub mode
DEFAULT_CASES = [
    GoldenCase(
        name="basic_8tok",
        token_ids=[101, 2023, 2003, 1037, 3231, 6251, 2005, 5604],
        max_tokens=4,
        expected_n_tokens=5,  # 1 prefill + 4 decode
        expected_tokens=[15, 31, 15, 31, 15],
        description="Basic 8-token prompt, 4 decode steps",
    ),
    GoldenCase(
        name="short_2tok",
        token_ids=[42, 99],
        max_tokens=2,
        expected_n_tokens=3,
        expected_tokens=[15, 31, 15],
        description="Short 2-token prompt, 2 decode steps",
    ),
    GoldenCase(
        name="single_token",
        token_ids=[7],
        max_tokens=1,
        expected_n_tokens=2,
        expected_tokens=[15, 31],
        description="Single token prompt, 1 decode step",
    ),
    GoldenCase(
        name="determinism",
        token_ids=[101, 2023, 2003, 1037, 3231, 6251, 2005, 5604],
        max_tokens=4,
        expected_n_tokens=5,
        expected_tokens=[15, 31, 15, 31, 15],
        description="Determinism check (same as basic_8tok)",
    ),
    GoldenCase(
        name="longer_decode",
        token_ids=[1, 2, 3],
        max_tokens=8,
        description="Longer decode run (8 steps)",
    ),
]

ALL_CASES = DEFAULT_CASES


# =========================================================================
# Golden runner
# =========================================================================

class GoldenRunner:
    """Runs golden test cases against the engine via TestPath."""

    def __init__(self, lib_path=None, trace_path=None):
        self.client = TestPathClient(lib_path)
        self.trace_path = trace_path
        self.results = []

    def start(self):
        self.client.start(trace_path=self.trace_path)

    def stop(self):
        if self.trace_path:
            self.client.write_trace()
        self.client.stop()

    def run_case(self, case):
        """Run a single golden test case. Returns (passed, details)."""
        resp = self.client.infer(
            token_ids=case.token_ids,
            max_tokens=case.max_tokens,
            vocab_size=case.vocab_size,
            kv_size_per_chip=case.kv_size,
            kv_step_size=case.kv_step,
            temperature=case.temperature,
            top_p=case.top_p,
            stop_token=case.stop_token,
        )

        output = resp.output_tokens
        passed = True
        details = []

        # Check token count
        if case.expected_n_tokens is not None:
            if len(output) != case.expected_n_tokens:
                passed = False
                details.append(
                    f"token count: expected {case.expected_n_tokens}, "
                    f"got {len(output)}")

        # Check exact token values
        if case.expected_tokens is not None:
            if output != case.expected_tokens:
                passed = False
                details.append(
                    f"tokens: expected {case.expected_tokens}, got {output}")

        # Basic sanity: at least 1 output
        if len(output) == 0:
            passed = False
            details.append("no output tokens")

        return passed, output, details

    def run_cases(self, cases, verbose=True):
        """Run multiple cases. Returns (n_passed, n_total)."""
        n_passed = 0
        n_total = len(cases)

        for case in cases:
            passed, output, details = self.run_case(case)
            status = "PASS" if passed else "FAIL"

            if verbose:
                print(f"  [{status}] {case.name}: {len(output)} tokens = {output}")
                if case.description:
                    print(f"         {case.description}")
                for d in details:
                    print(f"         ERROR: {d}")

            if passed:
                n_passed += 1

            self.results.append({
                "name": case.name,
                "passed": passed,
                "output": output,
                "details": details,
            })

        return n_passed, n_total

    def write_report(self, path="golden_report.json"):
        """Write JSON report of all results."""
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        return path


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Golden reference tests")
    parser.add_argument("--case", type=str, default=None,
                        help="Run a specific case by name")
    parser.add_argument("--all", action="store_true",
                        help="Run all cases (default: DEFAULT_CASES)")
    parser.add_argument("--list", action="store_true",
                        help="List available cases")
    parser.add_argument("--trace", action="store_true",
                        help="Generate Perfetto trace")
    parser.add_argument("--lib", type=str, default=None,
                        help="Path to pypto_testpath.so")
    parser.add_argument("--report", type=str, default="golden_report.json",
                        help="Path for JSON report")
    args = parser.parse_args()

    if args.list:
        print("Available golden test cases:")
        for c in ALL_CASES:
            print(f"  {c.name:20s} — {c.description}")
        return

    # Select cases
    if args.case:
        cases = [c for c in ALL_CASES if c.name == args.case]
        if not cases:
            print(f"ERROR: case '{args.case}' not found")
            sys.exit(1)
    elif args.all:
        cases = ALL_CASES
    else:
        cases = DEFAULT_CASES

    trace_path = "golden_trace.json" if args.trace else None

    print("=== Golden Reference Tests ===\n")
    runner = GoldenRunner(lib_path=args.lib, trace_path=trace_path)
    runner.start()

    n_passed, n_total = runner.run_cases(cases)

    runner.stop()

    report = runner.write_report(args.report)
    print(f"\nReport: {report}")
    print(f"\nResult: {n_passed}/{n_total} passed")

    if n_passed == n_total:
        print("\n=== ALL GOLDEN TESTS PASSED ===")
    else:
        print(f"\n=== {n_total - n_passed} GOLDEN TESTS FAILED ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
