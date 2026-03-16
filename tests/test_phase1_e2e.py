#!/usr/bin/env python3
"""
test_phase1_e2e.py — Phase 1 end-to-end test: Python → TestPath → Engine → Python

Validates the full acceptance criteria:
  Python → TestPath → L3 LevelRuntime → L2 stub → TestPath → Python

Usage:
    python tests/test_phase1_e2e.py [--trace]
    # Run from the pypto-serving directory (build/ must contain pypto_testpath.so)
"""

import argparse
import os
import sys

# Add src/frontend to path so we can import test_path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "frontend"))
from test_path import TestPathClient, InferRequest


def main():
    parser = argparse.ArgumentParser(description="Phase 1 E2E test")
    parser.add_argument("--trace", action="store_true",
                        help="Generate Perfetto trace file")
    parser.add_argument("--lib", type=str, default=None,
                        help="Path to pypto_testpath.so")
    args = parser.parse_args()

    # Locate the shared library
    lib_path = args.lib
    if lib_path is None:
        build_dir = os.path.join(os.path.dirname(__file__), "..", "build")
        lib_path = os.path.join(build_dir, "pypto_testpath.so")
        if not os.path.exists(lib_path):
            print(f"ERROR: Cannot find {lib_path}. Build the project first.")
            sys.exit(1)

    print("=== Phase 1: Python E2E Test via TestPath ===\n")

    trace_path = "phase1_e2e_trace.json" if args.trace else None
    client = TestPathClient(lib_path)
    client.start(trace_path=trace_path)
    print(f"[OK] Engine started (trace={'on' if trace_path else 'off'})")

    # ── Test 1: Basic inference ────────────────────────────────────────
    print("\n--- Test 1: Basic inference (8 tokens, max_tokens=4) ---")
    resp = client.infer(
        token_ids=[101, 2023, 2003, 1037, 3231, 6251, 2005, 5604],
        max_tokens=4,
        vocab_size=32,
    )
    print(f"  Output tokens ({len(resp.output_tokens)}): {resp.output_tokens}")
    assert len(resp.output_tokens) > 0, "Expected at least 1 output token"
    assert len(resp.output_tokens) <= 5, f"Expected <= 5 tokens, got {len(resp.output_tokens)}"
    print("  [PASS]")

    # ── Test 2: Short prompt ───────────────────────────────────────────
    print("\n--- Test 2: Short prompt (2 tokens, max_tokens=2) ---")
    resp2 = client.infer(
        token_ids=[42, 99],
        max_tokens=2,
        vocab_size=32,
    )
    print(f"  Output tokens ({len(resp2.output_tokens)}): {resp2.output_tokens}")
    assert len(resp2.output_tokens) > 0
    assert len(resp2.output_tokens) <= 3
    print("  [PASS]")

    # ── Test 3: Single token prompt ────────────────────────────────────
    print("\n--- Test 3: Single token prompt (max_tokens=1) ---")
    resp3 = client.infer(
        token_ids=[7],
        max_tokens=1,
        vocab_size=32,
    )
    print(f"  Output tokens ({len(resp3.output_tokens)}): {resp3.output_tokens}")
    assert len(resp3.output_tokens) > 0
    assert len(resp3.output_tokens) <= 2
    print("  [PASS]")

    # ── Test 4: Determinism ────────────────────────────────────────────
    print("\n--- Test 4: Determinism (same input → same output) ---")
    resp4a = client.infer(
        token_ids=[101, 2023, 2003, 1037, 3231, 6251, 2005, 5604],
        max_tokens=4,
        vocab_size=32,
    )
    resp4b = client.infer(
        token_ids=[101, 2023, 2003, 1037, 3231, 6251, 2005, 5604],
        max_tokens=4,
        vocab_size=32,
    )
    print(f"  Run A: {resp4a.output_tokens}")
    print(f"  Run B: {resp4b.output_tokens}")
    assert resp4a.output_tokens == resp4b.output_tokens, \
        f"Determinism failed: {resp4a.output_tokens} != {resp4b.output_tokens}"
    print("  [PASS]")

    # ── Trace ──────────────────────────────────────────────────────────
    if args.trace:
        print("\n--- Writing trace ---")
        ok = client.write_trace()
        print(f"  Trace written: {ok}")

    # ── Cleanup ────────────────────────────────────────────────────────
    client.stop()
    print("\n[OK] Engine stopped")

    print("\n=== Phase 1: Python E2E Test PASSED ===")
    print("Acceptance criteria met:")
    print("  Python → TestPath → L3 LevelRuntime → L2 stub → TestPath → Python ✓")


if __name__ == "__main__":
    main()
