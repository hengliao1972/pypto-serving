"""
serving_main.py — Main entry point (PyPTO DSL spec)

Sets up the full PC16 x2 serving system:
    L4  PodOrchestrator   (sched=1, workers=2)
    L3[0] Prefill PC16    (sched=1, workers=16)
    L3[1] Decode PC16     (sched=1, workers=16)
    L2  ChipBackend stubs (16 per L3)

Wires runtimes together, injects a test request, verifies output.
Supports --trace flag for Perfetto trace generation.

C++ implementation: engine/serving_system.cpp, tests/test_phase0_smoke.cpp
"""

import pl
from pl import Tensor, Level, Role, LevelRuntime
from L3_prefill_server import prefill_orchestrate
from L3_decode_server import decode_orchestrate
from L4_pod_orchestrator import pod_orchestrate

# ---------------------------------------------------------------------------
# Topology constants
# ---------------------------------------------------------------------------

NUM_L2_PER_L3: int = 16
NUM_L3_PER_L4: int = 2


def main():
    # ── Create runtimes ────────────────────────────────────────────────
    rt_l3_prefill = LevelRuntime(level=3, num_scheduler_threads=1,
                                 num_worker_threads=NUM_L2_PER_L3)
    rt_l3_decode = LevelRuntime(level=3, num_scheduler_threads=1,
                                num_worker_threads=NUM_L2_PER_L3)
    rt_l4 = LevelRuntime(level=4, num_scheduler_threads=1,
                         num_worker_threads=NUM_L3_PER_L4)

    # ── Trace setup ────────────────────────────────────────────────────
    trace_writer = pl.TraceWriter()
    if pl.args.trace:
        trace_writer.set_enabled(True)
        for rt in [rt_l3_prefill, rt_l3_decode, rt_l4]:
            rt.set_trace_writer(trace_writer)

        rt_l3_prefill.register_trace_instance(30000, "L3[0] Prefill PC16")
        rt_l3_decode.register_trace_instance(30001, "L3[1] Decode PC16")
        rt_l4.register_trace_instance(40000, "L4 Pod")

    # ── Create chip backends (stubs) ───────────────────────────────────
    chip_backend_prefill = pl.ChipBackendStub(num_chips=NUM_L2_PER_L3)
    chip_backend_decode = pl.ChipBackendStub(num_chips=NUM_L2_PER_L3)

    # ── Start runtimes ─────────────────────────────────────────────────
    for rt in [rt_l3_prefill, rt_l3_decode, rt_l4]:
        rt.start()

    # ── Inject test request ────────────────────────────────────────────
    request = Request(
        token_ids=[101, 2023, 2003, 1037, 3231, 6251, 2005, 5604],
        max_tokens=4,
        temperature=1.0,
        top_p=1.0,
        stop_token=-1,
        vocab_size=32,
        kv_size_per_chip=64,
        kv_step_size=64,
    )

    # ── Execute via L4 orchestrator ────────────────────────────────────
    response_future = rt_l4.submit_orchestrator(
        name="pod_orchestrate",
        fn=lambda: pod_orchestrate(
            rt_l3_prefill, rt_l3_decode, rt_l4,
            chip_backend_prefill, chip_backend_decode, request),
    )
    response = response_future.get()

    # ── Verify ─────────────────────────────────────────────────────────
    print(f"Response: {len(response.output_tokens)} tokens generated")
    assert len(response.output_tokens) > 0
    assert len(response.output_tokens) <= request.max_tokens + 1  # +1 for first token

    # ── Stop runtimes ──────────────────────────────────────────────────
    for rt in [rt_l4, rt_l3_decode, rt_l3_prefill]:
        rt.stop()

    # ── Write trace ────────────────────────────────────────────────────
    if pl.args.trace:
        path = trace_writer.write_json(
            pl.args.trace_path or "pypto_serving_trace.json")
        print(f"[TRACE] Written: {path}")

    print("=== pypto-serving Phase 0 smoke test PASSED ===")


if __name__ == "__main__":
    main()
