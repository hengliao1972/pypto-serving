"""
L4_pod_orchestrator.py — L4 Pod Orchestrator (PyPTO DSL spec)

Defines the L4 (Pod) orchestrator function for Prefill/Decode disaggregation.
The Pod contains 2 PC16 servers:
    L3[0] — Prefill server (L3_prefill_server.py)
    L3[1] — Decode server  (L3_decode_server.py)

PyPTO coding rule:
    - ONE orchestrator function: pod_orchestrate
    - No L4-level worker functions (orchestration only at this level;
      actual compute is delegated to L3 workers)

The L4 orchestrator:
1. Submits prefill_orchestrate to L3[0]
2. Waits for PrefillResult
3. Submits decode_orchestrate to L3[1] with the PrefillResult
4. Waits for DecodeResult
5. Returns Response

Grammar reference: machine_hierarchy_and_function_hierarchy.md §5.7
C++ implementation: engine/pod_orchestrator.cpp
"""

import pl
from pl import Tensor, Level, Role, LevelRuntime
from L3_prefill_server import prefill_orchestrate
from L3_decode_server import decode_orchestrate

NUM_L3_PER_L4: int = 2  # 2 PC16 servers per Pod
L3_PREFILL_IDX: int = 0
L3_DECODE_IDX: int = 1


# ===========================================================================
# L4 ORCHESTRATOR FUNCTION
#
# No L4 workers — this level only orchestrates between two L3 servers.
# All compute is performed by L3 worker threads.
# ===========================================================================

@pl.function(level=Level.POD, role=Role.ORCHESTRATOR)
def pod_orchestrate(rt_l3_prefill: LevelRuntime,
                    rt_l3_decode: LevelRuntime,
                    rt_l4: LevelRuntime,
                    chip_backend_prefill,
                    chip_backend_decode,
                    request) -> "Response":
    """L4 orchestrator: route request through Prefill → Decode pipeline.

    Step 1: Submit prefill_orchestrate to L3[0] (Prefill PC16)
            → 16 NPU chips process the prompt in parallel
            → Returns PrefillResult{kv_cache, first_token}

    Step 2: Submit decode_orchestrate to L3[1] (Decode PC16)
            → Autoregressive loop with 16 NPU chips per step
            → Returns DecodeResult{output_tokens}

    Step 3: Package output tokens into Response
    """
    # Phase 1: Prefill on L3[0]
    prefill_future = rt_l3_prefill.submit_orchestrator(
        name="prefill_orchestrate",
        fn=lambda: prefill_orchestrate(
            rt_l3_prefill, chip_backend_prefill, request),
    )
    prefill_result = prefill_future.get()

    # Phase 2: Decode on L3[1]
    decode_future = rt_l3_decode.submit_orchestrator(
        name="decode_orchestrate",
        fn=lambda: decode_orchestrate(
            rt_l3_decode, chip_backend_decode, prefill_result, request),
    )
    decode_result = decode_future.get()

    # Phase 3: Build response
    return Response(output_tokens=decode_result.output_tokens)
