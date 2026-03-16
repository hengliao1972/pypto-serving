"""
L3_prefill_server.py — L3 Prefill Server (PyPTO DSL spec)

Defines the L3 (Host/PC16) orchestrator and worker functions for the
Prefill server instance (L3[0]).  This PC16 server has 16 NPU chips.

PyPTO coding rule:
    - ONE orchestrator function: prefill_orchestrate
    - Multiple worker functions: model_prefill_host (×16, one per NPU chip)

The orchestrator submits 16 parallel model_prefill_host workers, each
targeting a different NPU chip.  Token sharding distributes the prompt
across chips for tensor parallelism.

Grammar reference: machine_hierarchy_and_function_hierarchy.md §5.7
C++ implementation: engine/inference_engine.cpp (prefill_orchestrate)
                    engine/L3_workers.cpp       (model_prefill_host)
"""

import pl
from pl import Tensor, Level, Role, LevelRuntime

NUM_L2_PER_L3: int = 16  # 16 NPU chips per PC16 server


# ===========================================================================
# L3 WORKER FUNCTIONS
#
# Pure compute: call L2 chip backend, never submit further tasks.
# Each worker handles one NPU chip's share of the prefill work.
# ===========================================================================

@pl.function(level=Level.HOST, role=Role.WORKER)
def model_prefill_host(chip_backend,
                       chip_id: int,
                       token_shard: Tensor,
                       kv_blocks: Tensor,
                       kv_out: Tensor,
                       first_logits: Tensor):
    """L3 worker: execute prefill on one NPU chip via ChipBackend.

    Wraps the L2 model_prefill call with host-side data movement:
      h2d_copy → L2 model_prefill → d2h_copy
    Phase 0: ChipBackend stub does everything in host memory.
    """
    chip_backend.model_prefill(chip_id, token_shard, kv_blocks,
                               kv_out, first_logits)


# ===========================================================================
# L3 ORCHESTRATOR FUNCTION
#
# Builds the task DAG: shards tokens, submits 16 parallel workers,
# gathers results into a PrefillResult.
# ===========================================================================

@pl.function(level=Level.HOST, role=Role.ORCHESTRATOR)
def prefill_orchestrate(rt_l3: LevelRuntime,
                        chip_backend,
                        request) -> "PrefillResult":
    """L3 orchestrator: distribute prefill across 16 NPU chips.

    1. Shard input tokens across NUM_L2_PER_L3 chips
    2. Submit 16 parallel model_prefill_host workers
    3. Gather per-chip KV outputs and first-token logits
    4. Merge logits (reduce across chips for tensor parallelism)
    5. Return PrefillResult{kv_cache, first_token_id}
    """
    token_ids = request.token_ids
    shard_size = (len(token_ids) + NUM_L2_PER_L3 - 1) // NUM_L2_PER_L3

    kv_outs: list[Tensor] = []
    logits_list: list[Tensor] = []

    for chip_id in range(NUM_L2_PER_L3):
        start = chip_id * shard_size
        end = min(start + shard_size, len(token_ids))

        token_shard = rt_l3.make_tensor(end - start)
        kv_blocks = rt_l3.make_tensor(0)  # empty on first prefill
        kv_out = rt_l3.make_tensor(request.kv_size_per_chip)
        first_logits = rt_l3.make_tensor(request.vocab_size)

        rt_l3.submit_worker(
            name="model_prefill_host",
            fn=lambda _cb=chip_backend, _cid=chip_id,
                      _ts=token_shard, _kb=kv_blocks,
                      _ko=kv_out, _fl=first_logits:
                model_prefill_host(_cb, _cid, _ts, _kb, _ko, _fl),
            inputs=[token_shard, kv_blocks],
            outputs=[kv_out, first_logits],
        )

        kv_outs.append(kv_out)
        logits_list.append(first_logits)

    # Merge logits across chips (tensor parallelism allreduce)
    merged_logits = pl.tree_reduce(
        rt_l3, logits_list,
        pair_fn=lambda a, b, out: element_wise_sum(a, b, out),
        name="logits_allreduce",
    )

    # Sample first token from merged logits
    first_token = rt_l3.make_tensor(1)
    rt_l3.submit_worker(
        name="sample_token",
        fn=lambda _l=merged_logits, _t=request.temperature,
                  _p=request.top_p, _out=first_token:
            sample_token_fn(_l, _t, _p, _out),
        inputs=[merged_logits],
        outputs=[first_token],
    )

    return PrefillResult(
        kv_cache=kv_outs,
        first_token=first_token,
    )


def element_wise_sum(a: Tensor, b: Tensor, out: Tensor):
    """Helper for allreduce: element-wise sum of logits across chips."""
    for k in range(out.count):
        out[k] = a[k] + b[k]


def sample_token_fn(logits: Tensor, temperature: float, top_p: float,
                    token_out: Tensor):
    """Sample one token from logits.  Phase 0: argmax."""
    best_idx = 0
    best_val = logits[0]
    for k in range(1, logits.count):
        if logits[k] > best_val:
            best_val = logits[k]
            best_idx = k
    token_out[0] = best_idx
