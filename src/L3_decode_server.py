"""
L3_decode_server.py — L3 Decode Server (PyPTO DSL spec)

Defines the L3 (Host/PC16) orchestrator and worker functions for the
Decode server instance (L3[1]).  This PC16 server has 16 NPU chips.

PyPTO coding rule:
    - ONE orchestrator function: decode_orchestrate
    - Multiple worker functions: model_decode_host (×16), sample_token

The orchestrator runs the autoregressive loop: each step submits 16
parallel model_decode_host workers (tensor parallelism), then a
sample_token worker to pick the next token.  Loop exits on EOS or
max_tokens.

Grammar reference: machine_hierarchy_and_function_hierarchy.md §5.7
C++ implementation: engine/inference_engine.cpp (decode_orchestrate)
                    engine/L3_workers.cpp       (model_decode_host, sample_token)
"""

import pl
from pl import Tensor, Level, Role, LevelRuntime

NUM_L2_PER_L3: int = 16  # 16 NPU chips per PC16 server
EOS_TOKEN: int = 2       # end-of-sequence token id


# ===========================================================================
# L3 WORKER FUNCTIONS
# ===========================================================================

@pl.function(level=Level.HOST, role=Role.WORKER)
def model_decode_host(chip_backend,
                      chip_id: int,
                      prev_token: Tensor,
                      kv_blocks: Tensor,
                      next_logits: Tensor,
                      updated_kv: Tensor):
    """L3 worker: single decode step on one NPU chip via ChipBackend.

    Wraps the L2 model_decode call with host-side data movement:
      h2d_copy → L2 model_decode → d2h_copy
    Phase 0: ChipBackend stub does everything in host memory.
    """
    chip_backend.model_decode(chip_id, prev_token, kv_blocks,
                              next_logits, updated_kv)


@pl.function(level=Level.HOST, role=Role.WORKER)
def sample_token(logits: Tensor,
                 temperature: float,
                 top_p: float,
                 token_out: Tensor):
    """L3 worker: sample next token from logits.

    Phase 0: simple argmax (temperature/top_p ignored in stub).
    Future: softmax → top-p filtering → temperature scaling → multinomial.
    """
    best_idx = 0
    best_val = logits[0]
    for k in range(1, logits.count):
        if logits[k] > best_val:
            best_val = logits[k]
            best_idx = k
    token_out[0] = best_idx


# ===========================================================================
# L3 ORCHESTRATOR FUNCTION
# ===========================================================================

@pl.function(level=Level.HOST, role=Role.ORCHESTRATOR)
def decode_orchestrate(rt_l3: LevelRuntime,
                       chip_backend,
                       prefill_result: "PrefillResult",
                       request) -> "DecodeResult":
    """L3 orchestrator: autoregressive decode loop.

    1. Start from prefill_result.first_token and kv_cache
    2. Each AR step:
       a. Submit 16 parallel model_decode_host workers (tensor parallelism)
       b. Allreduce logits across chips
       c. Submit sample_token worker
       d. Check stop conditions (EOS, max_tokens)
    3. Return DecodeResult{output_tokens}
    """
    kv_caches = prefill_result.kv_cache       # list of 16 per-chip KV tensors
    current_token = prefill_result.first_token
    output_tokens: list[Tensor] = [current_token]

    for step in range(request.max_tokens):
        # Submit 16 parallel decode workers (one per NPU chip)
        logits_list: list[Tensor] = []
        new_kv_list: list[Tensor] = []

        for chip_id in range(NUM_L2_PER_L3):
            next_logits = rt_l3.make_tensor(request.vocab_size)
            updated_kv = rt_l3.make_tensor(request.kv_step_size)

            rt_l3.submit_worker(
                name="model_decode_host",
                fn=lambda _cb=chip_backend, _cid=chip_id,
                          _pt=current_token, _kb=kv_caches[chip_id],
                          _nl=next_logits, _uk=updated_kv:
                    model_decode_host(_cb, _cid, _pt, _kb, _nl, _uk),
                inputs=[current_token, kv_caches[chip_id]],
                outputs=[next_logits, updated_kv],
            )

            logits_list.append(next_logits)
            new_kv_list.append(updated_kv)

        # Allreduce logits across 16 chips
        merged_logits = pl.tree_reduce(
            rt_l3, logits_list,
            pair_fn=lambda a, b, out: element_wise_sum(a, b, out),
            name="logits_allreduce",
        )

        # Sample next token
        next_token = rt_l3.make_tensor(1)
        rt_l3.submit_worker(
            name="sample_token",
            fn=lambda _l=merged_logits, _t=request.temperature,
                      _p=request.top_p, _out=next_token:
                sample_token(_l, _t, _p, _out),
            inputs=[merged_logits],
            outputs=[next_token],
        )

        output_tokens.append(next_token)

        # Check stop conditions
        if next_token.scalar() == EOS_TOKEN:
            break
        if request.stop_token >= 0 and next_token.scalar() == request.stop_token:
            break

        # Advance state for next AR step
        current_token = next_token
        kv_caches = new_kv_list

    return DecodeResult(output_tokens=output_tokens)


def element_wise_sum(a: Tensor, b: Tensor, out: Tensor):
    """Helper for allreduce: element-wise sum of logits across chips."""
    for k in range(out.count):
        out[k] = a[k] + b[k]
