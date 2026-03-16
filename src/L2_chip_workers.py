"""
L2_chip_workers.py — L2 Chip-level worker function stubs (PyPTO DSL spec)

Defines the L2 (Chip/NPU) worker functions that execute on each NPU chip.
In the real system these run on the simpler runtime; Phase 0 uses stubs.

Each PC16 server has 16 NPU chips (NUM_L2_PER_L3 = 16).
L3 host workers call these L2 functions via ChipBackend adapter.

Grammar reference: machine_hierarchy_and_function_hierarchy.md §5.7
"""

import pl
from pl import Tensor, Level, Role


# ===========================================================================
# L2 WORKER FUNCTIONS  (@pl.function with level=CHIP, role=WORKER)
#
# These are the chip-side kernels.  In production they are compiled and
# dispatched by the simpler runtime on AIC/AIV hardware.
# Phase 0 stub: fills outputs with deterministic test data.
# ===========================================================================

@pl.function(level=Level.CHIP, role=Role.WORKER)
def model_prefill(chip_id: int,
                  token_ids: Tensor,
                  kv_blocks: Tensor,
                  kv_out: Tensor,
                  first_logits: Tensor):
    """Process full prompt on a single NPU chip.

    Inputs:
        chip_id      — which NPU chip (0..15) on this PC16 server
        token_ids    — input token shard assigned to this chip
        kv_blocks    — existing KV cache blocks (empty on first call)
    Outputs:
        kv_out       — updated KV cache blocks after processing prompt
        first_logits — logits for the first predicted token

    Phase 0 stub: fills kv_out with chip_id-seeded data,
    first_logits with a one-hot vector at position (chip_id % vocab_size).
    """
    ...


@pl.function(level=Level.CHIP, role=Role.WORKER)
def model_decode(chip_id: int,
                 prev_token: Tensor,
                 kv_blocks: Tensor,
                 next_logits: Tensor,
                 updated_kv: Tensor):
    """Single-step decode: predict next token from previous token + KV cache.

    Inputs:
        chip_id      — which NPU chip (0..15)
        prev_token   — the last generated token
        kv_blocks    — current KV cache
    Outputs:
        next_logits  — logits for the next token prediction
        updated_kv   — KV cache updated with this decode step

    Phase 0 stub: fills next_logits with deterministic data,
    updated_kv = copy of kv_blocks with one additional entry.
    """
    ...
