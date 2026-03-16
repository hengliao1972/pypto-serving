#include "engine/L3_workers.h"
#include <cassert>
#include <cstdint>

namespace serving {

void model_prefill_host(ChipBackend* backend,
                        int chip_id,
                        linqu::LinquTensor token_shard,
                        linqu::LinquTensor kv_blocks,
                        linqu::LinquTensor kv_out,
                        linqu::LinquTensor first_logits) {
    backend->model_prefill(chip_id, token_shard, kv_blocks,
                           kv_out, first_logits);
}

void model_decode_host(ChipBackend* backend,
                       int chip_id,
                       linqu::LinquTensor prev_token,
                       linqu::LinquTensor kv_blocks,
                       linqu::LinquTensor next_logits,
                       linqu::LinquTensor updated_kv) {
    backend->model_decode(chip_id, prev_token, kv_blocks,
                          next_logits, updated_kv);
}

void sample_token(linqu::LinquTensor logits,
                  float /*temperature*/,
                  float /*top_p*/,
                  linqu::LinquTensor token_out) {
    // Phase 0: argmax over logits (temperature/top_p ignored).
    assert(logits.data_ptr() && token_out.data_ptr());
    assert(logits.count > 0 && token_out.count >= 1);

    uint64_t* lp = logits.data_ptr();
    uint64_t best_val = lp[0];
    uint64_t best_idx = 0;
    for (size_t k = 1; k < logits.count; k++) {
        if (lp[k] > best_val) {
            best_val = lp[k];
            best_idx = static_cast<uint64_t>(k);
        }
    }
    token_out.data_ptr()[0] = best_idx;
}

void logits_allreduce_pair(linqu::LinquTensor a,
                           linqu::LinquTensor b,
                           linqu::LinquTensor out) {
    assert(a.data_ptr() && b.data_ptr() && out.data_ptr());
    assert(a.count == b.count && out.count == a.count);
    uint64_t* ap = a.data_ptr();
    uint64_t* bp = b.data_ptr();
    uint64_t* op = out.data_ptr();
    for (size_t k = 0; k < out.count; k++)
        op[k] = ap[k] + bp[k];
}

}  // namespace serving
