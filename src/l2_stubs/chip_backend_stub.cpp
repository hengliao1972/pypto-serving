#include "l2_stubs/chip_backend_stub.h"
#include <cassert>
#include <cstdint>

namespace serving {

ChipBackendStub::ChipBackendStub(int num_chips)
    : num_chips_(num_chips) {}

void ChipBackendStub::model_prefill(int chip_id,
                                     linqu::LinquTensor token_ids,
                                     linqu::LinquTensor kv_blocks,
                                     linqu::LinquTensor kv_out,
                                     linqu::LinquTensor first_logits) {
    assert(chip_id >= 0 && chip_id < num_chips_);
    prefill_calls.fetch_add(1, std::memory_order_relaxed);

    uint64_t* kv_p = kv_out.data_ptr();
    if (kv_p) {
        uint64_t seed = static_cast<uint64_t>(chip_id) * 1000 + 42;
        for (size_t i = 0; i < kv_out.count; i++)
            kv_p[i] = seed + i;
    }

    uint64_t* logits_p = first_logits.data_ptr();
    if (logits_p) {
        for (size_t i = 0; i < first_logits.count; i++)
            logits_p[i] = 0;
        // Deterministic "prediction": chip_id selects the winning logit index.
        size_t winner = static_cast<size_t>(chip_id) % first_logits.count;
        logits_p[winner] = 100 + static_cast<uint64_t>(chip_id);
    }
}

void ChipBackendStub::model_decode(int chip_id,
                                    linqu::LinquTensor prev_token,
                                    linqu::LinquTensor kv_blocks,
                                    linqu::LinquTensor next_logits,
                                    linqu::LinquTensor updated_kv) {
    assert(chip_id >= 0 && chip_id < num_chips_);
    decode_calls.fetch_add(1, std::memory_order_relaxed);

    // Read previous token for deterministic next-token generation.
    uint64_t prev = 0;
    if (prev_token.data_ptr())
        prev = prev_token.data_ptr()[0];

    uint64_t* logits_p = next_logits.data_ptr();
    if (logits_p) {
        for (size_t i = 0; i < next_logits.count; i++)
            logits_p[i] = 0;
        // Deterministic: next token = (prev + chip_id + 1) % vocab_size.
        size_t winner = static_cast<size_t>((prev + chip_id + 1) % next_logits.count);
        logits_p[winner] = 200 + static_cast<uint64_t>(chip_id);
    }

    uint64_t* kv_p = updated_kv.data_ptr();
    if (kv_p) {
        // Copy existing KV and append a step marker.
        uint64_t* src = kv_blocks.data_ptr();
        size_t copy_len = kv_blocks.count < updated_kv.count
                          ? kv_blocks.count : updated_kv.count;
        if (src) {
            for (size_t i = 0; i < copy_len; i++)
                kv_p[i] = src[i];
        }
        for (size_t i = copy_len; i < updated_kv.count; i++)
            kv_p[i] = prev + chip_id;
    }
}

}  // namespace serving
