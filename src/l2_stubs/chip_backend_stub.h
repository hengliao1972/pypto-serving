#ifndef PYPTO_SERVING_L2_STUBS_CHIP_BACKEND_STUB_H
#define PYPTO_SERVING_L2_STUBS_CHIP_BACKEND_STUB_H

#include "l2_stubs/chip_backend.h"
#include <atomic>

namespace serving {

// Deterministic stub for L2 chip operations.
// Fills output tensors with repeatable data seeded by chip_id so tests
// can verify the full pipeline without real NPU hardware.
class ChipBackendStub : public ChipBackend {
public:
    explicit ChipBackendStub(int num_chips = 16);

    void model_prefill(int chip_id,
                       linqu::LinquTensor token_ids,
                       linqu::LinquTensor kv_blocks,
                       linqu::LinquTensor kv_out,
                       linqu::LinquTensor first_logits) override;

    void model_decode(int chip_id,
                      linqu::LinquTensor prev_token,
                      linqu::LinquTensor kv_blocks,
                      linqu::LinquTensor next_logits,
                      linqu::LinquTensor updated_kv) override;

    int num_chips() const { return num_chips_; }

    // Tracks total calls for verification.
    std::atomic<int> prefill_calls{0};
    std::atomic<int> decode_calls{0};

private:
    int num_chips_;
};

}  // namespace serving

#endif
