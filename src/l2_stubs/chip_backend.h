#ifndef PYPTO_SERVING_L2_STUBS_CHIP_BACKEND_H
#define PYPTO_SERVING_L2_STUBS_CHIP_BACKEND_H

#include "core/tensor.h"

namespace serving {

// Abstract interface for L2 chip operations.
// In production, this wraps dlopen of simpler's libhost_runtime.so.
// Phase 0: ChipBackendStub provides deterministic test data.
class ChipBackend {
public:
    virtual ~ChipBackend() = default;

    virtual void model_prefill(int chip_id,
                               linqu::LinquTensor token_ids,
                               linqu::LinquTensor kv_blocks,
                               linqu::LinquTensor kv_out,
                               linqu::LinquTensor first_logits) = 0;

    virtual void model_decode(int chip_id,
                              linqu::LinquTensor prev_token,
                              linqu::LinquTensor kv_blocks,
                              linqu::LinquTensor next_logits,
                              linqu::LinquTensor updated_kv) = 0;
};

}  // namespace serving

#endif
