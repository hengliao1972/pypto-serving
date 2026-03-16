#ifndef PYPTO_SERVING_L2_STUBS_CHIP_BACKEND_DLOPEN_H
#define PYPTO_SERVING_L2_STUBS_CHIP_BACKEND_DLOPEN_H

#include "l2_stubs/chip_backend.h"

#include <atomic>
#include <string>

namespace serving {

// ChipBackendDlopen — loads simpler's libhost_runtime.so at runtime via dlopen.
//
// Phase 4: wraps the real L2 API functions:
//   - simpler_model_prefill(chip_id, token_ids_ptr, n_tokens,
//                           kv_in_ptr, kv_size, kv_out_ptr, logits_ptr, vocab)
//   - simpler_model_decode(chip_id, prev_token, kv_in_ptr, kv_size,
//                          logits_ptr, vocab, updated_kv_ptr, kv_step)
//
// If the .so cannot be found, falls back to ChipBackendStub behavior
// (prints a warning, fills deterministic data).
//
// Expected function signatures (C linkage):
//   void simpler_model_prefill(int chip_id, const uint64_t* tokens, int n_tokens,
//                              const uint64_t* kv_in, int kv_size,
//                              uint64_t* kv_out, uint64_t* logits, int vocab);
//   void simpler_model_decode(int chip_id, uint64_t prev_token,
//                             const uint64_t* kv_in, int kv_size,
//                             uint64_t* logits, int vocab,
//                             uint64_t* updated_kv, int kv_step);

class ChipBackendDlopen : public ChipBackend {
public:
    explicit ChipBackendDlopen(int num_chips = 16,
                                const std::string& lib_path = "");

    ~ChipBackendDlopen() override;

    ChipBackendDlopen(const ChipBackendDlopen&) = delete;
    ChipBackendDlopen& operator=(const ChipBackendDlopen&) = delete;

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

    bool is_loaded() const { return handle_ != nullptr; }
    int num_chips() const { return num_chips_; }

    std::atomic<int> prefill_calls{0};
    std::atomic<int> decode_calls{0};

private:
    int num_chips_;
    void* handle_ = nullptr;  // dlopen handle

    // Function pointers
    using PrefillFn = void(*)(int, const uint64_t*, int,
                               const uint64_t*, int,
                               uint64_t*, uint64_t*, int);
    using DecodeFn = void(*)(int, uint64_t,
                              const uint64_t*, int,
                              uint64_t*, int,
                              uint64_t*, int);
    PrefillFn fn_prefill_ = nullptr;
    DecodeFn  fn_decode_  = nullptr;

    // Fallback stub behavior when .so not available.
    void stub_prefill(int chip_id, linqu::LinquTensor kv_out,
                      linqu::LinquTensor first_logits);
    void stub_decode(int chip_id, linqu::LinquTensor next_logits,
                     linqu::LinquTensor updated_kv);
};

}  // namespace serving

#endif
