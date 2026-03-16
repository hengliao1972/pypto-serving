#ifndef PYPTO_SERVING_COMMON_REQUEST_H
#define PYPTO_SERVING_COMMON_REQUEST_H

#include "core/tensor.h"
#include <cstdint>
#include <vector>

namespace serving {

struct Request {
    std::vector<uint64_t> token_ids;
    int32_t max_tokens    = 16;
    float   temperature   = 1.0f;
    float   top_p         = 1.0f;
    int32_t stop_token    = -1;
    int32_t vocab_size    = 32;
    int32_t kv_size_per_chip = 64;
    int32_t kv_step_size  = 64;
};

struct PrefillResult {
    std::vector<linqu::LinquTensor> kv_cache;
    linqu::LinquTensor              first_token;
};

struct DecodeResult {
    std::vector<linqu::LinquTensor> output_tokens;
};

struct Response {
    std::vector<linqu::LinquTensor> output_tokens;
};

}  // namespace serving

#endif
