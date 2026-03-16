#ifndef PYPTO_SERVING_ENGINE_L3_WORKERS_H
#define PYPTO_SERVING_ENGINE_L3_WORKERS_H

#include "core/tensor.h"
#include "l2_stubs/chip_backend.h"

namespace serving {

// L3 worker functions — pure compute, called by L3 LevelRuntime worker threads.
// These match the @pl.function(level=HOST, role=WORKER) definitions in
// L3_prefill_server.py and L3_decode_server.py.

// Prefill worker: calls ChipBackend.model_prefill for one NPU chip.
// DSL spec: L3_prefill_server.py::model_prefill_host
void model_prefill_host(ChipBackend* backend,
                        int chip_id,
                        linqu::LinquTensor token_shard,
                        linqu::LinquTensor kv_blocks,
                        linqu::LinquTensor kv_out,
                        linqu::LinquTensor first_logits);

// Decode worker: calls ChipBackend.model_decode for one NPU chip.
// DSL spec: L3_decode_server.py::model_decode_host
void model_decode_host(ChipBackend* backend,
                       int chip_id,
                       linqu::LinquTensor prev_token,
                       linqu::LinquTensor kv_blocks,
                       linqu::LinquTensor next_logits,
                       linqu::LinquTensor updated_kv);

// Sampling worker: pick next token from logits.
// DSL spec: L3_decode_server.py::sample_token
void sample_token(linqu::LinquTensor logits,
                  float temperature,
                  float top_p,
                  linqu::LinquTensor token_out);

// Element-wise sum for allreduce across chip logits.
// DSL spec: L3_prefill_server.py::element_wise_sum (helper)
void logits_allreduce_pair(linqu::LinquTensor a,
                           linqu::LinquTensor b,
                           linqu::LinquTensor out);

}  // namespace serving

#endif
