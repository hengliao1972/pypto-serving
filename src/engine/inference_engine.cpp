#include "engine/inference_engine.h"
#include "engine/L3_workers.h"
#include "runtime/tree_reduce.h"

#include <cassert>
#include <cstdio>
#include <future>
#include <vector>

namespace serving {

InferenceEngine::InferenceEngine(int num_chips, ChipBackend* backend,
                                 const linqu::LevelRuntimeConfig& cfg)
    : num_chips_(num_chips),
      backend_(backend),
      rt_(3, /*sched_threads=*/1, /*worker_threads=*/num_chips, cfg) {}

void InferenceEngine::set_trace_writer(linqu::TraceWriter* tw) {
    rt_.set_trace_writer(tw);
}

void InferenceEngine::register_trace_instance(int32_t trace_pid,
                                               const std::string& label) {
    rt_.register_trace_instance(trace_pid, label);
}

void InferenceEngine::start() { rt_.start(); }
void InferenceEngine::stop()  { rt_.stop(); }

// =========================================================================
// prefill_orchestrate  —  DSL spec: L3_prefill_server.py
//
// Orchestrator: submits 16 parallel model_prefill_host workers, one per
// NPU chip.  Then allreduces logits and samples the first token.
// =========================================================================

PrefillResult InferenceEngine::prefill_orchestrate(const Request& request,
                                                    int32_t trace_pid) {
    const int32_t vocab_size = request.vocab_size;
    const int32_t kv_size    = request.kv_size_per_chip;
    const auto token_count   = static_cast<int32_t>(request.token_ids.size());
    const int32_t shard_size = (token_count + num_chips_ - 1) / num_chips_;

    std::vector<linqu::LinquTensor> kv_outs(static_cast<size_t>(num_chips_));
    std::vector<linqu::LinquTensor> logits_list(static_cast<size_t>(num_chips_));

    // Submit 16 parallel prefill workers
    for (int chip = 0; chip < num_chips_; chip++) {
        int32_t start = chip * shard_size;
        int32_t end   = start + shard_size;
        if (end > token_count) end = token_count;
        int32_t shard_len = (start < token_count) ? (end - start) : 0;
        if (shard_len <= 0) shard_len = 1;

        linqu::LinquTensor token_shard = rt_.make_tensor(static_cast<size_t>(shard_len));
        linqu::LinquTensor kv_blocks   = rt_.make_tensor(1);
        linqu::LinquTensor kv_out      = rt_.make_tensor(static_cast<size_t>(kv_size));
        linqu::LinquTensor first_logits = rt_.make_tensor(static_cast<size_t>(vocab_size));

        kv_outs[static_cast<size_t>(chip)]      = kv_out;
        logits_list[static_cast<size_t>(chip)]  = first_logits;

        // token_shard and kv_blocks are initial data, not produced by any
        // prior task in this runtime — captured by lambda, NOT listed as
        // DAG inputs (otherwise the scheduler blocks on their ready flag).
        rt_.submit_worker(
            "model_prefill_host",
            [this, chip, token_shard, kv_blocks, kv_out, first_logits]() {
                model_prefill_host(backend_, chip,
                                   token_shard, kv_blocks,
                                   kv_out, first_logits);
            },
            {},
            {kv_out, first_logits},
            trace_pid);
    }

    // Allreduce logits across chips
    linqu::LinquTensor merged_logits = linqu::tree_reduce(
        rt_, logits_list,
        [](linqu::LinquTensor a, linqu::LinquTensor b, linqu::LinquTensor out) {
            logits_allreduce_pair(a, b, out);
        },
        "logits_allreduce", trace_pid);

    // Sample first token
    linqu::LinquTensor first_token = rt_.make_tensor(1);
    auto sample_fut = rt_.submit_worker(
        "sample_token",
        [merged_logits, first_token, temp = request.temperature,
         tp = request.top_p]() {
            sample_token(merged_logits, temp, tp, first_token);
        },
        {merged_logits},
        {first_token},
        trace_pid);
    sample_fut.get();

    return PrefillResult{kv_outs, first_token};
}

// =========================================================================
// decode_orchestrate  —  DSL spec: L3_decode_server.py
//
// Orchestrator: autoregressive loop.  Each step submits 16 parallel
// model_decode_host workers, allreduces logits, samples next token.
// =========================================================================

DecodeResult InferenceEngine::decode_orchestrate(const PrefillResult& prefill,
                                                  const Request& request,
                                                  int32_t trace_pid) {
    const int32_t vocab_size = request.vocab_size;
    const int32_t kv_step    = request.kv_step_size;
    const int32_t max_tokens = request.max_tokens;

    std::vector<linqu::LinquTensor> kv_caches = prefill.kv_cache;
    linqu::LinquTensor current_token = prefill.first_token;

    std::vector<linqu::LinquTensor> output_tokens;
    output_tokens.push_back(current_token);

    for (int32_t step = 0; step < max_tokens; step++) {
        std::vector<linqu::LinquTensor> logits_list(static_cast<size_t>(num_chips_));
        std::vector<linqu::LinquTensor> new_kv_list(static_cast<size_t>(num_chips_));

        // Submit 16 parallel decode workers
        for (int chip = 0; chip < num_chips_; chip++) {
            linqu::LinquTensor next_logits = rt_.make_tensor(
                static_cast<size_t>(vocab_size));
            linqu::LinquTensor updated_kv = rt_.make_tensor(
                static_cast<size_t>(kv_step));

            logits_list[static_cast<size_t>(chip)] = next_logits;
            new_kv_list[static_cast<size_t>(chip)] = updated_kv;

            rt_.submit_worker(
                "model_decode_host",
                [this, chip, current_token,
                 kv_in = kv_caches[static_cast<size_t>(chip)],
                 next_logits, updated_kv]() {
                    model_decode_host(backend_, chip,
                                     current_token, kv_in,
                                     next_logits, updated_kv);
                },
                {current_token, kv_caches[static_cast<size_t>(chip)]},
                {next_logits, updated_kv},
                trace_pid);
        }

        // Allreduce logits across chips
        linqu::LinquTensor merged_logits = linqu::tree_reduce(
            rt_, logits_list,
            [](linqu::LinquTensor a, linqu::LinquTensor b,
               linqu::LinquTensor out) {
                logits_allreduce_pair(a, b, out);
            },
            "logits_allreduce", trace_pid);

        // Sample next token
        linqu::LinquTensor next_token = rt_.make_tensor(1);
        auto sample_fut = rt_.submit_worker(
            "sample_token",
            [merged_logits, next_token, temp = request.temperature,
             tp = request.top_p]() {
                sample_token(merged_logits, temp, tp, next_token);
            },
            {merged_logits},
            {next_token},
            trace_pid);
        sample_fut.get();

        output_tokens.push_back(next_token);

        // Check stop conditions
        uint64_t tok_val = next_token.data_ptr()[0];
        if (request.stop_token >= 0 &&
            tok_val == static_cast<uint64_t>(request.stop_token))
            break;

        current_token = next_token;
        kv_caches = new_kv_list;
    }

    return DecodeResult{output_tokens};
}

}  // namespace serving
