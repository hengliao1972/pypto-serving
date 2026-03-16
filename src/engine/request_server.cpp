#include "engine/request_server.h"
#include "engine/inference_engine.h"
#include "engine/pod_orchestrator.h"

#include <cstdio>

namespace serving {

RequestServer::RequestServer(RadixTree* radix, KVCacheManager* kv_mgr)
    : radix_(radix), kv_mgr_(kv_mgr) {}

StopConfig RequestServer::make_stop_config(const Request& request) {
    StopConfig sc;
    sc.max_tokens = request.max_tokens;
    sc.eos_token  = request.stop_token;
    return sc;
}

// =========================================================================
// serve_request — Phase 3 orchestrator: Radix → Prefill → AR → Radix update
// =========================================================================

ServeResult RequestServer::serve_request(PodOrchestrator& pod,
                                          InferenceEngine& prefill_engine,
                                          InferenceEngine& decode_engine,
                                          const Request& request,
                                          int32_t trace_pid) {
    ServeResult result;
    StopConfig stop_cfg = make_stop_config(request);
    StopChecker checker(stop_cfg);

    // ── Step 1: Radix Tree prefix lookup ──────────────────────────────
    PrefixMatch match = radix_->find_prefix(request.token_ids);
    result.prefix_hit_tokens = match.matched_tokens;

    if (match.matched_tokens > 0 && match.node->kv_block != kInvalidBlock) {
        radix_->ref(match.node);
        radix_->touch(match.node);
    }

    // ── Step 2: Allocate KV block for this request ────────────────────
    size_t kv_block_size = static_cast<size_t>(request.kv_size_per_chip) *
                           static_cast<size_t>(16);  // 16 chips
    BlockHandle kv_block = kv_mgr_->alloc(CacheTier::L1, kv_block_size);
    if (kv_block == kInvalidBlock) {
        // Try evicting L1 and retry.
        kv_mgr_->evict(CacheTier::L1);
        kv_block = kv_mgr_->alloc(CacheTier::L1, kv_block_size);
    }
    if (kv_block != kInvalidBlock) {
        kv_mgr_->ref(kv_block);
    }

    // ── Step 3: Run the inference pipeline ────────────────────────────
    // We use the full L4 → L3 pipeline.  The prefix_hit info is logged
    // but the stub backend doesn't actually skip prefill for cached tokens
    // (that would require real KV data).  This validates the integration
    // plumbing; real skipping comes with Phase 4 real kernels.
    auto prefill_future = prefill_engine.runtime().submit_orchestrator(
        "prefill_orchestrate",
        [&]() -> PrefillResult {
            return prefill_engine.prefill_orchestrate(request, trace_pid);
        },
        trace_pid);

    PrefillResult prefill_result = prefill_future.get();

    // Check first token from prefill
    if (prefill_result.first_token.data_ptr()) {
        uint64_t first_tok = prefill_result.first_token.data_ptr()[0];
        if (checker.should_stop(first_tok)) {
            result.stop_reason = checker.stop_reason();
            result.total_tokens_generated = checker.tokens_generated();
            result.response.output_tokens.push_back(prefill_result.first_token);
            goto cleanup;
        }
    }

    {
        // ── Step 4: Decode (AR loop with stop checker) ────────────────
        auto decode_future = decode_engine.runtime().submit_orchestrator(
            "decode_orchestrate",
            [&]() -> DecodeResult {
                return decode_engine.decode_orchestrate(
                    prefill_result, request, trace_pid);
            },
            trace_pid);

        DecodeResult decode_result = decode_future.get();
        result.response.output_tokens = decode_result.output_tokens;

        // Apply stop checker retroactively to count tokens.
        for (auto& tok : decode_result.output_tokens) {
            if (tok.data_ptr()) {
                checker.should_stop(tok.data_ptr()[0]);
            }
        }
        result.total_tokens_generated = checker.tokens_generated();
        result.stop_reason = checker.stop_reason();
    }

cleanup:
    // ── Step 5: Update Radix Tree with this prefix ────────────────────
    radix_->insert(request.token_ids, kv_block);

    // Release references
    if (match.matched_tokens > 0 && match.node->kv_block != kInvalidBlock) {
        radix_->unref(match.node);
    }
    if (kv_block != kInvalidBlock) {
        kv_mgr_->unref(kv_block);
    }

    return result;
}

}  // namespace serving
