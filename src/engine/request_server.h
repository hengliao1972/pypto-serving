#ifndef PYPTO_SERVING_ENGINE_REQUEST_SERVER_H
#define PYPTO_SERVING_ENGINE_REQUEST_SERVER_H

#include "common/request.h"
#include "engine/stop_condition.h"
#include "kv/kv_cache_manager.h"
#include "kv/radix_tree.h"

namespace serving {

class InferenceEngine;
class PodOrchestrator;

// RequestServer — orchestrates a single inference request through the
// Radix Tree → KV Cache → Prefill → AR loop → Radix update pipeline.
//
// This corresponds to the Phase 3 "serve_request" orchestrator that
// integrates the KV cache prefix sharing with the L3 inference pipeline.
//
// Flow:
//   1. Parse request → StopConfig
//   2. Radix Tree prefix match → determine cache hit length
//   3. Allocate KV blocks for new tokens
//   4. Prefill only the un-cached suffix
//   5. AR loop with StopChecker
//   6. Update Radix Tree with new prefix
//   7. Return response

struct ServeResult {
    Response response;
    int prefix_hit_tokens = 0;
    int total_tokens_generated = 0;
    StopChecker::Reason stop_reason = StopChecker::Reason::NONE;
};

class RequestServer {
public:
    RequestServer(RadixTree* radix, KVCacheManager* kv_mgr);

    // Full serve_request flow: Radix lookup → prefill → decode → Radix update.
    // Uses the PodOrchestrator (L4) pipeline for actual computation.
    ServeResult serve_request(PodOrchestrator& pod,
                              InferenceEngine& prefill_engine,
                              InferenceEngine& decode_engine,
                              const Request& request,
                              int32_t trace_pid = -1);

    // Parse a Request into a StopConfig.
    static StopConfig make_stop_config(const Request& request);

private:
    RadixTree* radix_;
    KVCacheManager* kv_mgr_;
};

}  // namespace serving

#endif
