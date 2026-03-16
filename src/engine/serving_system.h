#ifndef PYPTO_SERVING_ENGINE_SERVING_SYSTEM_H
#define PYPTO_SERVING_ENGINE_SERVING_SYSTEM_H

#include "common/request.h"
#include "engine/inference_engine.h"
#include "engine/pod_orchestrator.h"
#include "engine/request_server.h"
#include "kv/kv_cache_manager.h"
#include "kv/radix_tree.h"
#include "l2_stubs/chip_backend_stub.h"
#include "profiling/trace_writer.h"

#include <memory>
#include <string>

namespace serving {

// ServingSystem — top-level class for the PC16 x2 serving topology.
//
// Owns all runtime components:
//   L4  PodOrchestrator   (sched=1, workers=2)
//   L3[0] InferenceEngine (Prefill PC16, sched=1, workers=16)
//   L3[1] InferenceEngine (Decode  PC16, sched=1, workers=16)
//   L2  ChipBackendStub   (16 chips per L3)
//   TraceWriter           (shared across all levels)
//
// DSL spec: serving_main.py

struct ServingConfig {
    int num_chips_per_server = 16;
    bool enable_trace = false;
    std::string trace_path = "pypto_serving_trace.json";
};

class ServingSystem {
public:
    explicit ServingSystem(const ServingConfig& cfg = {});
    ~ServingSystem();

    ServingSystem(const ServingSystem&) = delete;
    ServingSystem& operator=(const ServingSystem&) = delete;

    void start();
    void stop();

    Response infer(const Request& request);

    // Phase 3: serve_request with Radix + KV cache integration.
    ServeResult serve(const Request& request);

    // Write trace and return path (empty string if tracing disabled).
    std::string write_trace();

    linqu::TraceWriter& trace_writer() { return trace_writer_; }
    ChipBackendStub& prefill_backend() { return prefill_backend_; }
    ChipBackendStub& decode_backend()  { return decode_backend_; }
    RadixTree& radix_tree() { return radix_; }
    KVCacheManager& kv_cache() { return kv_mgr_; }

private:
    ServingConfig cfg_;
    linqu::TraceWriter trace_writer_;
    ChipBackendStub prefill_backend_;
    ChipBackendStub decode_backend_;
    InferenceEngine prefill_engine_;
    InferenceEngine decode_engine_;
    PodOrchestrator pod_;
    RadixTree radix_;
    KVCacheManager kv_mgr_;
    RequestServer req_server_;
};

}  // namespace serving

#endif
