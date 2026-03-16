#include "engine/pod_orchestrator.h"

#include <cassert>
#include <cstdio>
#include <future>

namespace serving {

PodOrchestrator::PodOrchestrator(InferenceEngine* prefill_engine,
                                 InferenceEngine* decode_engine,
                                 const linqu::LevelRuntimeConfig& cfg)
    : prefill_engine_(prefill_engine),
      decode_engine_(decode_engine),
      rt_(4, /*sched_threads=*/1, /*worker_threads=*/2, cfg) {}

void PodOrchestrator::set_trace_writer(linqu::TraceWriter* tw) {
    rt_.set_trace_writer(tw);
}

void PodOrchestrator::register_trace_instance(int32_t trace_pid,
                                               const std::string& label) {
    rt_.register_trace_instance(trace_pid, label);
}

void PodOrchestrator::start() { rt_.start(); }
void PodOrchestrator::stop()  { rt_.stop(); }

// =========================================================================
// handle_request  —  DSL spec: L4_pod_orchestrator.py::pod_orchestrate
//
// L4 orchestrator function.  No L4-level workers are used; this level
// only coordinates between the two L3 server instances.
//
// Step 1: Submit prefill_orchestrate to L3[0] (Prefill PC16)
// Step 2: Wait for PrefillResult
// Step 3: Submit decode_orchestrate to L3[1] (Decode PC16)
// Step 4: Wait for DecodeResult
// Step 5: Return Response
// =========================================================================

Response PodOrchestrator::handle_request(const Request& request,
                                          int32_t trace_pid) {
    // Step 1: Prefill on L3[0]
    auto prefill_future = prefill_engine_->runtime().submit_orchestrator(
        "prefill_orchestrate",
        [this, &request, trace_pid]() -> PrefillResult {
            return prefill_engine_->prefill_orchestrate(request, trace_pid);
        },
        trace_pid);

    PrefillResult prefill_result = prefill_future.get();

    // Step 2: Decode on L3[1]
    auto decode_future = decode_engine_->runtime().submit_orchestrator(
        "decode_orchestrate",
        [this, &prefill_result, &request, trace_pid]() -> DecodeResult {
            return decode_engine_->decode_orchestrate(
                prefill_result, request, trace_pid);
        },
        trace_pid);

    DecodeResult decode_result = decode_future.get();

    return Response{decode_result.output_tokens};
}

}  // namespace serving
