#ifndef PYPTO_SERVING_ENGINE_POD_ORCHESTRATOR_H
#define PYPTO_SERVING_ENGINE_POD_ORCHESTRATOR_H

#include "common/request.h"
#include "engine/inference_engine.h"
#include "runtime/level_runtime.h"
#include "profiling/trace_writer.h"

namespace serving {

// PodOrchestrator — L4 Pod-level orchestrator.
//
// Owns one LevelRuntime(level=4, sched=1, workers=2).
// Routes requests through the Prefill → Decode pipeline:
//   1. submit_orchestrator → L3[0] prefill_orchestrate
//   2. submit_orchestrator → L3[1] decode_orchestrate
//
// DSL spec: L4_pod_orchestrator.py

class PodOrchestrator {
public:
    PodOrchestrator(InferenceEngine* prefill_engine,
                    InferenceEngine* decode_engine,
                    const linqu::LevelRuntimeConfig& cfg = {});

    void set_trace_writer(linqu::TraceWriter* tw);
    void register_trace_instance(int32_t trace_pid, const std::string& label);

    void start();
    void stop();

    linqu::LevelRuntime& runtime() { return rt_; }

    // L4 orchestrator: full prefill → decode pipeline.
    // DSL spec: L4_pod_orchestrator.py::pod_orchestrate
    Response handle_request(const Request& request,
                            int32_t trace_pid = -1);

private:
    InferenceEngine* prefill_engine_;
    InferenceEngine* decode_engine_;
    linqu::LevelRuntime rt_;
};

}  // namespace serving

#endif
