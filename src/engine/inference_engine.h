#ifndef PYPTO_SERVING_ENGINE_INFERENCE_ENGINE_H
#define PYPTO_SERVING_ENGINE_INFERENCE_ENGINE_H

#include "common/request.h"
#include "l2_stubs/chip_backend.h"
#include "runtime/level_runtime.h"
#include "profiling/trace_writer.h"

namespace serving {

// InferenceEngine — L3 Host-level runtime for a single PC16 server.
//
// Owns one LevelRuntime(level=3, sched=1, workers=16).
// Provides two orchestration entry points matching the PyPTO DSL:
//   - prefill_orchestrate  (L3_prefill_server.py)
//   - decode_orchestrate   (L3_decode_server.py)
//
// The orchestration functions are called FROM the LevelRuntime's
// orchestrator thread (via submit_orchestrator).  They submit L3
// worker tasks that execute on the 16 worker threads.

class InferenceEngine {
public:
    InferenceEngine(int num_chips, ChipBackend* backend,
                    const linqu::LevelRuntimeConfig& cfg = {});

    void set_trace_writer(linqu::TraceWriter* tw);
    void register_trace_instance(int32_t trace_pid, const std::string& label);

    void start();
    void stop();

    linqu::LevelRuntime& runtime() { return rt_; }

    // L3 orchestrator: prefill (DSL spec: L3_prefill_server.py)
    PrefillResult prefill_orchestrate(const Request& request,
                                      int32_t trace_pid = -1);

    // L3 orchestrator: decode (DSL spec: L3_decode_server.py)
    DecodeResult decode_orchestrate(const PrefillResult& prefill,
                                    const Request& request,
                                    int32_t trace_pid = -1);

private:
    int num_chips_;
    ChipBackend* backend_;
    linqu::LevelRuntime rt_;
};

}  // namespace serving

#endif
