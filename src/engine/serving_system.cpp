#include "engine/serving_system.h"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <future>

namespace serving {

ServingSystem::ServingSystem(const ServingConfig& cfg)
    : cfg_(cfg),
      prefill_backend_(cfg.num_chips_per_server),
      decode_backend_(cfg.num_chips_per_server),
      prefill_engine_(cfg.num_chips_per_server, &prefill_backend_),
      decode_engine_(cfg.num_chips_per_server, &decode_backend_),
      pod_(&prefill_engine_, &decode_engine_) {

    if (cfg_.enable_trace) {
        trace_writer_.set_enabled(true);

        prefill_engine_.set_trace_writer(&trace_writer_);
        decode_engine_.set_trace_writer(&trace_writer_);
        pod_.set_trace_writer(&trace_writer_);

        prefill_engine_.register_trace_instance(30000, "L3[0] Prefill PC16");
        decode_engine_.register_trace_instance(30001, "L3[1] Decode PC16");
        pod_.register_trace_instance(40000, "L4 Pod");
    }
}

ServingSystem::~ServingSystem() {
    stop();
}

void ServingSystem::start() {
    prefill_engine_.start();
    decode_engine_.start();
    pod_.start();
}

void ServingSystem::stop() {
    pod_.stop();
    decode_engine_.stop();
    prefill_engine_.stop();
}

Response ServingSystem::infer(const Request& request) {
    int32_t trace_pid = cfg_.enable_trace ? 40000 : -1;

    auto response_future = pod_.runtime().submit_orchestrator(
        "pod_orchestrate",
        [this, &request, trace_pid]() -> Response {
            return pod_.handle_request(request, trace_pid);
        },
        trace_pid);

    return response_future.get();
}

std::string ServingSystem::write_trace() {
    if (!cfg_.enable_trace)
        return {};

    std::string written = trace_writer_.write_json(cfg_.trace_path);
    if (!written.empty()) {
        auto abs = std::filesystem::absolute(written);
        fprintf(stderr, "[TRACE] Written: %s\n", abs.c_str());
    }
    return written;
}

}  // namespace serving
