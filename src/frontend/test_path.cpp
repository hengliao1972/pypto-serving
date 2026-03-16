#include "frontend/test_path.h"
#include "engine/serving_system.h"

#include <cassert>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

// =========================================================================
// Wire format helpers
// =========================================================================

namespace {

struct WireRequest {
    uint32_t max_tokens;
    float    temperature;
    float    top_p;
    int32_t  stop_token;
    int32_t  vocab_size;
    int32_t  kv_size_per_chip;
    int32_t  kv_step_size;
    uint32_t n_tokens;
    // followed by n_tokens uint64_t values
};

static serving::Request deserialize_request(const void* buf, size_t len) {
    assert(len >= sizeof(WireRequest));
    auto* w = static_cast<const WireRequest*>(buf);

    serving::Request req;
    req.max_tokens      = static_cast<int32_t>(w->max_tokens);
    req.temperature     = w->temperature;
    req.top_p           = w->top_p;
    req.stop_token      = w->stop_token;
    req.vocab_size      = w->vocab_size;
    req.kv_size_per_chip = w->kv_size_per_chip;
    req.kv_step_size    = w->kv_step_size;

    auto* tokens = reinterpret_cast<const uint64_t*>(
        static_cast<const char*>(buf) + sizeof(WireRequest));
    req.token_ids.assign(tokens, tokens + w->n_tokens);

    return req;
}

static std::vector<uint8_t> serialize_response(const serving::Response& resp) {
    uint32_t n = static_cast<uint32_t>(resp.output_tokens.size());
    size_t total = sizeof(uint32_t) + n * sizeof(uint64_t);
    std::vector<uint8_t> buf(total);

    std::memcpy(buf.data(), &n, sizeof(n));
    auto* dst = reinterpret_cast<uint64_t*>(buf.data() + sizeof(uint32_t));
    for (uint32_t i = 0; i < n; i++) {
        auto& t = resp.output_tokens[i];
        dst[i] = (t.data_ptr() != nullptr) ? t.data_ptr()[0] : 0;
    }
    return buf;
}

// =========================================================================
// Global state
// =========================================================================

struct TestPathState {
    std::unique_ptr<serving::ServingSystem> system;

    std::mutex req_mu;
    std::condition_variable req_cv;
    std::queue<std::vector<uint8_t>> req_queue;

    std::mutex resp_mu;
    std::condition_variable resp_cv;
    std::queue<std::vector<uint8_t>> resp_queue;
};

static TestPathState* g_state = nullptr;

}  // namespace

// =========================================================================
// C API implementation
// =========================================================================

extern "C" {

int testpath_init(void) {
    if (g_state) return 0;
    g_state = new TestPathState();
    return 0;
}

void testpath_shutdown(void) {
    if (!g_state) return;
    testpath_stop();
    delete g_state;
    g_state = nullptr;
}

int testpath_start(const char* trace_path) {
    if (!g_state) return -1;

    serving::ServingConfig cfg;
    cfg.num_chips_per_server = 16;
    if (trace_path && trace_path[0]) {
        cfg.enable_trace = true;
        cfg.trace_path = trace_path;
    }

    g_state->system = std::make_unique<serving::ServingSystem>(cfg);
    g_state->system->start();
    return 0;
}

void testpath_stop(void) {
    if (!g_state || !g_state->system) return;
    g_state->system->stop();
    g_state->system.reset();
}

int testpath_inject_request(const void* buf, size_t len) {
    if (!g_state || !g_state->system) return -1;
    if (!buf || len < sizeof(WireRequest)) return -2;

    serving::Request req = deserialize_request(buf, len);
    serving::Response resp = g_state->system->infer(req);
    std::vector<uint8_t> resp_buf = serialize_response(resp);

    {
        std::lock_guard<std::mutex> lk(g_state->resp_mu);
        g_state->resp_queue.push(std::move(resp_buf));
    }
    g_state->resp_cv.notify_one();
    return 0;
}

int64_t testpath_get_response(void* buf, size_t max_len) {
    if (!g_state) return -1;

    std::unique_lock<std::mutex> lk(g_state->resp_mu);
    g_state->resp_cv.wait(lk, [] { return !g_state->resp_queue.empty(); });

    auto resp_buf = std::move(g_state->resp_queue.front());
    g_state->resp_queue.pop();
    lk.unlock();

    size_t copy = resp_buf.size();
    if (copy > max_len) copy = max_len;
    std::memcpy(buf, resp_buf.data(), copy);
    return static_cast<int64_t>(copy);
}

int testpath_write_trace(void) {
    if (!g_state || !g_state->system) return -1;
    std::string path = g_state->system->write_trace();
    return path.empty() ? -1 : 0;
}

}  // extern "C"
