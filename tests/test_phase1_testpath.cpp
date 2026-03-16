/*
 * test_phase1_testpath.cpp — Phase 1 integration test for TestPath C API.
 *
 * Validates the full round-trip:
 *   serialize Request → testpath_inject_request → L4→L3 pipeline
 *   → testpath_get_response → deserialize Response → verify tokens
 *
 * Uses the same C API that Python ctypes will use.
 */

#include "frontend/test_path.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Wire format structs (must match test_path.cpp)
#pragma pack(push, 1)
struct WireRequest {
    uint32_t max_tokens;
    float    temperature;
    float    top_p;
    int32_t  stop_token;
    int32_t  vocab_size;
    int32_t  kv_size_per_chip;
    int32_t  kv_step_size;
    uint32_t n_tokens;
};
#pragma pack(pop)

static std::vector<uint8_t> build_request_buf(
    const std::vector<uint64_t>& token_ids,
    uint32_t max_tokens = 4,
    float temperature = 1.0f,
    float top_p = 1.0f,
    int32_t stop_token = -1,
    int32_t vocab_size = 32,
    int32_t kv_size = 64,
    int32_t kv_step = 64)
{
    size_t n = token_ids.size();
    size_t total = sizeof(WireRequest) + n * sizeof(uint64_t);
    std::vector<uint8_t> buf(total);

    auto* w = reinterpret_cast<WireRequest*>(buf.data());
    w->max_tokens      = max_tokens;
    w->temperature     = temperature;
    w->top_p           = top_p;
    w->stop_token      = stop_token;
    w->vocab_size      = vocab_size;
    w->kv_size_per_chip = kv_size;
    w->kv_step_size    = kv_step;
    w->n_tokens        = static_cast<uint32_t>(n);

    auto* dst = reinterpret_cast<uint64_t*>(buf.data() + sizeof(WireRequest));
    for (size_t i = 0; i < n; i++)
        dst[i] = token_ids[i];

    return buf;
}

static std::vector<uint64_t> parse_response_buf(
    const uint8_t* buf, size_t len)
{
    assert(len >= sizeof(uint32_t));
    uint32_t n;
    std::memcpy(&n, buf, sizeof(n));

    auto* tokens = reinterpret_cast<const uint64_t*>(buf + sizeof(uint32_t));
    return std::vector<uint64_t>(tokens, tokens + n);
}

int main(int argc, char* argv[]) {
    bool do_trace = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--trace") == 0)
            do_trace = true;
    }

    fprintf(stderr, "=== Phase 1: TestPath C API Integration Test ===\n\n");

    // Step 1: Init
    int rc = testpath_init();
    assert(rc == 0);
    fprintf(stderr, "[OK] testpath_init\n");

    // Step 2: Start engine
    const char* trace_path = do_trace ? "phase1_testpath_trace.json" : nullptr;
    rc = testpath_start(trace_path);
    assert(rc == 0);
    fprintf(stderr, "[OK] testpath_start (trace=%s)\n",
            trace_path ? trace_path : "off");

    // Step 3: Build and inject request
    std::vector<uint64_t> input_tokens = {101, 2023, 2003, 1037, 3231, 6251, 2005, 5604};
    auto req_buf = build_request_buf(input_tokens, /*max_tokens=*/4);
    fprintf(stderr, "[REQ] %zu input tokens, max_tokens=4, vocab=32\n",
            input_tokens.size());
    fprintf(stderr, "[REQ] wire buffer: %zu bytes\n", req_buf.size());

    rc = testpath_inject_request(req_buf.data(), req_buf.size());
    assert(rc == 0);
    fprintf(stderr, "[OK] testpath_inject_request\n");

    // Step 4: Get response
    std::vector<uint8_t> resp_buf(4096);
    int64_t resp_len = testpath_get_response(resp_buf.data(), resp_buf.size());
    assert(resp_len > 0);
    fprintf(stderr, "[OK] testpath_get_response: %lld bytes\n",
            (long long)resp_len);

    auto output_tokens = parse_response_buf(resp_buf.data(),
                                             static_cast<size_t>(resp_len));
    fprintf(stderr, "[RESULT] %zu output tokens:", output_tokens.size());
    for (auto t : output_tokens)
        fprintf(stderr, " %llu", (unsigned long long)t);
    fprintf(stderr, "\n");

    // Verify
    assert(!output_tokens.empty());
    assert(output_tokens.size() <= 5);  // 1 (prefill) + 4 (decode)

    // Step 5: Second request (verify pipeline reuse)
    std::vector<uint64_t> input2 = {42, 99};
    auto req2 = build_request_buf(input2, /*max_tokens=*/2);
    rc = testpath_inject_request(req2.data(), req2.size());
    assert(rc == 0);

    resp_len = testpath_get_response(resp_buf.data(), resp_buf.size());
    assert(resp_len > 0);
    auto out2 = parse_response_buf(resp_buf.data(),
                                    static_cast<size_t>(resp_len));
    fprintf(stderr, "[RESULT] Request 2: %zu output tokens:", out2.size());
    for (auto t : out2)
        fprintf(stderr, " %llu", (unsigned long long)t);
    fprintf(stderr, "\n");
    assert(!out2.empty());
    assert(out2.size() <= 3);

    // Step 6: Trace
    if (do_trace) {
        rc = testpath_write_trace();
        assert(rc == 0);
        fprintf(stderr, "[OK] trace written\n");
    }

    // Step 7: Stop and shutdown
    testpath_stop();
    fprintf(stderr, "[OK] testpath_stop\n");

    testpath_shutdown();
    fprintf(stderr, "[OK] testpath_shutdown\n");

    fprintf(stderr, "\n=== Phase 1: TestPath C API Integration Test PASSED ===\n");
    fprintf(stderr, "Verified:\n");
    fprintf(stderr, "  1. Wire format: serialize Request → inject → deserialize\n");
    fprintf(stderr, "  2. Full pipeline: TestPath → L4 → L3[0] prefill → L3[1] decode\n");
    fprintf(stderr, "  3. Response: serialize → get_response → parse tokens\n");
    fprintf(stderr, "  4. Pipeline reuse: second request succeeds\n");
    if (do_trace)
        fprintf(stderr, "  5. Perfetto trace generated\n");

    return 0;
}
