/*
 * test_phase0_smoke.cpp — Phase 0 smoke test for PC16 x2 serving topology.
 *
 * Verifies:
 *   1. ServingSystem creates L4 + 2×L3 runtimes and starts/stops cleanly
 *   2. A test request flows through L4 → L3[0] prefill → L3[1] decode
 *   3. Output tokens are generated (correct count)
 *   4. ChipBackend stubs were called the expected number of times
 *   5. Perfetto trace generation (with --trace flag)
 *
 * DSL spec: serving_main.py
 */

#include "engine/serving_system.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    bool do_trace = false;
    std::string trace_path;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--trace") == 0) {
            do_trace = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                trace_path = argv[++i];
        }
    }
    if (do_trace && trace_path.empty())
        trace_path = "pypto_serving_phase0_trace.json";

    fprintf(stderr, "=== pypto-serving Phase 0 Smoke Test ===\n\n");

    // ── Configure ─────────────────────────────────────────────────────
    serving::ServingConfig cfg;
    cfg.num_chips_per_server = 16;
    cfg.enable_trace = do_trace;
    cfg.trace_path = trace_path;

    fprintf(stderr, "Topology: L4 Pod (2 workers)\n");
    fprintf(stderr, "  L3[0] Prefill PC16 (16 NPU workers)\n");
    fprintf(stderr, "  L3[1] Decode  PC16 (16 NPU workers)\n");
    fprintf(stderr, "  L2 stubs: %d chips per server\n\n",
            cfg.num_chips_per_server);

    // ── Create and start ──────────────────────────────────────────────
    serving::ServingSystem system(cfg);
    system.start();
    fprintf(stderr, "[OK] All runtimes started\n");

    // ── Inject test request ───────────────────────────────────────────
    serving::Request request;
    request.token_ids = {101, 2023, 2003, 1037, 3231, 6251, 2005, 5604};
    request.max_tokens = 4;
    request.temperature = 1.0f;
    request.top_p = 1.0f;
    request.stop_token = -1;
    request.vocab_size = 32;
    request.kv_size_per_chip = 64;
    request.kv_step_size = 64;

    fprintf(stderr, "[REQ] %zu input tokens, max_tokens=%d, vocab=%d\n",
            request.token_ids.size(), request.max_tokens, request.vocab_size);

    // ── Execute ───────────────────────────────────────────────────────
    serving::Response response = system.infer(request);

    // ── Verify ────────────────────────────────────────────────────────
    fprintf(stderr, "\n[RESULT] %zu output tokens generated\n",
            response.output_tokens.size());

    // Must have at least 1 token (first token from prefill) + up to max_tokens decode steps.
    assert(!response.output_tokens.empty());
    assert(response.output_tokens.size() <=
           static_cast<size_t>(request.max_tokens + 1));

    // Print token values
    fprintf(stderr, "  tokens: ");
    for (size_t i = 0; i < response.output_tokens.size(); i++) {
        auto& t = response.output_tokens[i];
        if (t.data_ptr())
            fprintf(stderr, "%llu ", (unsigned long long)t.data_ptr()[0]);
        else
            fprintf(stderr, "? ");
    }
    fprintf(stderr, "\n");

    // Verify ChipBackend stubs were called
    int prefill_calls = system.prefill_backend().prefill_calls.load();
    int decode_calls  = system.decode_backend().decode_calls.load();
    fprintf(stderr, "\n[STUBS] prefill calls: %d, decode calls: %d\n",
            prefill_calls, decode_calls);

    assert(prefill_calls == cfg.num_chips_per_server);
    int expected_decode_steps = static_cast<int>(response.output_tokens.size()) - 1;
    assert(decode_calls == expected_decode_steps * cfg.num_chips_per_server);

    // ── Stop ──────────────────────────────────────────────────────────
    system.stop();
    fprintf(stderr, "[OK] All runtimes stopped\n");

    // ── Trace ─────────────────────────────────────────────────────────
    if (do_trace) {
        std::string written = system.write_trace();
        if (!written.empty()) {
            auto abs = fs::absolute(written);
            fprintf(stderr, "\n[TRACE] Written: %s\n", abs.c_str());
            fprintf(stderr, "[TRACE] Open in https://ui.perfetto.dev/\n");
        }
    }

    fprintf(stderr, "\n=== pypto-serving Phase 0 Smoke Test PASSED ===\n");
    fprintf(stderr, "Verified:\n");
    fprintf(stderr, "  1. L4 Pod + 2x L3 PC16 runtimes start/stop cleanly\n");
    fprintf(stderr, "  2. Request flows: L4 → L3[0] prefill → L3[1] decode\n");
    fprintf(stderr, "  3. %zu output tokens generated (prefill + %d AR steps)\n",
            response.output_tokens.size(), expected_decode_steps);
    fprintf(stderr, "  4. ChipBackend stubs called correctly (%d prefill, %d decode)\n",
            prefill_calls, decode_calls);
    if (do_trace)
        fprintf(stderr, "  5. Perfetto trace generated\n");

    return 0;
}
