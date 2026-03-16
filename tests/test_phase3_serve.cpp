/*
 * test_phase3_serve.cpp — Phase 3 integration test.
 *
 * Validates:
 *   1. serve_request with Radix Tree + KV Cache integration
 *   2. Prefix sharing: second request with shared prefix gets cache hit
 *   3. Stop conditions: EOS, max_tokens, stop_sequence
 *   4. KV Cache allocation/reference counting
 *   5. Full pipeline: TestPath → Radix → Prefill → AR → Radix update
 */

#include "engine/serving_system.h"
#include "engine/stop_condition.h"
#include "kv/kv_cache_manager.h"
#include "kv/radix_tree.h"

#include <cassert>
#include <cstdio>
#include <cstring>

using namespace serving;

// ─── Test helpers ─────────────────────────────────────────────────────

static Request make_request(const std::vector<uint64_t>& tokens,
                            int32_t max_tokens = 4,
                            int32_t stop_token = -1) {
    Request req;
    req.token_ids = tokens;
    req.max_tokens = max_tokens;
    req.stop_token = stop_token;
    req.vocab_size = 32;
    req.kv_size_per_chip = 64;
    req.kv_step_size = 64;
    return req;
}

// ─── Test: StopChecker ────────────────────────────────────────────────

static void test_stop_max_tokens() {
    StopConfig cfg;
    cfg.max_tokens = 3;
    StopChecker checker(cfg);

    assert(!checker.should_stop(10));
    assert(!checker.should_stop(20));
    assert(checker.should_stop(30));
    assert(checker.stop_reason() == StopChecker::Reason::MAX_TOKENS);
    assert(checker.tokens_generated() == 3);
    fprintf(stderr, "  %-50s [PASS]\n", "stop_max_tokens");
}

static void test_stop_eos() {
    StopConfig cfg;
    cfg.max_tokens = 100;
    cfg.eos_token = 0;
    StopChecker checker(cfg);

    assert(!checker.should_stop(42));
    assert(!checker.should_stop(7));
    assert(checker.should_stop(0));
    assert(checker.stop_reason() == StopChecker::Reason::EOS);
    fprintf(stderr, "  %-50s [PASS]\n", "stop_eos");
}

static void test_stop_sequence() {
    StopConfig cfg;
    cfg.max_tokens = 100;
    cfg.stop_sequences = {{10, 20, 30}};
    StopChecker checker(cfg);

    assert(!checker.should_stop(5));
    assert(!checker.should_stop(10));
    assert(!checker.should_stop(20));
    assert(checker.should_stop(30));
    assert(checker.stop_reason() == StopChecker::Reason::STOP_SEQUENCE);
    fprintf(stderr, "  %-50s [PASS]\n", "stop_sequence");
}

static void test_stop_reset() {
    StopConfig cfg;
    cfg.max_tokens = 2;
    StopChecker checker(cfg);

    assert(!checker.should_stop(1));
    assert(checker.should_stop(2));

    checker.reset();
    assert(!checker.should_stop(1));
    assert(checker.should_stop(2));
    fprintf(stderr, "  %-50s [PASS]\n", "stop_reset");
}

// ─── Test: serve_request integration ──────────────────────────────────

static void test_serve_basic() {
    ServingConfig cfg;
    cfg.num_chips_per_server = 16;
    ServingSystem system(cfg);
    system.start();

    Request req = make_request({101, 2023, 2003, 1037}, 4);
    ServeResult result = system.serve(req);

    assert(!result.response.output_tokens.empty());
    assert(result.total_tokens_generated > 0);
    fprintf(stderr, "  %-50s [PASS] (%d tokens, prefix_hit=%d)\n",
            "serve_basic",
            result.total_tokens_generated,
            result.prefix_hit_tokens);

    system.stop();
}

static void test_serve_prefix_sharing() {
    ServingConfig cfg;
    cfg.num_chips_per_server = 16;
    ServingSystem system(cfg);
    system.start();

    // First request
    Request req1 = make_request({101, 2023, 2003, 1037}, 2);
    ServeResult r1 = system.serve(req1);
    assert(r1.prefix_hit_tokens == 0);  // first time, no cache

    // Second request with shared prefix
    Request req2 = make_request({101, 2023, 2003, 1037, 5604}, 2);
    ServeResult r2 = system.serve(req2);
    assert(r2.prefix_hit_tokens == 4);  // should hit 4-token prefix

    // Verify Radix tree has both entries
    assert(system.radix_tree().find_exact({101, 2023, 2003, 1037}) != nullptr);
    assert(system.radix_tree().find_exact({101, 2023, 2003, 1037, 5604}) != nullptr);

    fprintf(stderr, "  %-50s [PASS] (hit=%d tokens)\n",
            "serve_prefix_sharing", r2.prefix_hit_tokens);

    system.stop();
}

static void test_serve_kv_allocation() {
    ServingConfig cfg;
    cfg.num_chips_per_server = 16;
    ServingSystem system(cfg);
    system.start();

    assert(system.kv_cache().total_blocks() == 0);

    Request req = make_request({42, 99}, 1);
    system.serve(req);

    assert(system.kv_cache().total_blocks() >= 1);
    fprintf(stderr, "  %-50s [PASS] (blocks=%d)\n",
            "serve_kv_allocation",
            system.kv_cache().total_blocks());

    system.stop();
}

static void test_serve_with_trace() {
    ServingConfig cfg;
    cfg.num_chips_per_server = 16;
    cfg.enable_trace = true;
    cfg.trace_path = "phase3_serve_trace.json";
    ServingSystem system(cfg);
    system.start();

    Request req = make_request({1, 2, 3, 4, 5}, 3);
    ServeResult result = system.serve(req);

    assert(!result.response.output_tokens.empty());

    std::string trace_path = system.write_trace();
    assert(!trace_path.empty());
    fprintf(stderr, "  %-50s [PASS] (trace=%s)\n",
            "serve_with_trace", trace_path.c_str());

    system.stop();
}

// ─── Main ─────────────────────────────────────────────────────────────

int main() {
    fprintf(stderr, "=== Phase 3: Serve Request Integration Tests ===\n\n");

    fprintf(stderr, "StopChecker:\n");
    test_stop_max_tokens();
    test_stop_eos();
    test_stop_sequence();
    test_stop_reset();

    fprintf(stderr, "\nServe Request (Radix + KV + Pipeline):\n");
    test_serve_basic();
    test_serve_prefix_sharing();
    test_serve_kv_allocation();
    test_serve_with_trace();

    fprintf(stderr, "\n=== Phase 3: All Tests PASSED ===\n");
    return 0;
}
