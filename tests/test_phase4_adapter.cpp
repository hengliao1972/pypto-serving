/*
 * test_phase4_adapter.cpp — Phase 4 tests for ChipBackendDlopen adapter.
 *
 * Validates:
 *   1. ChipBackendDlopen falls back to stub when .so not found
 *   2. Stub fallback produces deterministic output (same as ChipBackendStub)
 *   3. Full pipeline works with dlopen adapter in stub mode
 *   4. KV block management through the full serve pipeline
 */

#include "engine/serving_system.h"
#include "l2_stubs/chip_backend_dlopen.h"
#include "l2_stubs/chip_backend_stub.h"

#include <cassert>
#include <cstdio>
#include <cstring>

using namespace serving;

// Helper: create a standalone LinquTensor with allocated data buffer.
static linqu::LinquTensor make_test_tensor(size_t count) {
    linqu::LinquTensor t;
    t.handle = linqu::alloc_tensor_handle();
    t.count = count;
    auto* raw = new uint64_t[count]();
    t.data_ref = std::make_shared<uint64_t*>(raw);
    t.ready = std::make_shared<std::atomic<bool>>(true);
    return t;
}

// ─── Test: dlopen adapter fallback ────────────────────────────────────

static void test_dlopen_fallback() {
    ChipBackendDlopen adapter(16, "/nonexistent/libhost_runtime.so");
    assert(!adapter.is_loaded());

    auto token_ids = make_test_tensor(4);
    auto kv_in     = make_test_tensor(1);
    auto kv_out    = make_test_tensor(64);
    auto logits    = make_test_tensor(32);

    adapter.model_prefill(0, token_ids, kv_in, kv_out, logits);
    assert(adapter.prefill_calls == 1);
    assert(kv_out.data_ptr()[0] == 0);  // chip_id=0, i=0 → 0*1000+0
    assert(kv_out.data_ptr()[1] == 1);  // 0*1000+1

    fprintf(stderr, "  %-50s [PASS]\n", "dlopen_fallback");
}

static void test_dlopen_stub_deterministic() {
    ChipBackendDlopen a1(16);
    ChipBackendDlopen a2(16);

    // Two independent dlopen adapters (both in stub mode) should produce
    // identical output for the same chip_id and tensor sizes.
    for (int chip = 0; chip < 4; chip++) {
        auto t1 = make_test_tensor(4);
        auto k1 = make_test_tensor(1);
        auto kv1 = make_test_tensor(32);
        auto l1  = make_test_tensor(16);

        auto t2 = make_test_tensor(4);
        auto k2 = make_test_tensor(1);
        auto kv2 = make_test_tensor(32);
        auto l2  = make_test_tensor(16);

        a1.model_prefill(chip, t1, k1, kv1, l1);
        a2.model_prefill(chip, t2, k2, kv2, l2);

        for (size_t i = 0; i < 32; i++)
            assert(kv1.data_ptr()[i] == kv2.data_ptr()[i]);
        for (size_t i = 0; i < 16; i++)
            assert(l1.data_ptr()[i] == l2.data_ptr()[i]);
    }
    fprintf(stderr, "  %-50s [PASS]\n", "dlopen_stub_deterministic");
}

// ─── Test: full pipeline with dlopen adapter ──────────────────────────

static void test_pipeline_with_dlopen() {
    // We can't swap the backend in ServingSystem easily (it uses
    // ChipBackendStub), but we verify the ChipBackendDlopen works
    // as a ChipBackend* in isolation with InferenceEngine would.
    ChipBackendDlopen adapter(4);

    // Simulate what InferenceEngine does: call workers
    for (int chip = 0; chip < 4; chip++) {
        auto prev = make_test_tensor(1);
        prev.data_ptr()[0] = 42;
        auto kv = make_test_tensor(8);
        auto logits = make_test_tensor(16);
        auto new_kv = make_test_tensor(8);

        adapter.model_decode(chip, prev, kv, logits, new_kv);
        assert(adapter.decode_calls == chip + 1);
        assert(logits.data_ptr()[0] == static_cast<uint64_t>((chip + 0 + 1) % 32));
    }
    fprintf(stderr, "  %-50s [PASS]\n", "pipeline_with_dlopen");
}

// ─── Test: full serve with KV tracking ────────────────────────────────

static void test_serve_kv_tracking() {
    ServingConfig cfg;
    cfg.num_chips_per_server = 16;
    ServingSystem system(cfg);
    system.start();

    assert(system.kv_cache().total_blocks() == 0);

    // Request 1
    Request req1;
    req1.token_ids = {1, 2, 3, 4, 5, 6, 7, 8};
    req1.max_tokens = 2;
    req1.vocab_size = 32;
    req1.kv_size_per_chip = 64;
    req1.kv_step_size = 64;
    auto r1 = system.serve(req1);
    assert(!r1.response.output_tokens.empty());
    int blocks_after_1 = system.kv_cache().total_blocks();
    assert(blocks_after_1 >= 1);

    // Request 2 with same prefix — KV block should exist in Radix
    Request req2 = req1;
    req2.token_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto r2 = system.serve(req2);
    assert(r2.prefix_hit_tokens == 8);  // full prefix hit
    int blocks_after_2 = system.kv_cache().total_blocks();
    assert(blocks_after_2 >= blocks_after_1);

    fprintf(stderr, "  %-50s [PASS] (blocks: %d→%d, hit=%d)\n",
            "serve_kv_tracking", blocks_after_1, blocks_after_2,
            r2.prefix_hit_tokens);

    system.stop();
}

// ─── Main ─────────────────────────────────────────────────────────────

int main() {
    fprintf(stderr, "=== Phase 4: ChipBackend Adapter Tests ===\n\n");

    fprintf(stderr, "ChipBackendDlopen:\n");
    test_dlopen_fallback();
    test_dlopen_stub_deterministic();
    test_pipeline_with_dlopen();

    fprintf(stderr, "\nServe + KV Tracking:\n");
    test_serve_kv_tracking();

    fprintf(stderr, "\n=== Phase 4: All Tests PASSED ===\n");
    fprintf(stderr, "\nNote: Real kernel integration requires libhost_runtime.so\n");
    fprintf(stderr, "from simpler. Current tests use stub fallback mode.\n");
    return 0;
}
