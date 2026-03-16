/*
 * test_phase8_persistence.cpp — Phase 8 persistence and stress tests.
 *
 * Validates:
 *   1. PersistenceManager: save/load Radix metadata round-trip
 *   2. KV block persistence via local file backend
 *   3. Auto-flush background thread
 *   4. Stress test: N sequential requests with Radix + KV caching
 *   5. Full-stack trace with all phases
 */

#include "engine/serving_system.h"
#include "kv/persistence_manager.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <thread>

using namespace serving;

namespace fs = std::filesystem;

static Request make_req(const std::vector<uint64_t>& tokens,
                        int max_tokens = 2) {
    Request req;
    req.token_ids = tokens;
    req.max_tokens = max_tokens;
    req.vocab_size = 32;
    req.kv_size_per_chip = 64;
    req.kv_step_size = 64;
    return req;
}

// ─── Persistence Tests ───────────────────────────────────────────────

static void test_persistence_radix_roundtrip() {
    std::string meta_path = "/tmp/pypto_test_p8_radix.bin";
    std::string kv_dir = "/tmp/pypto_test_p8_kv";

    RadixTree radix;
    KVCacheManager kv_mgr;

    PersistenceConfig pcfg;
    pcfg.radix_meta_path = meta_path;
    pcfg.kv_block_dir = kv_dir;

    PersistenceManager pm(&radix, &kv_mgr, pcfg);
    pm.wire_kv_backend();

    // Insert some data
    auto h1 = kv_mgr.alloc(CacheTier::L1, 256);
    auto h2 = kv_mgr.alloc(CacheTier::L1, 256);
    radix.insert({1, 2, 3}, h1);
    radix.insert({4, 5, 6}, h2);

    // Save
    bool ok = pm.save_radix();
    assert(ok);

    // Create new tree and load
    RadixTree radix2;
    KVCacheManager kv_mgr2;
    PersistenceManager pm2(&radix2, &kv_mgr2, pcfg);

    ok = pm2.load_radix();
    assert(ok);

    assert(radix2.find_exact({1, 2, 3}) != nullptr);
    assert(radix2.find_exact({4, 5, 6}) != nullptr);
    assert(radix2.find_exact({1, 2, 3})->kv_block == h1);

    // Cleanup
    fs::remove(meta_path);
    fs::remove_all(kv_dir);

    fprintf(stderr, "  %-50s [PASS]\n", "persistence_radix_roundtrip");
}

static void test_persistence_kv_demote_promote() {
    std::string kv_dir = "/tmp/pypto_test_p8_kv2";

    KVCacheConfig cfg;
    cfg.l1_capacity_bytes = 1024;
    cfg.l2_capacity_bytes = 1024;
    cfg.l3_capacity_bytes = 4096;
    KVCacheManager mgr(cfg);

    LocalFilePersistence persist(kv_dir);
    mgr.set_persistence_backend(persist.backend());

    auto h = mgr.alloc(CacheTier::L1, 128);
    uint8_t* p = mgr.data(h);
    for (int i = 0; i < 128; i++) p[i] = static_cast<uint8_t>(i);

    // Demote to L3 (writes to file)
    bool ok = mgr.demote(h, CacheTier::L3);
    assert(ok);
    assert(mgr.used_bytes(CacheTier::L1) == 0);
    assert(mgr.used_bytes(CacheTier::L3) == 128);

    // Promote back (reads from file)
    ok = mgr.promote(h, CacheTier::L1);
    assert(ok);
    p = mgr.data(h);
    for (int i = 0; i < 128; i++) assert(p[i] == static_cast<uint8_t>(i));

    mgr.free(h);
    persist.clear();
    fs::remove_all(kv_dir);

    fprintf(stderr, "  %-50s [PASS]\n", "persistence_kv_demote_promote");
}

static void test_auto_flush() {
    std::string meta_path = "/tmp/pypto_test_p8_auto.bin";
    std::string kv_dir = "/tmp/pypto_test_p8_auto_kv";

    RadixTree radix;
    KVCacheManager kv_mgr;

    PersistenceConfig pcfg;
    pcfg.radix_meta_path = meta_path;
    pcfg.kv_block_dir = kv_dir;
    pcfg.enable_auto_flush = true;
    pcfg.flush_interval_ms = 100;

    PersistenceManager pm(&radix, &kv_mgr, pcfg);

    radix.insert({10, 20}, 42);
    pm.start();

    // Wait for at least 2 auto-flushes
    std::this_thread::sleep_for(std::chrono::milliseconds(350));

    pm.stop();
    assert(pm.flush_count() >= 2);  // at least 2 auto + 1 final

    // Verify file was written
    assert(fs::exists(meta_path));

    fs::remove(meta_path);
    fs::remove_all(kv_dir);

    fprintf(stderr, "  %-50s [PASS] (flushes=%d)\n",
            "auto_flush", pm.flush_count());
}

// ─── Stress Test ─────────────────────────────────────────────────────

static void test_stress_sequential() {
    ServingConfig cfg;
    cfg.num_chips_per_server = 16;
    ServingSystem system(cfg);
    system.start();

    const int N = 10;
    int success = 0;
    auto t_start = std::chrono::steady_clock::now();

    for (int i = 0; i < N; i++) {
        std::vector<uint64_t> tokens;
        for (int j = 0; j <= i % 4; j++)
            tokens.push_back(static_cast<uint64_t>(100 + j + i));

        auto resp = system.infer(make_req(tokens, 1));
        if (!resp.output_tokens.empty())
            success++;
    }

    auto t_end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_end - t_start).count();

    system.stop();

    assert(success == N);
    fprintf(stderr, "  %-50s [PASS] (%d/%d, %lldms, %.1f req/s)\n",
            "stress_sequential",
            success, N, (long long)elapsed,
            N * 1000.0 / elapsed);
}

// ─── Full-stack trace ────────────────────────────────────────────────

static void test_full_trace() {
    ServingConfig cfg;
    cfg.num_chips_per_server = 16;
    cfg.enable_trace = true;
    cfg.trace_path = "phase8_full_trace.json";
    ServingSystem system(cfg);
    system.start();

    // Run several requests to populate the trace
    for (int i = 0; i < 3; i++) {
        system.infer(make_req({1, 2, 3, 4}, 1));
    }

    auto trace_path = system.write_trace();
    system.stop();

    assert(!trace_path.empty());
    assert(fs::exists(trace_path));
    auto fsize = fs::file_size(trace_path);
    assert(fsize > 100);

    fprintf(stderr, "  %-50s [PASS] (trace=%s, %zuB)\n",
            "full_trace", trace_path.c_str(), fsize);
}

// ─── Main ─────────────────────────────────────────────────────────────

int main() {
    fprintf(stderr, "=== Phase 8: Persistence & Production Hardening ===\n\n");

    fprintf(stderr, "Persistence:\n");
    test_persistence_radix_roundtrip();
    test_persistence_kv_demote_promote();
    test_auto_flush();

    fprintf(stderr, "\nStress Test:\n");
    test_stress_sequential();

    fprintf(stderr, "\nFull-Stack Trace:\n");
    test_full_trace();

    fprintf(stderr, "\n=== Phase 8: All Tests PASSED ===\n");
    return 0;
}
