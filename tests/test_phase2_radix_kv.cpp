/*
 * test_phase2_radix_kv.cpp — Phase 2 unit tests for RadixTree and KVCacheManager.
 *
 * Tests:
 *   1. Radix: insert / find_prefix / find_exact
 *   2. Radix: prefix sharing (branch/split)
 *   3. Radix: remove subtree
 *   4. Radix: serialize / deserialize round-trip
 *   5. KV: alloc / free / data access
 *   6. KV: promote / demote between tiers
 *   7. KV: LRU eviction (L1 → L2 → L3)
 *   8. KV: ref_count prevents eviction
 *   9. Integration: Radix + KV together
 *  10. Persistence: local file write/read round-trip
 */

#include "kv/kv_cache_manager.h"
#include "kv/kv_persistence.h"
#include "kv/radix_tree.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>

using namespace serving;

#define TEST(name) static void test_##name()
#define RUN(name) do { \
    fprintf(stderr, "  %-50s", #name); \
    test_##name(); \
    fprintf(stderr, " [PASS]\n"); \
} while(0)

// ─── Radix Tree Tests ─────────────────────────────────────────────────

TEST(radix_insert_find) {
    RadixTree tree;
    std::vector<TokenId> seq = {1, 2, 3, 4, 5};

    auto* node = tree.insert(seq, 100);
    assert(node != nullptr);
    assert(node->kv_block == 100);
    assert(tree.size() == 1);

    auto match = tree.find_prefix(seq);
    assert(match.matched_tokens == 5);
    assert(match.node == node);

    auto* exact = tree.find_exact(seq);
    assert(exact == node);

    auto* miss = tree.find_exact({1, 2, 3});
    assert(miss == nullptr);  // no block assigned at partial prefix
}

TEST(radix_prefix_sharing) {
    RadixTree tree;
    tree.insert({1, 2, 3, 4}, 10);
    tree.insert({1, 2, 5, 6}, 20);

    assert(tree.size() >= 3);  // at least: split at (1,2), two leaves

    auto m1 = tree.find_prefix({1, 2, 3, 4, 7, 8});
    assert(m1.matched_tokens == 4);

    auto m2 = tree.find_prefix({1, 2, 5, 6, 9});
    assert(m2.matched_tokens == 4);

    auto m3 = tree.find_prefix({1, 2, 9, 9});
    assert(m3.matched_tokens == 0);  // root has no block
}

TEST(radix_prefix_extension) {
    RadixTree tree;
    tree.insert({10, 20, 30}, 100);
    tree.insert({10, 20, 30, 40, 50}, 200);

    auto* short_node = tree.find_exact({10, 20, 30});
    assert(short_node != nullptr);
    assert(short_node->kv_block == 100);

    auto* long_node = tree.find_exact({10, 20, 30, 40, 50});
    assert(long_node != nullptr);
    assert(long_node->kv_block == 200);

    auto m = tree.find_prefix({10, 20, 30, 40});
    assert(m.matched_tokens == 3);  // longest prefix with a block is (10,20,30)
}

TEST(radix_remove) {
    RadixTree tree;
    tree.insert({1, 2, 3}, 10);
    tree.insert({1, 2, 4}, 20);
    tree.insert({1, 2, 3, 5, 6}, 30);

    int removed = tree.remove({1, 2, 3});
    assert(removed >= 1);

    assert(tree.find_exact({1, 2, 3}) == nullptr);
    assert(tree.find_exact({1, 2, 3, 5, 6}) == nullptr);

    auto* still_there = tree.find_exact({1, 2, 4});
    assert(still_there != nullptr);
    assert(still_there->kv_block == 20);
}

TEST(radix_serialize_roundtrip) {
    RadixTree tree;
    tree.insert({1, 2, 3}, 10);
    tree.insert({1, 2, 4}, 20);
    tree.insert({5, 6}, 30);

    auto buf = tree.serialize();
    assert(!buf.empty());

    RadixTree tree2;
    bool ok = tree2.deserialize(buf);
    assert(ok);

    assert(tree2.find_exact({1, 2, 3})->kv_block == 10);
    assert(tree2.find_exact({1, 2, 4})->kv_block == 20);
    assert(tree2.find_exact({5, 6})->kv_block == 30);
}

TEST(radix_eviction_candidates) {
    RadixTree tree;
    auto* n1 = tree.insert({1, 2}, 10);
    auto* n2 = tree.insert({3, 4}, 20);

    tree.ref(n1);  // n1 is in use

    auto candidates = tree.eviction_candidates();
    assert(candidates.size() >= 1);
    // n2 should be a candidate, n1 should not
    bool found_n2 = false;
    for (auto* c : candidates) {
        assert(c != n1);
        if (c == n2) found_n2 = true;
    }
    assert(found_n2);

    tree.unref(n1);
    candidates = tree.eviction_candidates();
    // Now both should be candidates
    assert(candidates.size() >= 2);
}

// ─── KV Cache Manager Tests ──────────────────────────────────────────

TEST(kv_alloc_free) {
    KVCacheConfig cfg;
    cfg.l1_capacity_bytes = 1024;
    KVCacheManager mgr(cfg);

    auto h = mgr.alloc(CacheTier::L1, 256);
    assert(h != kInvalidBlock);
    assert(mgr.used_bytes(CacheTier::L1) == 256);
    assert(mgr.block_count(CacheTier::L1) == 1);

    uint8_t* p = mgr.data(h);
    assert(p != nullptr);
    p[0] = 42;
    assert(mgr.data(h)[0] == 42);

    mgr.free(h);
    assert(mgr.used_bytes(CacheTier::L1) == 0);
    assert(mgr.total_blocks() == 0);
}

TEST(kv_capacity_limit) {
    KVCacheConfig cfg;
    cfg.l1_capacity_bytes = 100;
    KVCacheManager mgr(cfg);

    auto h1 = mgr.alloc(CacheTier::L1, 60);
    assert(h1 != kInvalidBlock);

    auto h2 = mgr.alloc(CacheTier::L1, 60);
    assert(h2 == kInvalidBlock);  // over capacity

    auto h3 = mgr.alloc(CacheTier::L1, 40);
    assert(h3 != kInvalidBlock);  // fits
}

TEST(kv_promote_demote) {
    KVCacheConfig cfg;
    cfg.l1_capacity_bytes = 1024;
    cfg.l2_capacity_bytes = 1024;
    cfg.l3_capacity_bytes = 1024;
    KVCacheManager mgr(cfg);

    auto h = mgr.alloc(CacheTier::L2, 100);
    assert(h != kInvalidBlock);

    uint8_t* p = mgr.data(h);
    std::memset(p, 0xAB, 100);

    bool ok = mgr.promote(h, CacheTier::L1);
    assert(ok);
    assert(mgr.used_bytes(CacheTier::L1) == 100);
    assert(mgr.used_bytes(CacheTier::L2) == 0);
    assert(mgr.data(h)[0] == 0xAB);

    ok = mgr.demote(h, CacheTier::L3);
    assert(ok);
    assert(mgr.used_bytes(CacheTier::L1) == 0);
    assert(mgr.used_bytes(CacheTier::L3) == 100);
}

TEST(kv_lru_eviction) {
    KVCacheConfig cfg;
    cfg.l1_capacity_bytes = 200;
    cfg.l2_capacity_bytes = 400;
    cfg.l3_capacity_bytes = 10000;
    KVCacheManager mgr(cfg);

    // Fill L1 to capacity
    auto h1 = mgr.alloc(CacheTier::L1, 100);
    auto h2 = mgr.alloc(CacheTier::L1, 100);
    assert(h1 != kInvalidBlock);
    assert(h2 != kInvalidBlock);

    // Access h2 to make h1 the LRU victim
    mgr.data(h2);

    // Alloc one more — triggers eviction need manually
    // Force capacity down by allocating another
    auto h3 = mgr.alloc(CacheTier::L1, 100);
    assert(h3 == kInvalidBlock);  // full

    // Evict — should demote h1 to L2
    int evicted = mgr.evict(CacheTier::L1);
    (void)evicted;

    // Now L1 should have space
    h3 = mgr.alloc(CacheTier::L1, 100);
    assert(h3 != kInvalidBlock);
    assert(mgr.used_bytes(CacheTier::L2) > 0);
}

TEST(kv_ref_prevents_eviction) {
    KVCacheConfig cfg;
    cfg.l1_capacity_bytes = 100;
    cfg.l2_capacity_bytes = 1000;
    KVCacheManager mgr(cfg);

    auto h = mgr.alloc(CacheTier::L1, 100);
    mgr.ref(h);

    // Try to evict — should fail because block is referenced
    int evicted = mgr.evict(CacheTier::L1);
    assert(evicted == 0);
    assert(mgr.used_bytes(CacheTier::L1) == 100);

    mgr.unref(h);
    evicted = mgr.evict(CacheTier::L1);
    assert(evicted == 1);
    assert(mgr.used_bytes(CacheTier::L1) == 0);
}

// ─── Integration: Radix + KV ─────────────────────────────────────────

TEST(integration_radix_kv) {
    RadixTree tree;
    KVCacheConfig cfg;
    cfg.l1_capacity_bytes = 4096;
    KVCacheManager mgr(cfg);

    std::vector<TokenId> prompt = {101, 2023, 2003, 1037};

    // Check prefix — initially no match
    auto match = tree.find_prefix(prompt);
    assert(match.matched_tokens == 0);

    // Compute KV and store
    auto block = mgr.alloc(CacheTier::L1, 512);
    std::memset(mgr.data(block), 0xFF, 512);

    tree.insert(prompt, block);

    // New request with shared prefix
    std::vector<TokenId> new_prompt = {101, 2023, 2003, 1037, 5604, 999};
    match = tree.find_prefix(new_prompt);
    assert(match.matched_tokens == 4);
    assert(match.node->kv_block == block);

    // Reuse cached KV data
    const uint8_t* cached = mgr.data(match.node->kv_block);
    assert(cached[0] == 0xFF);
}

// ─── Persistence Tests ───────────────────────────────────────────────

TEST(persistence_kv_file) {
    std::string dir = "/tmp/pypto_test_kv_persist";
    LocalFilePersistence persist(dir);

    KVCacheConfig cfg;
    cfg.l1_capacity_bytes = 1024;
    cfg.l2_capacity_bytes = 1024;
    cfg.l3_capacity_bytes = 4096;
    KVCacheManager mgr(cfg);
    mgr.set_persistence_backend(persist.backend());

    auto h = mgr.alloc(CacheTier::L1, 256);
    uint8_t* p = mgr.data(h);
    for (int i = 0; i < 256; i++) p[i] = static_cast<uint8_t>(i);

    // Demote to L3 (writes to file)
    bool ok = mgr.demote(h, CacheTier::L3);
    assert(ok);

    // Promote back to L1 (reads from file)
    ok = mgr.promote(h, CacheTier::L1);
    assert(ok);
    p = mgr.data(h);
    for (int i = 0; i < 256; i++) assert(p[i] == static_cast<uint8_t>(i));

    mgr.free(h);
    persist.clear();
    std::filesystem::remove_all(dir);
}

TEST(persistence_radix_file) {
    std::string path = "/tmp/pypto_test_radix_meta.bin";
    LocalRadixPersistence persist(path);

    RadixTree tree;
    tree.insert({1, 2, 3}, 10);
    tree.insert({4, 5}, 20);

    auto buf = tree.serialize();
    bool ok = persist.save(buf);
    assert(ok);

    auto loaded = persist.load();
    assert(!loaded.empty());

    RadixTree tree2;
    ok = tree2.deserialize(loaded);
    assert(ok);
    assert(tree2.find_exact({1, 2, 3})->kv_block == 10);
    assert(tree2.find_exact({4, 5})->kv_block == 20);

    std::filesystem::remove(path);
}

// ─── Main ─────────────────────────────────────────────────────────────

int main() {
    fprintf(stderr, "=== Phase 2: Radix Tree & KV Cache Unit Tests ===\n\n");

    fprintf(stderr, "Radix Tree:\n");
    RUN(radix_insert_find);
    RUN(radix_prefix_sharing);
    RUN(radix_prefix_extension);
    RUN(radix_remove);
    RUN(radix_serialize_roundtrip);
    RUN(radix_eviction_candidates);

    fprintf(stderr, "\nKV Cache Manager:\n");
    RUN(kv_alloc_free);
    RUN(kv_capacity_limit);
    RUN(kv_promote_demote);
    RUN(kv_lru_eviction);
    RUN(kv_ref_prevents_eviction);

    fprintf(stderr, "\nIntegration:\n");
    RUN(integration_radix_kv);

    fprintf(stderr, "\nPersistence:\n");
    RUN(persistence_kv_file);
    RUN(persistence_radix_file);

    fprintf(stderr, "\n=== Phase 2: All Tests PASSED ===\n");
    return 0;
}
