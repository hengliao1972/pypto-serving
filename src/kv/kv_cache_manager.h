#ifndef PYPTO_SERVING_KV_KV_CACHE_MANAGER_H
#define PYPTO_SERVING_KV_KV_CACHE_MANAGER_H

#include "kv/radix_tree.h"

#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace serving {

// Three-tier KV cache management:
//   L1: GPU VRAM (hot)   — fastest, smallest capacity
//   L2: Host Memory / lingqu_shmem (warm) — medium speed and capacity
//   L3: SSD / lingqu_block / lingqu_dfs (cold) — slow, large capacity
//
// Phase 0 implementation: all tiers use in-memory buffers.  The interface
// is designed so that L2 and L3 can later be swapped for real lingqu_shmem
// and lingqu_block backends.

enum class CacheTier : int { L1 = 1, L2 = 2, L3 = 3 };

struct KVBlock {
    BlockHandle handle;
    CacheTier tier;
    size_t size_bytes;
    std::vector<uint8_t> data;   // in-memory for Phase 0
    int64_t last_access_ts = 0;
    int ref_count = 0;
};

struct KVCacheConfig {
    size_t l1_capacity_bytes = 256 * 1024 * 1024;   // 256 MB
    size_t l2_capacity_bytes = 1024 * 1024 * 1024;   // 1 GB
    size_t l3_capacity_bytes = 4ULL * 1024 * 1024 * 1024;  // 4 GB
    size_t block_size_bytes  = 4096;                 // 4 KB per block
};

// Callbacks for persistence backends (Phase 0: no-ops or local file stubs).
struct KVPersistenceBackend {
    // Write block to persistent storage.  Returns true on success.
    std::function<bool(BlockHandle, const uint8_t*, size_t)> write_block;
    // Read block from persistent storage into buf.  Returns bytes read, or -1.
    std::function<int64_t(BlockHandle, uint8_t*, size_t)> read_block;
    // Delete block from persistent storage.
    std::function<void(BlockHandle)> delete_block;
};

class KVCacheManager {
public:
    explicit KVCacheManager(const KVCacheConfig& cfg = {});
    ~KVCacheManager();

    // Set the persistence backend for L3 tier.
    void set_persistence_backend(KVPersistenceBackend backend);

    // Allocate a new KV block in the given tier.
    // Returns the block handle, or kInvalidBlock if out of capacity.
    BlockHandle alloc(CacheTier tier, size_t size_bytes);

    // Free a block.
    void free(BlockHandle handle);

    // Get a mutable pointer to the block's data.
    // Returns nullptr if the block is not in memory (evicted to L3).
    uint8_t* data(BlockHandle handle);
    const uint8_t* data(BlockHandle handle) const;

    // Get block size.
    size_t block_size(BlockHandle handle) const;

    // Reference counting for in-flight use.
    void ref(BlockHandle handle);
    void unref(BlockHandle handle);

    // Promote a block from a lower tier to a higher tier.
    // e.g., L3 → L2 → L1.
    bool promote(BlockHandle handle, CacheTier target_tier);

    // Demote a block to a lower tier.
    // e.g., L1 → L2 → L3.
    bool demote(BlockHandle handle, CacheTier target_tier);

    // Run LRU eviction on the given tier until used bytes < capacity.
    // Returns number of blocks evicted.
    int evict(CacheTier tier);

    // Stats.
    size_t used_bytes(CacheTier tier) const;
    size_t capacity_bytes(CacheTier tier) const;
    int block_count(CacheTier tier) const;
    int total_blocks() const;

private:
    KVCacheConfig cfg_;
    KVPersistenceBackend persist_;
    mutable std::mutex mu_;
    int64_t next_handle_ = 1;
    int64_t clock_ = 0;

    std::unordered_map<BlockHandle, std::unique_ptr<KVBlock>> blocks_;

    // Per-tier tracking.
    struct TierState {
        size_t used_bytes = 0;
        size_t capacity_bytes = 0;
        std::list<BlockHandle> lru_list;  // front = oldest
        std::unordered_map<BlockHandle,
                           std::list<BlockHandle>::iterator> lru_map;
    };
    TierState tiers_[4];  // indexed by CacheTier (1,2,3)

    TierState& tier_state(CacheTier t) { return tiers_[static_cast<int>(t)]; }
    const TierState& tier_state(CacheTier t) const {
        return tiers_[static_cast<int>(t)];
    }

    void touch_lru(BlockHandle h, TierState& ts);
    void remove_lru(BlockHandle h, TierState& ts);
    void add_lru(BlockHandle h, TierState& ts);
};

}  // namespace serving

#endif
