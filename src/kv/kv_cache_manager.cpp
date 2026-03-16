#include "kv/kv_cache_manager.h"

#include <algorithm>
#include <cassert>

namespace serving {

KVCacheManager::KVCacheManager(const KVCacheConfig& cfg) : cfg_(cfg) {
    tiers_[1].capacity_bytes = cfg_.l1_capacity_bytes;
    tiers_[2].capacity_bytes = cfg_.l2_capacity_bytes;
    tiers_[3].capacity_bytes = cfg_.l3_capacity_bytes;

    persist_.write_block = [](BlockHandle, const uint8_t*, size_t) { return true; };
    persist_.read_block  = [](BlockHandle, uint8_t*, size_t) -> int64_t { return -1; };
    persist_.delete_block = [](BlockHandle) {};
}

KVCacheManager::~KVCacheManager() = default;

void KVCacheManager::set_persistence_backend(KVPersistenceBackend backend) {
    std::lock_guard<std::mutex> lk(mu_);
    persist_ = std::move(backend);
}

// ─── Alloc / Free ─────────────────────────────────────────────────────

BlockHandle KVCacheManager::alloc(CacheTier tier, size_t size_bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    auto& ts = tier_state(tier);

    if (ts.used_bytes + size_bytes > ts.capacity_bytes) {
        return kInvalidBlock;
    }

    BlockHandle h = next_handle_++;
    auto blk = std::make_unique<KVBlock>();
    blk->handle = h;
    blk->tier = tier;
    blk->size_bytes = size_bytes;
    blk->data.resize(size_bytes, 0);
    blk->last_access_ts = ++clock_;

    ts.used_bytes += size_bytes;
    blocks_[h] = std::move(blk);
    add_lru(h, ts);
    return h;
}

void KVCacheManager::free(BlockHandle handle) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = blocks_.find(handle);
    if (it == blocks_.end()) return;

    auto& blk = it->second;
    auto& ts = tier_state(blk->tier);
    ts.used_bytes -= blk->size_bytes;
    remove_lru(handle, ts);

    if (blk->tier == CacheTier::L3) {
        persist_.delete_block(handle);
    }

    blocks_.erase(it);
}

// ─── Data access ──────────────────────────────────────────────────────

uint8_t* KVCacheManager::data(BlockHandle handle) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = blocks_.find(handle);
    if (it == blocks_.end()) return nullptr;
    it->second->last_access_ts = ++clock_;
    touch_lru(handle, tier_state(it->second->tier));
    return it->second->data.data();
}

const uint8_t* KVCacheManager::data(BlockHandle handle) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = blocks_.find(handle);
    if (it == blocks_.end()) return nullptr;
    return it->second->data.data();
}

size_t KVCacheManager::block_size(BlockHandle handle) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = blocks_.find(handle);
    return (it != blocks_.end()) ? it->second->size_bytes : 0;
}

// ─── Reference counting ──────────────────────────────────────────────

void KVCacheManager::ref(BlockHandle handle) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = blocks_.find(handle);
    if (it != blocks_.end()) it->second->ref_count++;
}

void KVCacheManager::unref(BlockHandle handle) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = blocks_.find(handle);
    if (it != blocks_.end() && it->second->ref_count > 0)
        it->second->ref_count--;
}

// ─── Tier promotion / demotion ────────────────────────────────────────

bool KVCacheManager::promote(BlockHandle handle, CacheTier target_tier) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = blocks_.find(handle);
    if (it == blocks_.end()) return false;

    auto& blk = it->second;
    if (static_cast<int>(target_tier) >= static_cast<int>(blk->tier))
        return false;  // can only promote to a higher (numerically lower) tier

    auto& src_ts = tier_state(blk->tier);
    auto& dst_ts = tier_state(target_tier);

    if (dst_ts.used_bytes + blk->size_bytes > dst_ts.capacity_bytes)
        return false;

    src_ts.used_bytes -= blk->size_bytes;
    remove_lru(handle, src_ts);

    if (blk->tier == CacheTier::L3 && blk->data.empty()) {
        // Need to read back from persistent storage.
        blk->data.resize(blk->size_bytes);
        persist_.read_block(handle, blk->data.data(), blk->size_bytes);
    }

    blk->tier = target_tier;
    dst_ts.used_bytes += blk->size_bytes;
    add_lru(handle, dst_ts);
    blk->last_access_ts = ++clock_;
    return true;
}

bool KVCacheManager::demote(BlockHandle handle, CacheTier target_tier) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = blocks_.find(handle);
    if (it == blocks_.end()) return false;

    auto& blk = it->second;
    if (static_cast<int>(target_tier) <= static_cast<int>(blk->tier))
        return false;  // can only demote to a lower (numerically higher) tier

    auto& src_ts = tier_state(blk->tier);
    auto& dst_ts = tier_state(target_tier);

    if (dst_ts.used_bytes + blk->size_bytes > dst_ts.capacity_bytes)
        return false;

    src_ts.used_bytes -= blk->size_bytes;
    remove_lru(handle, src_ts);

    if (target_tier == CacheTier::L3) {
        persist_.write_block(handle, blk->data.data(), blk->size_bytes);
    }

    blk->tier = target_tier;
    dst_ts.used_bytes += blk->size_bytes;
    add_lru(handle, dst_ts);
    blk->last_access_ts = ++clock_;
    return true;
}

// ─── LRU eviction ─────────────────────────────────────────────────────

int KVCacheManager::evict(CacheTier tier) {
    std::lock_guard<std::mutex> lk(mu_);
    auto& ts = tier_state(tier);
    int evicted = 0;

    while (ts.used_bytes >= ts.capacity_bytes && !ts.lru_list.empty()) {
        BlockHandle victim = ts.lru_list.front();
        auto it = blocks_.find(victim);
        if (it == blocks_.end()) {
            ts.lru_list.pop_front();
            ts.lru_map.erase(victim);
            continue;
        }

        auto& blk = it->second;
        if (blk->ref_count > 0) {
            // Can't evict a referenced block — move to back and try next.
            ts.lru_list.pop_front();
            ts.lru_list.push_back(victim);
            ts.lru_map[victim] = std::prev(ts.lru_list.end());

            // Avoid infinite loop if all blocks are referenced.
            if (evicted == 0 && ts.lru_list.front() == victim)
                break;
            continue;
        }

        CacheTier next_tier = static_cast<CacheTier>(static_cast<int>(tier) + 1);
        if (static_cast<int>(next_tier) <= 3) {
            auto& dst = tier_state(next_tier);
            if (dst.used_bytes + blk->size_bytes <= dst.capacity_bytes) {
                // Demote to next tier.
                ts.used_bytes -= blk->size_bytes;
                remove_lru(victim, ts);

                if (next_tier == CacheTier::L3) {
                    persist_.write_block(victim, blk->data.data(), blk->size_bytes);
                }

                blk->tier = next_tier;
                dst.used_bytes += blk->size_bytes;
                add_lru(victim, dst);
                evicted++;
                continue;
            }
        }

        // No room in next tier — just free.
        ts.used_bytes -= blk->size_bytes;
        remove_lru(victim, ts);

        if (tier == CacheTier::L3) {
            persist_.delete_block(victim);
        }

        blocks_.erase(it);
        evicted++;
    }

    return evicted;
}

// ─── Stats ────────────────────────────────────────────────────────────

size_t KVCacheManager::used_bytes(CacheTier tier) const {
    std::lock_guard<std::mutex> lk(mu_);
    return tier_state(tier).used_bytes;
}

size_t KVCacheManager::capacity_bytes(CacheTier tier) const {
    std::lock_guard<std::mutex> lk(mu_);
    return tier_state(tier).capacity_bytes;
}

int KVCacheManager::block_count(CacheTier tier) const {
    std::lock_guard<std::mutex> lk(mu_);
    return static_cast<int>(tier_state(tier).lru_list.size());
}

int KVCacheManager::total_blocks() const {
    std::lock_guard<std::mutex> lk(mu_);
    return static_cast<int>(blocks_.size());
}

// ─── LRU helpers ──────────────────────────────────────────────────────

void KVCacheManager::touch_lru(BlockHandle h, TierState& ts) {
    auto it = ts.lru_map.find(h);
    if (it != ts.lru_map.end()) {
        ts.lru_list.erase(it->second);
        ts.lru_list.push_back(h);
        it->second = std::prev(ts.lru_list.end());
    }
}

void KVCacheManager::remove_lru(BlockHandle h, TierState& ts) {
    auto it = ts.lru_map.find(h);
    if (it != ts.lru_map.end()) {
        ts.lru_list.erase(it->second);
        ts.lru_map.erase(it);
    }
}

void KVCacheManager::add_lru(BlockHandle h, TierState& ts) {
    ts.lru_list.push_back(h);
    ts.lru_map[h] = std::prev(ts.lru_list.end());
}

}  // namespace serving
