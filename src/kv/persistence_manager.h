#ifndef PYPTO_SERVING_KV_PERSISTENCE_MANAGER_H
#define PYPTO_SERVING_KV_PERSISTENCE_MANAGER_H

#include "kv/kv_cache_manager.h"
#include "kv/kv_persistence.h"
#include "kv/radix_tree.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

namespace serving {

// PersistenceManager — coordinates periodic flushing of Radix metadata
// and KV cache blocks to persistent storage.
//
// Phase 0: uses local file stubs.
// Future: replace with lingqu_db (metadata) and lingqu_block (KV blocks).

struct PersistenceConfig {
    std::string radix_meta_path = "radix_meta.bin";
    std::string kv_block_dir = "kv_blocks";
    int flush_interval_ms = 5000;  // auto-flush every N ms (0 = manual only)
    bool enable_auto_flush = false;
};

class PersistenceManager {
public:
    PersistenceManager(RadixTree* radix, KVCacheManager* kv_mgr,
                        const PersistenceConfig& cfg = {});
    ~PersistenceManager();

    // Start auto-flush background thread (if enabled).
    void start();

    // Stop auto-flush.
    void stop();

    // Manual save: persist current Radix metadata.
    bool save_radix();

    // Manual load: restore Radix metadata from persistent storage.
    bool load_radix();

    // Wire up the KV persistence backend.
    void wire_kv_backend();

    // Stats.
    int flush_count() const { return flush_count_.load(); }

private:
    RadixTree* radix_;
    KVCacheManager* kv_mgr_;
    PersistenceConfig cfg_;

    LocalRadixPersistence radix_persist_;
    LocalFilePersistence kv_persist_;

    std::atomic<bool> running_{false};
    std::atomic<int> flush_count_{0};
    std::thread flush_thread_;
    std::mutex mu_;
    std::condition_variable cv_;

    void flush_loop();
};

}  // namespace serving

#endif
