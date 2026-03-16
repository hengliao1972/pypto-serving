#include "kv/persistence_manager.h"

#include <chrono>
#include <cstdio>

namespace serving {

PersistenceManager::PersistenceManager(RadixTree* radix,
                                         KVCacheManager* kv_mgr,
                                         const PersistenceConfig& cfg)
    : radix_(radix),
      kv_mgr_(kv_mgr),
      cfg_(cfg),
      radix_persist_(cfg.radix_meta_path),
      kv_persist_(cfg.kv_block_dir) {}

PersistenceManager::~PersistenceManager() {
    stop();
}

void PersistenceManager::start() {
    wire_kv_backend();

    if (cfg_.enable_auto_flush && cfg_.flush_interval_ms > 0) {
        running_ = true;
        flush_thread_ = std::thread(&PersistenceManager::flush_loop, this);
    }
}

void PersistenceManager::stop() {
    if (running_) {
        running_ = false;
        cv_.notify_all();
        if (flush_thread_.joinable())
            flush_thread_.join();
    }
    // Final flush
    save_radix();
}

bool PersistenceManager::save_radix() {
    auto buf = radix_->serialize();
    bool ok = radix_persist_.save(buf);
    if (ok) flush_count_++;
    return ok;
}

bool PersistenceManager::load_radix() {
    auto buf = radix_persist_.load();
    if (buf.empty()) return false;
    return radix_->deserialize(buf);
}

void PersistenceManager::wire_kv_backend() {
    kv_mgr_->set_persistence_backend(kv_persist_.backend());
}

void PersistenceManager::flush_loop() {
    while (running_) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait_for(lk, std::chrono::milliseconds(cfg_.flush_interval_ms),
                     [this] { return !running_.load(); });

        if (!running_) break;
        save_radix();
    }
}

}  // namespace serving
