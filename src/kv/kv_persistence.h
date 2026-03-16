#ifndef PYPTO_SERVING_KV_KV_PERSISTENCE_H
#define PYPTO_SERVING_KV_KV_PERSISTENCE_H

#include "kv/kv_cache_manager.h"

#include <string>

namespace serving {

// Phase 0 stub: KV block persistence using local files.
// Each block is stored as a file: <dir>/kv_block_<handle>.bin
//
// Future: replace with lingqu_block async I/O.

class LocalFilePersistence {
public:
    explicit LocalFilePersistence(const std::string& dir = "/tmp/pypto_kv_blocks");
    ~LocalFilePersistence();

    // Returns a KVPersistenceBackend suitable for KVCacheManager.
    KVPersistenceBackend backend();

    // Cleanup all persisted files.
    void clear();

private:
    std::string dir_;

    bool write_block(BlockHandle handle, const uint8_t* data, size_t size);
    int64_t read_block(BlockHandle handle, uint8_t* buf, size_t max_size);
    void delete_block(BlockHandle handle);
    std::string block_path(BlockHandle handle) const;
};

// Phase 0 stub: Radix tree metadata persistence using a local file.
// Stores serialized RadixTree to: <path>
//
// Future: replace with lingqu_db key-value store.

class LocalRadixPersistence {
public:
    explicit LocalRadixPersistence(
        const std::string& path = "/tmp/pypto_radix_meta.bin");

    bool save(const std::vector<uint8_t>& data);
    std::vector<uint8_t> load();

private:
    std::string path_;
};

}  // namespace serving

#endif
