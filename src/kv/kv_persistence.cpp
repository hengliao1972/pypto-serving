#include "kv/kv_persistence.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace serving {

// ─── LocalFilePersistence ─────────────────────────────────────────────

LocalFilePersistence::LocalFilePersistence(const std::string& dir)
    : dir_(dir) {
    fs::create_directories(dir_);
}

LocalFilePersistence::~LocalFilePersistence() = default;

KVPersistenceBackend LocalFilePersistence::backend() {
    return {
        [this](BlockHandle h, const uint8_t* d, size_t s) {
            return write_block(h, d, s);
        },
        [this](BlockHandle h, uint8_t* b, size_t s) {
            return read_block(h, b, s);
        },
        [this](BlockHandle h) { delete_block(h); },
    };
}

void LocalFilePersistence::clear() {
    fs::remove_all(dir_);
    fs::create_directories(dir_);
}

std::string LocalFilePersistence::block_path(BlockHandle handle) const {
    return dir_ + "/kv_block_" + std::to_string(handle) + ".bin";
}

bool LocalFilePersistence::write_block(BlockHandle handle,
                                        const uint8_t* data, size_t size) {
    std::ofstream f(block_path(handle), std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(data), size);
    return f.good();
}

int64_t LocalFilePersistence::read_block(BlockHandle handle,
                                          uint8_t* buf, size_t max_size) {
    std::ifstream f(block_path(handle), std::ios::binary | std::ios::ate);
    if (!f) return -1;
    auto fsize = f.tellg();
    f.seekg(0);
    size_t to_read = std::min(static_cast<size_t>(fsize), max_size);
    f.read(reinterpret_cast<char*>(buf), to_read);
    return f.good() ? static_cast<int64_t>(to_read) : -1;
}

void LocalFilePersistence::delete_block(BlockHandle handle) {
    fs::remove(block_path(handle));
}

// ─── LocalRadixPersistence ────────────────────────────────────────────

LocalRadixPersistence::LocalRadixPersistence(const std::string& path)
    : path_(path) {}

bool LocalRadixPersistence::save(const std::vector<uint8_t>& data) {
    std::ofstream f(path_, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(data.data()), data.size());
    return f.good();
}

std::vector<uint8_t> LocalRadixPersistence::load() {
    std::ifstream f(path_, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto size = f.tellg();
    if (size <= 0) return {};
    f.seekg(0);
    std::vector<uint8_t> buf(size);
    f.read(reinterpret_cast<char*>(buf.data()), size);
    return f.good() ? buf : std::vector<uint8_t>{};
}

}  // namespace serving
