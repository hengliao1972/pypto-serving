#ifndef PYPTO_SERVING_KV_RADIX_TREE_H
#define PYPTO_SERVING_KV_RADIX_TREE_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace serving {

// A Radix Tree for KV cache prefix sharing.
//
// Each edge carries a sequence of token IDs.  Nodes store the KV block
// handle for the prefix that ends at that node.  Prefix matching allows
// new requests to reuse cached KV data for shared prompt prefixes.
//
// Thread-safety: all public methods are guarded by an internal mutex.

using TokenId = uint64_t;
using BlockHandle = int64_t;  // opaque handle into KVCacheManager

constexpr BlockHandle kInvalidBlock = -1;

struct RadixNode {
    std::unordered_map<TokenId, std::unique_ptr<RadixNode>> children;
    std::vector<TokenId> edge_tokens;  // tokens on the incoming edge
    BlockHandle kv_block = kInvalidBlock;
    int64_t last_access_ts = 0;  // monotonic timestamp for LRU
    int ref_count = 0;           // active references (in-flight requests)
};

struct PrefixMatch {
    RadixNode* node = nullptr;
    int matched_tokens = 0;  // how many tokens from input were matched
};

class RadixTree {
public:
    RadixTree();
    ~RadixTree();

    // Insert a token sequence and associate a KV block handle with its
    // terminal node.  If the prefix already exists, only the block handle
    // is updated.  Returns the terminal node.
    RadixNode* insert(const std::vector<TokenId>& tokens, BlockHandle block);

    // Find the longest prefix match for the given token sequence.
    PrefixMatch find_prefix(const std::vector<TokenId>& tokens) const;

    // Look up the exact node for a complete token sequence.
    // Returns nullptr if not found.
    RadixNode* find_exact(const std::vector<TokenId>& tokens) const;

    // Remove a prefix and all its descendant nodes.
    // Returns the number of nodes removed.
    int remove(const std::vector<TokenId>& tokens);

    // Increment / decrement reference count on a node.
    void ref(RadixNode* node);
    void unref(RadixNode* node);

    // Collect all leaf nodes sorted by last_access_ts (ascending = oldest
    // first) that have ref_count == 0.  Used by LRU eviction.
    std::vector<RadixNode*> eviction_candidates() const;

    // Total number of nodes (excluding root).
    int size() const;

    // Update the access timestamp on a node.
    void touch(RadixNode* node);

    // Serialize metadata to a binary buffer (for lingqu_db persistence).
    std::vector<uint8_t> serialize() const;

    // Deserialize and rebuild the tree from a buffer.
    bool deserialize(const std::vector<uint8_t>& data);

private:
    mutable std::mutex mu_;
    std::unique_ptr<RadixNode> root_;
    int64_t clock_ = 0;
    int node_count_ = 0;

    // Internal helpers (caller must hold mu_).
    RadixNode* insert_unlocked(const std::vector<TokenId>& tokens,
                               BlockHandle block);
    PrefixMatch find_prefix_unlocked(const std::vector<TokenId>& tokens) const;
    RadixNode* find_exact_unlocked(const std::vector<TokenId>& tokens) const;
    int remove_subtree(RadixNode* parent, TokenId first_token);
    void collect_eviction(const RadixNode* node,
                          std::vector<RadixNode*>& out) const;
    void serialize_node(const RadixNode* node,
                        std::vector<uint8_t>& buf) const;
    RadixNode* deserialize_node(const uint8_t*& ptr, const uint8_t* end);
};

}  // namespace serving

#endif
