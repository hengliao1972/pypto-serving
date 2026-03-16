#include "kv/radix_tree.h"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace serving {

RadixTree::RadixTree() : root_(std::make_unique<RadixNode>()) {}
RadixTree::~RadixTree() = default;

// ─── Public API (thread-safe) ─────────────────────────────────────────

RadixNode* RadixTree::insert(const std::vector<TokenId>& tokens,
                             BlockHandle block) {
    std::lock_guard<std::mutex> lk(mu_);
    return insert_unlocked(tokens, block);
}

PrefixMatch RadixTree::find_prefix(const std::vector<TokenId>& tokens) const {
    std::lock_guard<std::mutex> lk(mu_);
    return find_prefix_unlocked(tokens);
}

RadixNode* RadixTree::find_exact(const std::vector<TokenId>& tokens) const {
    std::lock_guard<std::mutex> lk(mu_);
    return find_exact_unlocked(tokens);
}

int RadixTree::remove(const std::vector<TokenId>& tokens) {
    std::lock_guard<std::mutex> lk(mu_);
    if (tokens.empty()) return 0;

    // Walk to the parent of the subtree to remove.
    RadixNode* cur = root_.get();
    size_t pos = 0;
    RadixNode* parent = nullptr;
    TokenId split_key = 0;

    while (pos < tokens.size()) {
        auto it = cur->children.find(tokens[pos]);
        if (it == cur->children.end()) return 0;

        RadixNode* child = it->second.get();
        auto& edge = child->edge_tokens;

        size_t match = 0;
        while (match < edge.size() && pos + match < tokens.size() &&
               edge[match] == tokens[pos + match]) {
            match++;
        }

        if (match < edge.size()) return 0;  // partial edge match, no exact

        parent = cur;
        split_key = tokens[pos];
        pos += match;
        cur = child;
    }

    if (!parent) return 0;
    return remove_subtree(parent, split_key);
}

void RadixTree::ref(RadixNode* node) {
    std::lock_guard<std::mutex> lk(mu_);
    if (node) node->ref_count++;
}

void RadixTree::unref(RadixNode* node) {
    std::lock_guard<std::mutex> lk(mu_);
    if (node && node->ref_count > 0) node->ref_count--;
}

std::vector<RadixNode*> RadixTree::eviction_candidates() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<RadixNode*> out;
    collect_eviction(root_.get(), out);
    std::sort(out.begin(), out.end(), [](const RadixNode* a, const RadixNode* b) {
        return a->last_access_ts < b->last_access_ts;
    });
    return out;
}

int RadixTree::size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return node_count_;
}

void RadixTree::touch(RadixNode* node) {
    std::lock_guard<std::mutex> lk(mu_);
    if (node) node->last_access_ts = ++clock_;
}

// ─── Internal: insert ─────────────────────────────────────────────────

RadixNode* RadixTree::insert_unlocked(const std::vector<TokenId>& tokens,
                                       BlockHandle block) {
    RadixNode* cur = root_.get();
    size_t pos = 0;

    while (pos < tokens.size()) {
        TokenId key = tokens[pos];
        auto it = cur->children.find(key);

        if (it == cur->children.end()) {
            // No child with this first token — create a new leaf.
            auto node = std::make_unique<RadixNode>();
            node->edge_tokens.assign(tokens.begin() + pos, tokens.end());
            node->kv_block = block;
            node->last_access_ts = ++clock_;
            RadixNode* ptr = node.get();
            cur->children[key] = std::move(node);
            node_count_++;
            return ptr;
        }

        RadixNode* child = it->second.get();
        auto& edge = child->edge_tokens;

        // Match as many tokens as possible on this edge.
        size_t match = 0;
        while (match < edge.size() && pos + match < tokens.size() &&
               edge[match] == tokens[pos + match]) {
            match++;
        }

        if (match == edge.size()) {
            // Full edge consumed — continue to next node.
            pos += match;
            cur = child;
            continue;
        }

        // Partial match — split the edge.
        //   Before: cur --[A B C D]--> child
        //   After:  cur --[A B]--> split --[C D]--> child
        //                                \--[E F]--> new_leaf
        auto split = std::make_unique<RadixNode>();
        split->edge_tokens.assign(edge.begin(), edge.begin() + match);
        split->last_access_ts = ++clock_;

        // Reparent child under split with remaining edge.
        std::vector<TokenId> remainder(edge.begin() + match, edge.end());
        child->edge_tokens = std::move(remainder);
        TokenId child_key = child->edge_tokens[0];

        // Transfer ownership: cur->children[key] currently owns child.
        split->children[child_key] = std::move(cur->children[key]);
        node_count_++;  // split node added

        pos += match;
        if (pos == tokens.size()) {
            // The insertion point is at the split node itself.
            split->kv_block = block;
            RadixNode* ptr = split.get();
            cur->children[key] = std::move(split);
            return ptr;
        }

        // Create a new leaf for the remaining tokens.
        auto leaf = std::make_unique<RadixNode>();
        leaf->edge_tokens.assign(tokens.begin() + pos, tokens.end());
        leaf->kv_block = block;
        leaf->last_access_ts = ++clock_;
        RadixNode* leaf_ptr = leaf.get();
        split->children[tokens[pos]] = std::move(leaf);
        node_count_++;

        cur->children[key] = std::move(split);
        return leaf_ptr;
    }

    // All tokens consumed — cur is the target node.
    cur->kv_block = block;
    cur->last_access_ts = ++clock_;
    return cur;
}

// ─── Internal: find_prefix ────────────────────────────────────────────

PrefixMatch RadixTree::find_prefix_unlocked(
    const std::vector<TokenId>& tokens) const {
    PrefixMatch best{const_cast<RadixNode*>(root_.get()), 0};
    RadixNode* cur = root_.get();
    size_t pos = 0;

    while (pos < tokens.size()) {
        auto it = cur->children.find(tokens[pos]);
        if (it == cur->children.end()) break;

        RadixNode* child = it->second.get();
        auto& edge = child->edge_tokens;

        size_t match = 0;
        while (match < edge.size() && pos + match < tokens.size() &&
               edge[match] == tokens[pos + match]) {
            match++;
        }

        pos += match;

        if (match == edge.size()) {
            // Fully traversed this edge.
            if (child->kv_block != kInvalidBlock) {
                best = {child, static_cast<int>(pos)};
            }
            cur = child;
        } else {
            // Partial edge match — stop here.
            break;
        }
    }

    return best;
}

// ─── Internal: find_exact ─────────────────────────────────────────────

RadixNode* RadixTree::find_exact_unlocked(
    const std::vector<TokenId>& tokens) const {
    RadixNode* cur = root_.get();
    size_t pos = 0;

    while (pos < tokens.size()) {
        auto it = cur->children.find(tokens[pos]);
        if (it == cur->children.end()) return nullptr;

        RadixNode* child = it->second.get();
        auto& edge = child->edge_tokens;

        size_t match = 0;
        while (match < edge.size() && pos + match < tokens.size() &&
               edge[match] == tokens[pos + match]) {
            match++;
        }

        if (match != edge.size()) return nullptr;
        pos += match;
        cur = child;
    }

    return cur;
}

// ─── Internal: remove_subtree ─────────────────────────────────────────

static int count_nodes(const RadixNode* node) {
    int c = 1;
    for (auto& [k, child] : node->children)
        c += count_nodes(child.get());
    return c;
}

int RadixTree::remove_subtree(RadixNode* parent, TokenId first_token) {
    auto it = parent->children.find(first_token);
    if (it == parent->children.end()) return 0;
    int removed = count_nodes(it->second.get());
    parent->children.erase(it);
    node_count_ -= removed;
    return removed;
}

// ─── Internal: eviction ───────────────────────────────────────────────

void RadixTree::collect_eviction(const RadixNode* node,
                                  std::vector<RadixNode*>& out) const {
    for (auto& [k, child] : node->children) {
        if (child->children.empty() && child->ref_count == 0 &&
            child->kv_block != kInvalidBlock) {
            out.push_back(child.get());
        }
        collect_eviction(child.get(), out);
    }
}

// ─── Serialization (for lingqu_db persistence) ────────────────────────

// Wire: [n_children:u32]
//       for each child: [n_edge_tokens:u32][edge_tokens...][kv_block:i64]
//                       [last_access_ts:i64][ref_count:i32][recurse]

static void write_u32(std::vector<uint8_t>& buf, uint32_t v) {
    buf.insert(buf.end(), reinterpret_cast<uint8_t*>(&v),
               reinterpret_cast<uint8_t*>(&v) + 4);
}
static void write_i32(std::vector<uint8_t>& buf, int32_t v) {
    buf.insert(buf.end(), reinterpret_cast<uint8_t*>(&v),
               reinterpret_cast<uint8_t*>(&v) + 4);
}
static void write_i64(std::vector<uint8_t>& buf, int64_t v) {
    buf.insert(buf.end(), reinterpret_cast<uint8_t*>(&v),
               reinterpret_cast<uint8_t*>(&v) + 8);
}
static void write_u64(std::vector<uint8_t>& buf, uint64_t v) {
    buf.insert(buf.end(), reinterpret_cast<uint8_t*>(&v),
               reinterpret_cast<uint8_t*>(&v) + 8);
}

template <typename T>
static bool read_val(const uint8_t*& ptr, const uint8_t* end, T& out) {
    if (ptr + sizeof(T) > end) return false;
    std::memcpy(&out, ptr, sizeof(T));
    ptr += sizeof(T);
    return true;
}

void RadixTree::serialize_node(const RadixNode* node,
                                std::vector<uint8_t>& buf) const {
    write_u32(buf, static_cast<uint32_t>(node->children.size()));
    for (auto& [k, child] : node->children) {
        auto& edge = child->edge_tokens;
        write_u32(buf, static_cast<uint32_t>(edge.size()));
        for (auto t : edge) write_u64(buf, t);
        write_i64(buf, child->kv_block);
        write_i64(buf, child->last_access_ts);
        write_i32(buf, child->ref_count);
        serialize_node(child.get(), buf);
    }
}

RadixNode* RadixTree::deserialize_node(const uint8_t*& ptr,
                                        const uint8_t* end) {
    uint32_t n_children = 0;
    if (!read_val(ptr, end, n_children)) return nullptr;

    auto node = new RadixNode();  // caller takes ownership
    for (uint32_t i = 0; i < n_children; i++) {
        uint32_t n_edge = 0;
        if (!read_val(ptr, end, n_edge)) { delete node; return nullptr; }

        auto child = std::make_unique<RadixNode>();
        child->edge_tokens.resize(n_edge);
        for (uint32_t j = 0; j < n_edge; j++) {
            uint64_t t;
            if (!read_val(ptr, end, t)) { delete node; return nullptr; }
            child->edge_tokens[j] = t;
        }
        if (!read_val(ptr, end, child->kv_block)) { delete node; return nullptr; }
        if (!read_val(ptr, end, child->last_access_ts)) { delete node; return nullptr; }
        if (!read_val(ptr, end, child->ref_count)) { delete node; return nullptr; }

        // Recursively deserialize children.
        RadixNode* sub = deserialize_node(ptr, end);
        if (!sub) { delete node; return nullptr; }
        // Transfer sub's children to child.
        child->children = std::move(sub->children);
        delete sub;

        TokenId key = child->edge_tokens.empty() ? 0 : child->edge_tokens[0];
        node_count_++;
        node->children[key] = std::move(child);
    }
    return node;
}

std::vector<uint8_t> RadixTree::serialize() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<uint8_t> buf;
    write_i64(buf, clock_);
    serialize_node(root_.get(), buf);
    return buf;
}

bool RadixTree::deserialize(const std::vector<uint8_t>& data) {
    std::lock_guard<std::mutex> lk(mu_);
    if (data.size() < 8) return false;

    const uint8_t* ptr = data.data();
    const uint8_t* end = ptr + data.size();

    int64_t saved_clock;
    if (!read_val(ptr, end, saved_clock)) return false;

    node_count_ = 0;
    RadixNode* new_root = deserialize_node(ptr, end);
    if (!new_root) return false;

    root_.reset(new_root);
    clock_ = saved_clock;
    return true;
}

}  // namespace serving
