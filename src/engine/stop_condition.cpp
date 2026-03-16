#include "engine/stop_condition.h"

#include <algorithm>

namespace serving {

StopChecker::StopChecker(const StopConfig& cfg) : cfg_(cfg) {
    size_t max_seq = 0;
    for (auto& seq : cfg_.stop_sequences)
        max_seq = std::max(max_seq, seq.size());
    recent_tokens_.reserve(max_seq + 1);
}

void StopChecker::reset() {
    reason_ = Reason::NONE;
    tokens_generated_ = 0;
    recent_tokens_.clear();
}

bool StopChecker::should_stop(uint64_t new_token) {
    tokens_generated_++;

    // 1. Max tokens
    if (tokens_generated_ >= cfg_.max_tokens) {
        reason_ = Reason::MAX_TOKENS;
        return true;
    }

    // 2. EOS
    if (cfg_.eos_token >= 0 &&
        new_token == static_cast<uint64_t>(cfg_.eos_token)) {
        reason_ = Reason::EOS;
        return true;
    }

    // 3. Stop sequences
    if (!cfg_.stop_sequences.empty()) {
        recent_tokens_.push_back(new_token);
        for (auto& seq : cfg_.stop_sequences) {
            if (recent_tokens_.size() >= seq.size()) {
                bool match = true;
                size_t offset = recent_tokens_.size() - seq.size();
                for (size_t i = 0; i < seq.size(); i++) {
                    if (recent_tokens_[offset + i] != seq[i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    reason_ = Reason::STOP_SEQUENCE;
                    return true;
                }
            }
        }
    }

    return false;
}

}  // namespace serving
