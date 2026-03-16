#ifndef PYPTO_SERVING_ENGINE_STOP_CONDITION_H
#define PYPTO_SERVING_ENGINE_STOP_CONDITION_H

#include <cstdint>
#include <vector>

namespace serving {

// StopCondition — checks whether the autoregressive loop should terminate.
//
// Three termination conditions:
//   1. EOS token encountered
//   2. A user-specified stop sequence appears in the output
//   3. max_tokens budget exhausted

struct StopConfig {
    int32_t max_tokens = 256;
    int32_t eos_token  = -1;
    std::vector<std::vector<uint64_t>> stop_sequences;  // multi-token stop
};

class StopChecker {
public:
    explicit StopChecker(const StopConfig& cfg);

    // Call after each new token is generated.
    // Returns true if generation should stop.
    bool should_stop(uint64_t new_token);

    // Reset for a new request.
    void reset();

    enum class Reason { NONE, MAX_TOKENS, EOS, STOP_SEQUENCE };
    Reason stop_reason() const { return reason_; }
    int tokens_generated() const { return tokens_generated_; }

private:
    StopConfig cfg_;
    Reason reason_ = Reason::NONE;
    int tokens_generated_ = 0;
    std::vector<uint64_t> recent_tokens_;
};

}  // namespace serving

#endif
