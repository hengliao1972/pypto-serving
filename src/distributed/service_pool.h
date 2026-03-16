#ifndef PYPTO_SERVING_DISTRIBUTED_SERVICE_POOL_H
#define PYPTO_SERVING_DISTRIBUTED_SERVICE_POOL_H

#include "common/request.h"
#include "engine/serving_system.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace serving {

// L5 ServicePool — manages a pool of L4 Pod instances for either
// prefill or decode, supporting batch-size tiers.
//
// DSL spec: L5 Supernode level.
// Each pool entry is a ServingSystem instance with its own L4 Pod + L3 engines.

enum class ServiceRole { PREFILL, DECODE, MIXED };

struct PoolEntry {
    std::string name;
    int32_t batch_size;
    ServiceRole role;
    std::unique_ptr<ServingSystem> system;
    int32_t active_requests = 0;
};

struct ServicePoolConfig {
    std::string name = "service_pool";
    ServiceRole role = ServiceRole::MIXED;
    // Instance definitions: each pair is (batch_size, num_chips_per_server)
    std::vector<std::pair<int32_t, int32_t>> instances;
    bool enable_trace = false;
};

class ServicePool {
public:
    explicit ServicePool(const ServicePoolConfig& cfg);
    ~ServicePool();

    void start();
    void stop();

    // Select an instance based on load (least loaded).
    // Returns the index of the selected instance, or -1 if none available.
    int select_instance();

    // Execute a request on a specific instance.
    Response execute(int instance_idx, const Request& request);

    // Execute on the best available instance.
    Response execute_auto(const Request& request);

    int pool_size() const { return static_cast<int>(entries_.size()); }
    const PoolEntry& entry(int idx) const { return entries_[idx]; }

private:
    ServicePoolConfig cfg_;
    std::vector<PoolEntry> entries_;
    mutable std::mutex mu_;
};

// ─── L6 QoS Tier ─────────────────────────────────────────────────────

enum class QosTier { LOW_LATENCY, HIGH_THROUGHPUT };

struct QosConfig {
    QosTier tier = QosTier::LOW_LATENCY;
    float max_ttft_ms = 100.0f;    // Time To First Token
    float max_tpot_ms = 50.0f;     // Time Per Output Token
};

// L6 ClusterCoordinator — manages QoS-differentiated service pools.
//
// DSL spec: L6 Cluster Coordinator level.
// Routes requests to the appropriate prefill/decode pool based on QoS tier.

struct ClusterConfig {
    std::string name = "cluster";
    std::vector<ServicePoolConfig> prefill_pools;
    std::vector<ServicePoolConfig> decode_pools;
    bool enable_trace = false;
};

class ClusterCoordinator {
public:
    explicit ClusterCoordinator(const ClusterConfig& cfg);
    ~ClusterCoordinator();

    void start();
    void stop();

    // Route a request with a specific QoS tier.
    Response route_request(const Request& request, QosTier tier = QosTier::LOW_LATENCY);

    int num_prefill_pools() const { return static_cast<int>(prefill_pools_.size()); }
    int num_decode_pools() const { return static_cast<int>(decode_pools_.size()); }

private:
    ClusterConfig cfg_;
    std::vector<std::unique_ptr<ServicePool>> prefill_pools_;
    std::vector<std::unique_ptr<ServicePool>> decode_pools_;
    mutable std::mutex mu_;
};

// ─── L7 GlobalRouter ──────────────────────────────────────────────────

// L7 GlobalRouter — top-level request router.
//
// DSL spec: L7 Global Coordinator level.
// Routes requests by QoS tier to the appropriate L6 cluster.

struct GlobalRouterConfig {
    std::string name = "global_router";
    std::vector<ClusterConfig> clusters;
    bool enable_trace = false;
};

class GlobalRouter {
public:
    explicit GlobalRouter(const GlobalRouterConfig& cfg);
    ~GlobalRouter();

    void start();
    void stop();

    Response route(const Request& request, QosTier tier = QosTier::LOW_LATENCY);

    int num_clusters() const { return static_cast<int>(clusters_.size()); }

private:
    GlobalRouterConfig cfg_;
    std::vector<std::unique_ptr<ClusterCoordinator>> clusters_;
    mutable std::mutex mu_;
};

}  // namespace serving

#endif
