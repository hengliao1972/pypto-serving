#include "distributed/service_pool.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <limits>

namespace serving {

// =========================================================================
// L5 ServicePool
// =========================================================================

ServicePool::ServicePool(const ServicePoolConfig& cfg) : cfg_(cfg) {
    for (auto& [batch_size, num_chips] : cfg_.instances) {
        PoolEntry entry;
        entry.name = cfg_.name + "_bs" + std::to_string(batch_size);
        entry.batch_size = batch_size;
        entry.role = cfg_.role;

        ServingConfig scfg;
        scfg.num_chips_per_server = num_chips;
        scfg.enable_trace = cfg_.enable_trace;
        scfg.trace_path = entry.name + "_trace.json";
        entry.system = std::make_unique<ServingSystem>(scfg);

        entries_.push_back(std::move(entry));
    }
}

ServicePool::~ServicePool() { stop(); }

void ServicePool::start() {
    for (auto& e : entries_)
        e.system->start();
}

void ServicePool::stop() {
    for (auto& e : entries_)
        e.system->stop();
}

int ServicePool::select_instance() {
    std::lock_guard<std::mutex> lk(mu_);
    if (entries_.empty()) return -1;

    int best = 0;
    int min_load = entries_[0].active_requests;
    for (int i = 1; i < static_cast<int>(entries_.size()); i++) {
        if (entries_[i].active_requests < min_load) {
            min_load = entries_[i].active_requests;
            best = i;
        }
    }
    return best;
}

Response ServicePool::execute(int idx, const Request& request) {
    assert(idx >= 0 && idx < static_cast<int>(entries_.size()));
    auto& e = entries_[idx];

    {
        std::lock_guard<std::mutex> lk(mu_);
        e.active_requests++;
    }

    Response resp = e.system->infer(request);

    {
        std::lock_guard<std::mutex> lk(mu_);
        e.active_requests--;
    }

    return resp;
}

Response ServicePool::execute_auto(const Request& request) {
    int idx = select_instance();
    assert(idx >= 0);
    return execute(idx, request);
}

// =========================================================================
// L6 ClusterCoordinator
// =========================================================================

ClusterCoordinator::ClusterCoordinator(const ClusterConfig& cfg) : cfg_(cfg) {
    for (auto& pc : cfg_.prefill_pools) {
        prefill_pools_.push_back(std::make_unique<ServicePool>(pc));
    }
    for (auto& dc : cfg_.decode_pools) {
        decode_pools_.push_back(std::make_unique<ServicePool>(dc));
    }
}

ClusterCoordinator::~ClusterCoordinator() { stop(); }

void ClusterCoordinator::start() {
    for (auto& p : prefill_pools_) p->start();
    for (auto& p : decode_pools_) p->start();
}

void ClusterCoordinator::stop() {
    for (auto& p : prefill_pools_) p->stop();
    for (auto& p : decode_pools_) p->stop();
}

Response ClusterCoordinator::route_request(const Request& request,
                                            QosTier tier) {
    // In the current topology, prefill and decode are combined in a single
    // ServingSystem (which has both engines).  Route to a prefill pool
    // instance which handles the full pipeline.
    //
    // QoS tier selects which pool to use:
    //   LOW_LATENCY → first pool (smaller batch sizes)
    //   HIGH_THROUGHPUT → last pool (larger batch sizes)

    std::lock_guard<std::mutex> lk(mu_);
    if (prefill_pools_.empty()) {
        // Fallback: try decode pools
        if (!decode_pools_.empty()) {
            return decode_pools_[0]->execute_auto(request);
        }
        assert(false && "No pools available");
    }

    int pool_idx = 0;
    if (tier == QosTier::HIGH_THROUGHPUT && prefill_pools_.size() > 1) {
        pool_idx = static_cast<int>(prefill_pools_.size()) - 1;
    }

    return prefill_pools_[pool_idx]->execute_auto(request);
}

// =========================================================================
// L7 GlobalRouter
// =========================================================================

GlobalRouter::GlobalRouter(const GlobalRouterConfig& cfg) : cfg_(cfg) {
    for (auto& cc : cfg_.clusters) {
        clusters_.push_back(std::make_unique<ClusterCoordinator>(cc));
    }
}

GlobalRouter::~GlobalRouter() { stop(); }

void GlobalRouter::start() {
    for (auto& c : clusters_) c->start();
}

void GlobalRouter::stop() {
    for (auto& c : clusters_) c->stop();
}

Response GlobalRouter::route(const Request& request, QosTier tier) {
    std::lock_guard<std::mutex> lk(mu_);
    if (clusters_.empty()) {
        assert(false && "No clusters available");
    }

    // Round-robin or first-available for now.
    // Future: content-based routing, affinity, locality.
    return clusters_[0]->route_request(request, tier);
}

}  // namespace serving
