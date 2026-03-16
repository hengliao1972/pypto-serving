/*
 * test_phase7_distributed.cpp — Phase 7 distributed inference tests.
 *
 * Validates:
 *   1. L5 ServicePool: multi-instance pool, load balancing
 *   2. L6 ClusterCoordinator: QoS-based routing
 *   3. L7 GlobalRouter: top-level request routing
 *   4. Multi-level hierarchy: L7 → L6 → L5 → L4 → L3 → L2
 */

#include "distributed/service_pool.h"

#include <cassert>
#include <cstdio>

using namespace serving;

static Request make_req(int n_tokens = 4, int max_tokens = 2) {
    Request req;
    for (int i = 0; i < n_tokens; i++)
        req.token_ids.push_back(static_cast<uint64_t>(100 + i));
    req.max_tokens = max_tokens;
    req.vocab_size = 32;
    req.kv_size_per_chip = 64;
    req.kv_step_size = 64;
    return req;
}

// ─── L5 ServicePool ──────────────────────────────────────────────────

static void test_service_pool_basic() {
    ServicePoolConfig cfg;
    cfg.name = "test_pool";
    cfg.role = ServiceRole::MIXED;
    cfg.instances = {{8, 16}};  // 1 instance, batch_size=8, 16 chips

    ServicePool pool(cfg);
    pool.start();

    assert(pool.pool_size() == 1);

    auto resp = pool.execute_auto(make_req());
    assert(!resp.output_tokens.empty());

    pool.stop();
    fprintf(stderr, "  %-50s [PASS]\n", "service_pool_basic");
}

static void test_service_pool_multi_instance() {
    ServicePoolConfig cfg;
    cfg.name = "multi_pool";
    cfg.role = ServiceRole::MIXED;
    cfg.instances = {{8, 16}, {32, 16}};  // 2 instances

    ServicePool pool(cfg);
    pool.start();

    assert(pool.pool_size() == 2);

    // Both instances should work
    auto r1 = pool.execute(0, make_req());
    auto r2 = pool.execute(1, make_req());
    assert(!r1.output_tokens.empty());
    assert(!r2.output_tokens.empty());

    // Auto-select should work
    auto r3 = pool.execute_auto(make_req());
    assert(!r3.output_tokens.empty());

    pool.stop();
    fprintf(stderr, "  %-50s [PASS]\n", "service_pool_multi_instance");
}

// ─── L6 ClusterCoordinator ───────────────────────────────────────────

static void test_cluster_basic() {
    ServicePoolConfig prefill_pool;
    prefill_pool.name = "prefill_low_lat";
    prefill_pool.role = ServiceRole::PREFILL;
    prefill_pool.instances = {{8, 16}};

    ClusterConfig cfg;
    cfg.name = "test_cluster";
    cfg.prefill_pools = {prefill_pool};

    ClusterCoordinator cluster(cfg);
    cluster.start();

    assert(cluster.num_prefill_pools() == 1);

    auto resp = cluster.route_request(make_req(), QosTier::LOW_LATENCY);
    assert(!resp.output_tokens.empty());

    cluster.stop();
    fprintf(stderr, "  %-50s [PASS]\n", "cluster_basic");
}

static void test_cluster_qos_routing() {
    ServicePoolConfig low_lat;
    low_lat.name = "prefill_low_lat";
    low_lat.role = ServiceRole::PREFILL;
    low_lat.instances = {{8, 16}};

    ServicePoolConfig high_tp;
    high_tp.name = "prefill_high_tp";
    high_tp.role = ServiceRole::PREFILL;
    high_tp.instances = {{32, 16}};

    ClusterConfig cfg;
    cfg.name = "qos_cluster";
    cfg.prefill_pools = {low_lat, high_tp};

    ClusterCoordinator cluster(cfg);
    cluster.start();

    assert(cluster.num_prefill_pools() == 2);

    auto r1 = cluster.route_request(make_req(), QosTier::LOW_LATENCY);
    auto r2 = cluster.route_request(make_req(), QosTier::HIGH_THROUGHPUT);
    assert(!r1.output_tokens.empty());
    assert(!r2.output_tokens.empty());

    cluster.stop();
    fprintf(stderr, "  %-50s [PASS]\n", "cluster_qos_routing");
}

// ─── L7 GlobalRouter ─────────────────────────────────────────────────

static void test_global_router() {
    ServicePoolConfig pool_cfg;
    pool_cfg.name = "global_pool";
    pool_cfg.role = ServiceRole::MIXED;
    pool_cfg.instances = {{8, 16}};

    ClusterConfig cluster_cfg;
    cluster_cfg.name = "global_cluster";
    cluster_cfg.prefill_pools = {pool_cfg};

    GlobalRouterConfig rcfg;
    rcfg.name = "test_router";
    rcfg.clusters = {cluster_cfg};

    GlobalRouter router(rcfg);
    router.start();

    assert(router.num_clusters() == 1);

    auto resp = router.route(make_req(), QosTier::LOW_LATENCY);
    assert(!resp.output_tokens.empty());

    router.stop();
    fprintf(stderr, "  %-50s [PASS]\n", "global_router");
}

// ─── Full L7→L6→L5→L4→L3→L2 hierarchy ──────────────────────────────

static void test_full_hierarchy() {
    ServicePoolConfig low_lat;
    low_lat.name = "prefill_ll";
    low_lat.role = ServiceRole::MIXED;
    low_lat.instances = {{8, 16}};

    ServicePoolConfig high_tp;
    high_tp.name = "prefill_ht";
    high_tp.role = ServiceRole::MIXED;
    high_tp.instances = {{32, 16}};

    ClusterConfig cluster;
    cluster.name = "main_cluster";
    cluster.prefill_pools = {low_lat, high_tp};

    GlobalRouterConfig gcfg;
    gcfg.name = "full_router";
    gcfg.clusters = {cluster};

    GlobalRouter router(gcfg);
    router.start();

    // Low latency request
    auto r1 = router.route(make_req(4, 2), QosTier::LOW_LATENCY);
    assert(!r1.output_tokens.empty());

    // High throughput request
    auto r2 = router.route(make_req(8, 4), QosTier::HIGH_THROUGHPUT);
    assert(!r2.output_tokens.empty());

    router.stop();

    fprintf(stderr, "  %-50s [PASS] (ll=%zu, ht=%zu tokens)\n",
            "full_hierarchy",
            r1.output_tokens.size(), r2.output_tokens.size());
}

// ─── Main ─────────────────────────────────────────────────────────────

int main() {
    fprintf(stderr, "=== Phase 7: Distributed Inference Tests ===\n\n");

    fprintf(stderr, "L5 ServicePool:\n");
    test_service_pool_basic();
    test_service_pool_multi_instance();

    fprintf(stderr, "\nL6 ClusterCoordinator:\n");
    test_cluster_basic();
    test_cluster_qos_routing();

    fprintf(stderr, "\nL7 GlobalRouter:\n");
    test_global_router();

    fprintf(stderr, "\nFull Hierarchy (L7→L6→L5→L4→L3→L2):\n");
    test_full_hierarchy();

    fprintf(stderr, "\n=== Phase 7: All Tests PASSED ===\n");
    return 0;
}
