#ifndef PYPTO_SERVING_FRONTEND_TEST_PATH_H
#define PYPTO_SERVING_FRONTEND_TEST_PATH_H

#include <cstddef>
#include <cstdint>

// =========================================================================
// TestPath — in-process request injection / response retrieval interface.
//
// A zero-network-overhead path for testing: Python (or C++) injects a
// serialized request via inject_request(), the serving engine picks it up,
// runs the full L4→L3→L2 pipeline, and posts the response.  The caller
// retrieves it via get_response().
//
// Wire format (Phase 1, simple):
//   Request  buffer: [max_tokens:u32][temperature:f32][top_p:f32]
//                    [stop_token:i32][vocab_size:i32][kv_size:i32]
//                    [kv_step:i32][n_tokens:u32][token_0:u64]...[token_N:u64]
//   Response buffer: [n_tokens:u32][token_0:u64]...[token_N:u64]
//
// C-linkage API so Python can call via ctypes.
// =========================================================================

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the TestPath subsystem.  Must be called once before inject/get.
// Returns 0 on success, non-zero on error.
int testpath_init(void);

// Shut down the TestPath subsystem.
void testpath_shutdown(void);

// Start the serving engine (L4+L3 runtimes).
// trace_path: if non-NULL, enable Perfetto trace to this file.
int testpath_start(const char* trace_path);

// Stop the serving engine.
void testpath_stop(void);

// Inject a serialized request.  Blocks until the request is queued.
// Returns 0 on success.
int testpath_inject_request(const void* buf, size_t len);

// Retrieve the response for the most recent request.
// Blocks until the response is available.
// Writes up to max_len bytes into buf.  Returns actual bytes written,
// or -1 on error.
int64_t testpath_get_response(void* buf, size_t max_len);

// Write trace file (if tracing was enabled at start).
// Returns 0 on success, -1 if tracing disabled.
int testpath_write_trace(void);

#ifdef __cplusplus
}
#endif

#endif
