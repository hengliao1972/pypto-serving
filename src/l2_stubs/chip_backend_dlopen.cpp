#include "l2_stubs/chip_backend_dlopen.h"

#include <cstdio>
#include <cstring>
#include <dlfcn.h>

namespace serving {

static const char* kDefaultLibPaths[] = {
    "libhost_runtime.so",
    "./libhost_runtime.so",
    "../simpler/build/libhost_runtime.so",
    nullptr,
};

ChipBackendDlopen::ChipBackendDlopen(int num_chips,
                                       const std::string& lib_path)
    : num_chips_(num_chips) {
    // Try to load the shared library.
    if (!lib_path.empty()) {
        handle_ = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle_) {
            fprintf(stderr, "[ChipBackendDlopen] Warning: cannot load '%s': %s\n",
                    lib_path.c_str(), dlerror());
        }
    } else {
        for (const char** p = kDefaultLibPaths; *p; p++) {
            handle_ = dlopen(*p, RTLD_NOW | RTLD_LOCAL);
            if (handle_) break;
        }
    }

    if (handle_) {
        fn_prefill_ = reinterpret_cast<PrefillFn>(
            dlsym(handle_, "simpler_model_prefill"));
        fn_decode_ = reinterpret_cast<DecodeFn>(
            dlsym(handle_, "simpler_model_decode"));

        if (!fn_prefill_ || !fn_decode_) {
            fprintf(stderr, "[ChipBackendDlopen] Warning: symbols not found, "
                    "falling back to stub\n");
            dlclose(handle_);
            handle_ = nullptr;
            fn_prefill_ = nullptr;
            fn_decode_ = nullptr;
        } else {
            fprintf(stderr, "[ChipBackendDlopen] Loaded real L2 kernel\n");
        }
    } else {
        fprintf(stderr, "[ChipBackendDlopen] Using stub fallback "
                "(libhost_runtime.so not found)\n");
    }
}

ChipBackendDlopen::~ChipBackendDlopen() {
    if (handle_) {
        dlclose(handle_);
        handle_ = nullptr;
    }
}

// ─── model_prefill ────────────────────────────────────────────────────

void ChipBackendDlopen::model_prefill(int chip_id,
                                        linqu::LinquTensor token_ids,
                                        linqu::LinquTensor kv_blocks,
                                        linqu::LinquTensor kv_out,
                                        linqu::LinquTensor first_logits) {
    prefill_calls++;

    if (fn_prefill_) {
        fn_prefill_(chip_id,
                    token_ids.data_ptr(),
                    static_cast<int>(token_ids.count),
                    kv_blocks.data_ptr(),
                    static_cast<int>(kv_blocks.count),
                    kv_out.data_ptr(),
                    first_logits.data_ptr(),
                    static_cast<int>(first_logits.count));
    } else {
        stub_prefill(chip_id, kv_out, first_logits);
    }
}

void ChipBackendDlopen::model_decode(int chip_id,
                                       linqu::LinquTensor prev_token,
                                       linqu::LinquTensor kv_blocks,
                                       linqu::LinquTensor next_logits,
                                       linqu::LinquTensor updated_kv) {
    decode_calls++;

    if (fn_decode_) {
        uint64_t prev = (prev_token.data_ptr() && prev_token.count > 0)
                        ? prev_token.data_ptr()[0] : 0;
        fn_decode_(chip_id, prev,
                   kv_blocks.data_ptr(),
                   static_cast<int>(kv_blocks.count),
                   next_logits.data_ptr(),
                   static_cast<int>(next_logits.count),
                   updated_kv.data_ptr(),
                   static_cast<int>(updated_kv.count));
    } else {
        stub_decode(chip_id, next_logits, updated_kv);
    }
}

// ─── Stub fallback (identical to ChipBackendStub) ─────────────────────

void ChipBackendDlopen::stub_prefill(int chip_id,
                                       linqu::LinquTensor kv_out,
                                       linqu::LinquTensor first_logits) {
    if (kv_out.data_ptr()) {
        for (size_t i = 0; i < kv_out.count; i++)
            kv_out.data_ptr()[i] = static_cast<uint64_t>(chip_id * 1000 + i);
    }
    if (first_logits.data_ptr()) {
        for (size_t i = 0; i < first_logits.count; i++)
            first_logits.data_ptr()[i] = static_cast<uint64_t>((chip_id + i) % 32);
    }
}

void ChipBackendDlopen::stub_decode(int chip_id,
                                      linqu::LinquTensor next_logits,
                                      linqu::LinquTensor updated_kv) {
    if (next_logits.data_ptr()) {
        for (size_t i = 0; i < next_logits.count; i++)
            next_logits.data_ptr()[i] = static_cast<uint64_t>((chip_id + i + 1) % 32);
    }
    if (updated_kv.data_ptr()) {
        for (size_t i = 0; i < updated_kv.count; i++)
            updated_kv.data_ptr()[i] = static_cast<uint64_t>(chip_id * 100 + i);
    }
}

}  // namespace serving
