// src/prefill/npu/npu_abi.cpp - Talus: the tk_qwen4exp_p2_npu_* C ABI entries.
//
// Thin extern "C" shims over NpuAssist so the Rust prefill (qwen4exp/prefill.rs,
// via the P2Ffi dlsym table) can drive the XDNA2 NPU assist. Stubs are exported
// when the build has no XRT (TALUS_NPU undefined), so the FFI table always
// resolves.
#include <cstdint>

#include <hip/hip_runtime.h>

#if defined(TALUS_NPU)
#include "npu_assist.hpp"
using talus::npu::NpuAssist;
static NpuAssist& assist() {
    static NpuAssist a;
    return a;
}
static bool npu_inited = false;
#endif

extern "C" {

#if defined(TALUS_NPU)
int tk_qwen4exp_p2_npu_init() {
    if (npu_inited) return assist().active() ? 1 : 0;
    npu_inited = true;
    std::string err;
    if (!assist().Init(&err)) {
        std::fprintf(stderr, "talus prefill: NPU init failed: %s\n", err.c_str());
        return 0;
    }
    return assist().active() ? 1 : 0;
}

int tk_qwen4exp_p2_npu_stage(const void* native_data, int native_type, long long ne1,
                             long long ne0, void* stream) {
    if (!assist().active() || native_data == nullptr) return 0;
    return static_cast<int>(assist().StageGate(
        native_data, native_type, ne1, ne0, assist().stage_scratch(),
        static_cast<hipStream_t>(stream)));
}

unsigned short* tk_qwen4exp_p2_npu_scratch() { return assist().stage_scratch(); }

int tk_qwen4exp_p2_npu_usable(long long t, int slices) {
    return NpuAssist::Usable(t, static_cast<unsigned>(slices)) ? 1 : 0;
}

int tk_qwen4exp_p2_npu_submit(const unsigned short* x, int x_is_bf16, long long t,
                              const void* native_data, void* stream) {
    const std::vector<int>& b = assist().Staged(native_data);
    if (b.empty()) return 0;
    assist().SubmitGate(x, x_is_bf16 != 0, t, b, static_cast<hipStream_t>(stream));
    return 1;
}

void tk_qwen4exp_p2_npu_finish(float* z, long long t, long long z_pitch,
                               const void* native_data, void* stream) {
    const std::vector<int>& b = assist().Staged(native_data);
    if (b.empty()) return;
    assist().FinishGate(z, t, z_pitch, b, static_cast<hipStream_t>(stream));
}
#else
int tk_qwen4exp_p2_npu_init() { return 0; }
int tk_qwen4exp_p2_npu_stage(const void*, int, long long, long long, void*) { return 0; }
unsigned short* tk_qwen4exp_p2_npu_scratch() { return nullptr; }
int tk_qwen4exp_p2_npu_usable(long long, int) { return 0; }
int tk_qwen4exp_p2_npu_submit(const unsigned short*, int, long long, const void*, void*) { return 0; }
void tk_qwen4exp_p2_npu_finish(float*, long long, long long, const void*, void*) {}
#endif

}  // extern "C"
