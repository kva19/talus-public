// src/prefill/npu/npu_assist.hpp - Talus: the XDNA2 NPU assist for GDN prefill.
//
// Offloads the attn_gate (z) projection of linear-attention layers to the NPU
// (f-gufo's technique): the GPU runs attn_qkv + alpha/beta + the conv while
// the NPU computes z concurrently over the same unified memory. Switched on by
// STRATA_PF_NPU=<dir with the .xclbin + insts>; unset = fully off (no XRT
// call is made, the binary just carries the code).
//
// One NpuGemm context serves everything (shape 4096 x 2560 x 1024). Chunks of
// T tokens run as T/4096 groups (<= kMaxGroups in flight); activations are
// packed into pinned A buffers by a GPU kernel, the per-layer BF16-dequantized
// gate weights are staged once into pinned B buffers, and the z output is
// decoded back to F32 rows after Wait.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>

#include "npu_gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"

namespace talus {
namespace npu {

// Launchers in npu_kernels.cu (compiled as HIP; declared here for the C++ TU).
void EncodeActivations(const std::uint16_t* x, bool is_bf16, void* dst,
                       std::uint32_t tokens, std::size_t ldx, std::uint32_t k,
                       hipStream_t stream);
void EncodeWeightsBf16(const std::uint16_t* w, void* dst, std::uint32_t rows,
                       std::uint32_t k, hipStream_t stream);
void DecodeOutput(const void* src, float* out, std::uint32_t tokens,
                  std::uint32_t cols, std::size_t out_pitch, std::uint32_t col0,
                  std::uint32_t valid_cols, hipStream_t stream);

class NpuAssist {
 public:
  static constexpr std::uint32_t kTokens = 4096;   // the NPU kernel's M
  static constexpr std::uint32_t kK = 2560;        // hidden (attn_gate ne0)
  static constexpr std::uint32_t kSlice = 1024;    // NPU output columns per run
  static constexpr std::uint32_t kMaxSlices = 12;  // attn_gate 6144 rows = 6
  static constexpr std::uint32_t kMaxGroups = 4;   // in-flight 4096-token groups

  bool active() const { return gemm_ != nullptr; }
  std::uint32_t slices() const { return slices_; }

  /// A staged projection: NPU B buffers (one per 1024-row slice) + the
  /// weight's real row count (the last slice may be partial: 640-row shexp).
  struct StagedWeight {
    std::vector<int> b;
    std::uint32_t rows;
  };

  /// The staged entry of a weight staged by Stage (empty b if none).
  const StagedWeight* Staged(const void* native_data) {
    auto it = staged_.find(native_data);
    return it != staged_.end() ? &it->second : nullptr;
  }

  /// Reads STRATA_PF_NPU and, when set, creates the NPU context and buffers.
  /// Returns false with `err` filled on a real failure (a missing variable
  /// just leaves this inactive).
  bool Init(std::string* err) {
    const char* dir = std::getenv("STRATA_PF_NPU");
    if (dir == nullptr || dir[0] == '\0') return true;
    gufo::models::qwen38_flash_next::rocm::NpuGemm::Config cfg;
    cfg.xclbin = std::string(dir) + "/final_4096x2560x1024_128x64x128.xclbin";
    cfg.insts = std::string(dir) + "/insts_4096x2560x1024_128x64x128.txt";
    cfg.tokens = kTokens; cfg.k = kK; cfg.n = kSlice;
    gemm_ = gufo::models::qwen38_flash_next::rocm::NpuGemm::Create(cfg, err);
    if (!gemm_) return false;
    for (std::uint32_t g = 0; g < kMaxGroups; ++g) {
      const int a = gemm_->NewBuffer(gemm_->ABytes(), err);
      if (a < 0) return false;
      a_.push_back(a);
      for (std::uint32_t i = 0; i < kMaxSlices; ++i) {
        const int c = gemm_->NewBuffer(gemm_->CBytes(), err);
        if (c < 0) return false;
        c_.push_back(c);
      }
    }
    if (hipMalloc(&stage_scratch_, static_cast<size_t>(kSlice) * kK * 2) !=
        hipSuccess) {
      *err = "NPU staging scratch alloc failed";
      return false;
    }
    std::fprintf(stderr,
                 "talus prefill: NPU assist on (%s), %u groups x %u slices\n",
                 dir, kMaxGroups, kMaxSlices);
    return true;
  }

  std::uint16_t* stage_scratch() { return stage_scratch_; }

  /// Stages one projection's weights (any native GGUF type, any row count:
  /// partial last slices are zero-padded and skipped at decode) into pinned
  /// NPU B buffers, dequantizing through `scratch16` (>= kSlice * kK bf16
  /// elements, a device buffer). Idempotent per weight pointer.
  /// Returns the slice count, or 0 when this weight stays on the GPU.
  std::uint32_t Stage(const void* native_data, int ggml_type, int64_t ne1,
                      int64_t ne0, std::uint16_t* scratch16,
                      hipStream_t stream) {
    if (!gemm_ || ne0 != kK || ne1 <= 0) return 0;
    auto it = staged_.find(native_data);
    if (it != staged_.end()) return static_cast<std::uint32_t>(it->second.b.size());
    const std::uint32_t rows_pad =
        static_cast<std::uint32_t>((ne1 + kSlice - 1) / kSlice) * kSlice;
    const std::uint32_t slices = rows_pad / kSlice;
    if (slices > kMaxSlices) return 0;
    std::vector<int> b;
    std::string err;
    // zero the scratch once: the partial last slice reads padded zero rows
    hipMemsetAsync(scratch16, 0, static_cast<size_t>(kSlice) * kK * 2, stream);
    for (std::uint32_t s = 0; s < slices; ++s) {
      const int64_t row0 = static_cast<int64_t>(s) * kSlice;
      const int64_t rows = std::min<int64_t>(kSlice, ne1 - row0);
      strata::kernels::dequant_bf16(ggml_type, native_data, row0, rows, ne0,
                                    scratch16, stream);
      const int id = gemm_->NewBuffer(gemm_->BBytes(), &err);
      if (id < 0) {
        std::fprintf(stderr, "talus prefill: NPU B buffer: %s\n", err.c_str());
        return 0;
      }
      EncodeWeightsBf16(scratch16, gemm_->Host(id), kSlice, kK, stream);
      b.push_back(id);
    }
    hipStreamSynchronize(stream);
    for (int id : b) gemm_->FinalizeWeights(id);
    staged_.emplace(native_data, StagedWeight{b, static_cast<std::uint32_t>(ne1)});
    return slices;
  }

  /// `z` GEMM for this chunk: encodes the activations and submits the NPU
  /// runs, to be issued right before the GPU's own attn_qkv GEMM so both
  /// devices work concurrently. The caller must have confirmed
  /// Usable(T, slices).
  void Submit(const std::uint16_t* x, bool x_is_bf16, int64_t T,
              const StagedWeight& w, hipStream_t stream) {
    const std::vector<int>& b = w.b;
    const std::uint32_t groups = static_cast<std::uint32_t>(T) / kTokens;
    for (std::uint32_t g = 0; g < groups; ++g) {
      EncodeActivations(x + static_cast<std::size_t>(g) * kTokens * kK, x_is_bf16,
                        gemm_->Device(a_[g]), kTokens, kK, kK, stream);
      gemm_->Submit(stream, a_[g], b.data(),
                    c_.data() + static_cast<std::size_t>(g) * kMaxSlices + 0,
                    static_cast<int>(b.size()));
    }
  }

  /// Blocks the stream until the NPU runs finished and decodes z (F32, row
  /// stride z_pitch) for the whole chunk. Call right before the first z reader.
  void Finish(float* out, int64_t T, int64_t pitch, const StagedWeight& w,
              hipStream_t stream) {
    const std::uint32_t groups = static_cast<std::uint32_t>(T) / kTokens;
    gemm_->Wait(stream);
    for (std::uint32_t g = 0; g < groups; ++g) {
      for (std::size_t i = 0; i < w.b.size(); ++i) {
        const std::uint32_t valid =
            std::min<std::uint32_t>(kSlice, w.rows - static_cast<std::uint32_t>(i) * kSlice);
        DecodeOutput(gemm_->Staged(stream, c_[g * kMaxSlices + i]),
                     out + static_cast<std::size_t>(g) * kTokens * pitch,
                     kTokens, kSlice, static_cast<std::size_t>(pitch),
                     static_cast<std::uint32_t>(i) * kSlice, valid, stream);
      }
    }
  }

  static bool Usable(int64_t T, std::uint32_t slices) {
    return slices > 0 && T > 0 && T % kTokens == 0 &&
           static_cast<std::uint64_t>(T) / kTokens <= kMaxGroups;
  }

 private:
  std::unique_ptr<gufo::models::qwen38_flash_next::rocm::NpuGemm> gemm_;
  std::uint16_t* stage_scratch_ = nullptr;
  std::vector<int> a_, c_;
  std::unordered_map<const void*, StagedWeight> staged_;
  std::uint32_t slices_ = 0;
};

}  // namespace npu
}  // namespace talus
