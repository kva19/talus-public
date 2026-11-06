// src/prefill/npu/npu_kernels.cu - Talus: BFP (v8bfp16ebs8) staging kernels for the XDNA2 NPU GEMM.
//
// Format and tile layouts ported from f-gufo (MIT, ZakFong/f-gufo,
// src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp): 9-byte groups
// (one shared exponent byte + eight signed 8-bit mantissas,
// value = q * 2^(e - 127) / 64). A tiles 128x64 row-major over (M, K) with
// 2x1 super blocks of 8x8 row-major blocks; B tiles column-major over (K, N)
// with 1x2 super blocks of 8x8 column-major blocks; C tiles 512x128 row-major
// with 2x2 super blocks (read back by NpuDecodeOutputKernel).
//
// Differences from f-gufo's versions: the activations kernel takes 16-bit
// input that is FP16 or BF16 (`is_bf16`, the Talus prompt path's two modes),
// and the weights kernel encodes from the BF16 image that
// strata::kernels::dequant_bf16 produces for every native GGUF type (so the
// mixed-quant attn_gate layers all stage through the same path).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstddef>

namespace talus {
namespace npu {

constexpr int kNpuTileM = 128;
constexpr int kNpuTileK = 64;
constexpr int kNpuTileN = 128;
constexpr int kNpuTileRowsC = 4 * kNpuTileM;   // 512
constexpr int kNpuThreads = 256;

__device__ __forceinline__ void EncodeGroup(const float* v, std::uint8_t* out) {
  unsigned max_exp = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    max_exp = max(max_exp, (__float_as_uint(v[i]) >> 23) & 0xFFu);
  }
  // 2^(133 - max_exp): the exact reciprocal of the group's mantissa unit.
  const float inv = __uint_as_float(max_exp >= 7 ? (260u - max_exp) << 23 : 0u);
  out[0] = static_cast<std::uint8_t>(max_exp);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float q = fminf(127.0F, fmaxf(-128.0F, rintf(v[i] * inv)));
    out[1 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
  }
}

__device__ __forceinline__ float In16(std::uint16_t bits, bool is_bf16) {
  return is_bf16 ? __uint_as_float(static_cast<std::uint32_t>(bits) << 16)
                 : __half2float(__ushort_as_half(bits));
}

/// One block per A tile (128 x 64) of one 4096-token group: the FP16/BF16
/// activations of `tokens` rows (row stride ldx) packed into BFP groups.
/// f-gufo's NpuEncodeActivationsKernel with the 16-bit-mode parameter.
__global__ void __launch_bounds__(kNpuThreads)
EncodeActivationsKernel(const std::uint16_t* __restrict__ x, bool is_bf16,
                        std::uint32_t tokens, std::size_t ldx,
                        std::uint32_t k_valid, std::uint8_t* __restrict__ dst) {
  __shared__ __align__(16) float tile[kNpuTileM * kNpuTileK];
  __shared__ __align__(16) std::uint8_t staged[kNpuTileM * kNpuTileK / 8 * 9];
  const std::uint32_t tiles_k = k_valid / kNpuTileK;
  const std::uint32_t tile_row = blockIdx.x / tiles_k;
  const std::uint32_t tile_col = blockIdx.x % tiles_k;
  const std::uint32_t t = threadIdx.x;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const std::uint32_t row = (t >> 3) + 32 * j;
    const std::uint32_t vec = t & 7;
    const std::uint32_t col = tile_col * kNpuTileK + vec * 8;
    const std::uint32_t gr = tile_row * kNpuTileM + row;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      tile[row * kNpuTileK + vec * 8 + i] =
          gr < tokens ? In16(x[static_cast<std::size_t>(gr) * ldx + col + i], is_bf16) : 0.0F;
    }
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const std::uint32_t gi = t + 256 * j;
    const std::uint32_t rib = gi & 7;
    const std::uint32_t block = (gi >> 3) & 1;
    const std::uint32_t sc = (gi >> 4) & 7;
    const std::uint32_t sr = gi >> 7;
    const std::uint32_t row = sr * 16 + block * 8 + rib;
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = tile[row * kNpuTileK + sc * 8 + i];
    EncodeGroup(v, staged + gi * 9);
  }
  __syncthreads();
  const std::uint32_t words = (kNpuTileM * kNpuTileK / 8) * 9 / 16;  // 576 uint4s
  for (std::uint32_t i = t; i < words; i += kNpuThreads) {
    reinterpret_cast<uint4*>(dst)[blockIdx.x * words + i] =
        reinterpret_cast<const uint4*>(staged)[i];
  }
}

/// Weights from the BF16 dequant image: `rows` x `k` row-major. One thread per
/// group; the B tile layout matches f-gufo's NpuEncodeWeightsQ8Kernel (the
/// source is a bf16 matrix instead of Q8_0 blocks).
__global__ void EncodeWeightsBf16Kernel(const std::uint16_t* __restrict__ w,
                                        std::uint8_t* __restrict__ out,
                                        std::uint32_t rows, std::uint32_t k) {
  const std::size_t g = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  const std::size_t total = std::size_t{k} * kNpuTileN / 8 * (rows / kNpuTileN);
  if (g >= total) return;
  constexpr std::uint32_t kGroupsPerTile = kNpuTileK * kNpuTileN / 8;  // 1024
  const std::size_t tile = g / kGroupsPerTile;
  const std::uint32_t gi = static_cast<std::uint32_t>(g % kGroupsPerTile);
  const std::uint32_t tiles_k = k / kNpuTileK;                         // 40
  const std::uint32_t tile_col = static_cast<std::uint32_t>(tile / tiles_k);
  const std::uint32_t tile_row = static_cast<std::uint32_t>(tile % tiles_k);
  const std::uint32_t cib = gi & 7;
  const std::uint32_t block = (gi >> 3) & 1;
  const std::uint32_t sr = (gi >> 4) & 7;
  const std::uint32_t sc2 = gi >> 7;
  const std::uint32_t kk = tile_row * kNpuTileK + sr * 8;
  const std::uint32_t nn = tile_col * kNpuTileN + (sc2 * 2 + block) * 8 + cib;
  float v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i)
    v[i] = __uint_as_float(static_cast<std::uint32_t>(w[static_cast<std::size_t>(nn) * k + kk + i]) << 16);
  EncodeGroup(v, out + g * 9);
}

/// Decodes a C tile stream (512 x 128 tiles) into F32 rows of `out`
/// (row pitch out_pitch, columns col0..col0+cols). f-gufo's kernel as-is.
__global__ void DecodeOutputKernel(const std::uint8_t* __restrict__ in,
                                   float* __restrict__ out,
                                   std::uint32_t tokens, std::uint32_t cols,
                                   std::size_t out_pitch, std::uint32_t col0,
                                   std::uint32_t valid_cols) {
  const std::size_t g = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (g >= std::size_t{tokens} * cols / 8) return;
  constexpr std::uint32_t kGroupsPerTile = kNpuTileRowsC * kNpuTileN / 8;  // 8192
  const std::size_t tile = g / kGroupsPerTile;
  const std::uint32_t gi = static_cast<std::uint32_t>(g % kGroupsPerTile);
  const std::uint32_t tiles_n = cols / kNpuTileN;
  const std::uint32_t tile_row = static_cast<std::uint32_t>(tile / tiles_n);
  const std::uint32_t tile_col = static_cast<std::uint32_t>(tile % tiles_n);
  const std::uint32_t rib = gi & 7;
  const std::uint32_t bc = (gi >> 3) & 1;
  const std::uint32_t br = (gi >> 4) & 1;
  const std::uint32_t sc = (gi >> 5) & 7;
  const std::uint32_t sr = gi >> 8;
  const std::size_t row =
      std::size_t{tile_row} * kNpuTileRowsC + (sr * 2 + br) * 8 + rib;
  const std::uint32_t col = tile_col * kNpuTileN + (sc * 2 + bc) * 8;
  if (valid_cols != 0 && col >= valid_cols) return;
  const std::uint8_t* p = in + g * 9;
  const float mult = ldexpf(1.0F, static_cast<int>(p[0]) - 127) / 64.0F;
  float* dst = out + row * out_pitch + col0 + col;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    dst[i] = static_cast<float>(static_cast<std::int8_t>(p[1 + i])) * mult;
  }
}

void EncodeActivations(const std::uint16_t* x, bool is_bf16, void* dst,
                       std::uint32_t tokens, std::size_t ldx, std::uint32_t k,
                       cudaStream_t stream) {
  const std::uint32_t tiles = (tokens / kNpuTileM) * (k / kNpuTileK);
  EncodeActivationsKernel<<<tiles, kNpuThreads, 0, stream>>>(
      x, is_bf16, tokens, ldx, k, static_cast<std::uint8_t*>(dst));
}

void EncodeWeightsBf16(const std::uint16_t* w, void* dst, std::uint32_t rows,
                       std::uint32_t k, cudaStream_t stream) {
  const std::size_t groups =
      std::size_t{k} * kNpuTileN / 8 * (rows / kNpuTileN);
  const std::uint32_t blocks = std::uint32_t((groups + kNpuThreads - 1) / kNpuThreads);
  EncodeWeightsBf16Kernel<<<blocks, kNpuThreads, 0, stream>>>(w,
      static_cast<std::uint8_t*>(dst), rows, k);
}

void DecodeOutput(const void* src, float* out, std::uint32_t tokens,
                  std::uint32_t cols, std::size_t out_pitch, std::uint32_t col0,
                  std::uint32_t valid_cols, cudaStream_t stream) {
  const std::size_t groups = std::size_t{tokens} * cols / 8;
  const std::uint32_t blocks = std::uint32_t((groups + kNpuThreads - 1) / kNpuThreads);
  DecodeOutputKernel<<<blocks, kNpuThreads, 0, stream>>>(
      static_cast<const std::uint8_t*>(src), out, tokens, cols, out_pitch,
      col0, valid_cols);
}

}  // namespace npu
}  // namespace talus
