// include/strata/core/dflash_drafter.hpp - Talus: the DFlash2 block-diffusion drafter (C++ host).
//
// A faithful transcription of the Rust DFlash host (rust/crates/talus-model/src/qwen35/graphs.rs,
// `DFlashModel` + `DFlash`), calling the same kernel launchers (src/qwen35/q36_dflash.cu via
// strata::q36::dflash::*) and the generic native MMVQ path directly. Built for the Flash-Next
// (qwen4exp) engine: the drafter GGUF is target-matched (dflash.embedding_length == the target's
// n_embd; dflash.target_layers name five target layers), and the head is the TARGET's output head
// (the shipped Flash-Next drafter carries no head of its own), projected through the draft
// vocabulary map like the MTP head.
//
// Lifecycle (mirrors qwen35/engine.rs):
//   init()          once: weights, buffers, target head, draft vocab map
//   wants_tap(li)   per target layer (li = 1..n_layer): does the drafter read this layer?
//   tap(li, x, T)   during the target forward: copy the layer's T hidden rows to tap_buf
//   inject(T, pos)  after each accepted token: project taps -> drafter KV ring
//   draft(anchor, pos, out)  per verify window: the block walk -> B-1 draft tokens
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// One quantized matrix in GGUF block form (row-major blocks, n_in contiguous).
struct DfMat {
    const void* d = nullptr;   // device
    int n_in = 0, n_out = 0;
    int ggml_type = 0;
    bool empty() const { return d == nullptr; }
};

class DFlashDrafter {
public:
    static constexpr int kTaps = 5;
    static constexpr int kTopK = 16;

    ~DFlashDrafter();

    /// `path`: the drafter GGUF. `target_head_*`: the TARGET model's output head (device blocks +
    /// ggml type, [vocab, n_embd]); the shipped Flash-Next drafter has no own head.
    /// `draft_map`: device i32[head_vocab] mapping drafter-vocab ids -> target-vocab ids, or null
    /// for identity (the full-vocabulary head). `max_batch`: rows the tap buffers must hold.
    bool init(const std::string& path, int target_n_layer, int target_d_model,
              const void* target_head_d, int target_head_ggml_type, int target_n_vocab,
              const int32_t* draft_map, int head_vocab, int max_batch, void* stream,
              std::string& err);

    bool ready() const { return ready_; }
    int n_draft() const { return block_size_ - 1; }
    int mask_token() const { return mask_token_; }
    int block() const { return block_size_; }
    bool wants_tap(int next_layer_1based) const;
    /// The tap slot (0..4) reading that layer, or -1.
    int tap_slot(int next_layer_1based) const;
    /// The persistent device tap-slot buffer j (the graph's D2D copy destination).
    float* tap_buf_ptr(int j) { return (j >= 0 && j < kTaps) ? tap_buf_[j] : nullptr; }
    /// Device scratch the host-draft wrapper reads the chain from.
    int32_t* draft_out() { return draft_out_; }

    /// Copy `T` rows of the target layer's hidden output (n_embd wide) into tap slot `next`.
    void tap(int next_layer_1based, const float* x_layer, int T, void* stream);

    /// Project the tapped rows [0, T) at device positions `pos_d` into the drafter's KV ring.
    void inject(int T, const int32_t* pos_d, void* stream);
    /// The same over host positions `pos0 + r0 .. pos0 + T` in ring-sized steps.
    void inject_at(int T, int64_t pos0, int r0, void* stream);

    /// One block-diffusion draft: `anchor_d` (device) the last committed token, `pos_d` its
    /// position; writes block_size_-1 draft token ids to `out_d` (device). Greedy walk.
    void draft(const void* embd_table, int embd_ggml_type, size_t embd_row_bytes,
               const int32_t* anchor_d, const int32_t* pos_d, int32_t* out_d, void* stream,
               const float* pair_dev = nullptr);

private:
    bool load_gguf(const std::string& path, int target_n_layer, int target_d_model,
                   std::string& err);
    void apply_mm(const DfMat& w, const float* in, float* out, int rows, void* stream);
    void apply_mm_preq8(const DfMat& w, float* out, int rows, void* stream);

    // model geometry
    int n_layers_ = 0, d_model_ = 0, n_heads_ = 0, n_kv_ = 0, head_dim_ = 128;
    int d_ff_ = 0, window_ = 2048, block_size_ = 8, conv_group_ = 16, mask_token_ = 0;
    float rope_theta_ = 1e4f, rms_eps_ = 1e-6f;
    int tap_layers_[kTaps] = {-1, -1, -1, -1, -1};
    int ring_len_ = 0, max_rows_ = 0;
    int n_vocab_ = 0;             // the head's vocabulary (target or mapped draft vocab)
    int head_vocab_ = 0;
    bool ready_ = false;

    // weights
    float *emb_norm_ = nullptr, *tap_norm_ = nullptr, *out_norm_ = nullptr;
    DfMat tap_proj_[kTaps];
    struct Layer {
        float *norm_q = nullptr, *norm_kv = nullptr, *q_norm = nullptr, *k_norm = nullptr;
        float *norm_ffn = nullptr;
        float *conv_base = nullptr, *ffn_conv_base = nullptr;
        DfMat wq, wk, wv, wo, w_dyn, w_ffn_dyn, w_gate, w_up, w_down;
    };
    std::vector<Layer> layers_;
    DfMat sel_hidden_, head_;     // head_: the target head (shared) in draft-vocab rows or full
    const float* trans_gate_ = nullptr;
    const void* trans_pred_ = nullptr, *trans_succ_ = nullptr;  // Q4_K [256, vocab]
    const int32_t* draft_map_ = nullptr;  // head_vocab -> target vocab (device)

    // buffers (device)
    float *x_ = nullptr, *h_q_ = nullptr, *h_kv_ = nullptr;
    float *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr, *att_part_ = nullptr;
    float *dyn_w_ = nullptr, *dyn_f_ = nullptr, *conv_out_ = nullptr;
    float *g_ = nullptr, *u_ = nullptr, *s_buf_ = nullptr, *ffn_out_ = nullptr;
    float *x_q8_ = nullptr;  // q8_1 scratch
    float *logits_ = nullptr, *cand_u_ = nullptr, *chain_sc_ = nullptr, *q_prop_ = nullptr;
    int32_t *cand_id_ = nullptr, *block_tok_ = nullptr, *pos_d_ = nullptr;
    int32_t* draft_out_ = nullptr;
    float *tap_buf_[kTaps] = {};
    float *ctx_part_[kTaps] = {};
    float* ctx_vec_ = nullptr;
    std::vector<float*> k_ring_, v_ring_;
    int tap_rows_ = 0;

    std::vector<void*> allocs_;
    void* tk_ = nullptr;  // kernels context for the q36 launchers (unused: launchers take raw args)
};

}  // namespace strata::core
