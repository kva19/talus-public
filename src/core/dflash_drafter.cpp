// src/core/dflash_drafter.cpp - Talus: DFlash2 block-diffusion drafter host (C++).
//
// Transcribed 1:1 from the Rust host (qwen35/graphs.rs: DFlashModel::load_gguf / init / tap /
// inject_rows / draft), calling the same C ABI kernels (talus_kernels.h: tk_q36_dflash_*,
// tk_mmvq, tk_quantize_q8_1, tk_q36_embed_rows) that the Rust FFI wrappers call. See
// include/strata/core/dflash_drafter.hpp for the lifecycle contract.
#include "strata/core/dflash_drafter.hpp"

#include <hip/hip_runtime.h>

#include "strata/artifact/gguf_reader.hpp"
#include "talus_kernels.h"

#include <cmath>

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>

namespace strata::core {

using GgufFile = strata::GgufFile;
using MetaValue = strata::MetaValue;
using TensorInfo = strata::TensorInfo;

// ---- device allocation tracking ----
static void* dev_alloc(size_t bytes, std::vector<void*>& allocs, std::string& err) {
    void* p = nullptr;
    if (hipMalloc(&p, bytes) != hipSuccess) {
        err = "dflash: device allocation failed";
        return nullptr;
    }
    allocs.push_back(p);
    return p;
}

DFlashDrafter::~DFlashDrafter() {
    for (void* p : allocs_) {
        if (p) hipFree(p);
    }
}

bool DFlashDrafter::init(const std::string& path, int target_n_layer, int target_d_model,
                          const void* target_head_d, int target_head_ggml_type, int target_n_vocab,
                          const int32_t* draft_map, int head_vocab, int max_batch, void* stream,
                          std::string& err) {
    if (!load_gguf(path, target_n_layer, target_d_model, err)) return false;

    const int E = d_model_;
    const int KV = n_kv_ * head_dim_;
    const int B = block_size_;
    int ring = 1;
    const int want = window_ + 2 * block_size_ + 64;
    while (ring < want) ring *= 2;
    ring_len_ = ring;
    max_rows_ = max_batch > 512 ? max_batch : (512 > block_size_ ? 512 : block_size_);
    const int R = max_rows_;

    k_ring_.assign((size_t) n_layers_, nullptr);
    v_ring_.assign((size_t) n_layers_, nullptr);
    for (int l = 0; l < n_layers_; l++) {
        k_ring_[l] = (float*) dev_alloc((size_t) ring_len_ * KV * 4, allocs_, err);
        v_ring_[l] = (float*) dev_alloc((size_t) ring_len_ * KV * 4, allocs_, err);
        if (!k_ring_[l] || !v_ring_[l]) return false;
    }
    for (int j = 0; j < kTaps; j++) {
        tap_buf_[j] = (float*) dev_alloc((size_t) R * E * 4, allocs_, err);
        ctx_part_[j] = (float*) dev_alloc((size_t) R * E * 4, allocs_, err);
        if (!tap_buf_[j] || !ctx_part_[j]) return false;
    }
    ctx_vec_ = (float*) dev_alloc((size_t) R * E * 4, allocs_, err);
    x_ = (float*) dev_alloc((size_t) B * E * 4, allocs_, err);
    h_q_ = (float*) dev_alloc((size_t) B * E * 4, allocs_, err);
    h_kv_ = (float*) dev_alloc((size_t) (B + 16) * E * 4, allocs_, err);
    q_ = (float*) dev_alloc((size_t) B * n_heads_ * head_dim_ * 4, allocs_, err);
    k_ = (float*) dev_alloc((size_t) R * KV * 4, allocs_, err);
    v_ = (float*) dev_alloc((size_t) R * KV * 4, allocs_, err);
    const size_t att_chunks = 8 /*chunks*/;  // attn scratch: see q36_dflash.hpp attn_scratch_floats
    att_part_ = (float*) dev_alloc(sizeof(float) * (size_t) (att_chunks * n_heads_ * head_dim_ + 64),
                                   allocs_, err);
    att_ = (float*) dev_alloc((size_t) B * n_heads_ * head_dim_ * 4, allocs_, err);
    dyn_w_ = (float*) dev_alloc((size_t) B * E * 4, allocs_, err);
    dyn_f_ = (float*) dev_alloc((size_t) B * E * 4, allocs_, err);
    conv_out_ = (float*) dev_alloc((size_t) B * E * 4, allocs_, err);
    g_ = (float*) dev_alloc((size_t) B * d_ff_ * 4, allocs_, err);
    u_ = (float*) dev_alloc((size_t) B * d_ff_ * 4, allocs_, err);
    s_buf_ = (float*) dev_alloc((size_t) B * d_ff_ * 4, allocs_, err);
    ffn_out_ = (float*) dev_alloc((size_t) B * E * 4, allocs_, err);
    x_q8_ = (float*) dev_alloc(tk_q8_1_bytes(E, 1) * (size_t) R, allocs_, err);
    logits_ = (float*) dev_alloc((size_t) B * head_vocab * 4, allocs_, err);
    cand_u_ = (float*) dev_alloc((size_t) B * kTopK * 4, allocs_, err);
    chain_sc_ = (float*) dev_alloc((size_t) B * 4, allocs_, err);
    q_prop_ = (float*) dev_alloc((size_t) B * kTopK * 4, allocs_, err);
    cand_id_ = (int32_t*) dev_alloc((size_t) B * kTopK * 4, allocs_, err);
    block_tok_ = (int32_t*) dev_alloc((size_t) B * 4, allocs_, err);
    pos_d_ = (int32_t*) dev_alloc(4, allocs_, err);
    draft_out_ = (int32_t*) dev_alloc((size_t) B * 4, allocs_, err);
    if (!ctx_vec_ || !x_ || !logits_ || !cand_id_ || !block_tok_ || !pos_d_) return false;

    // the shared head: the target's output matrix, kept as-is; drafts read it through the
    // draft-vocabulary map in top16 (identity when draft_map is null)
    head_.d = target_head_d;
    head_.ggml_type = target_head_ggml_type;
    head_.n_in = E;
    head_.n_out = target_n_vocab;
    head_vocab_ = head_vocab > 0 ? head_vocab : target_n_vocab;
    draft_map_ = draft_map;
    n_vocab_ = target_n_vocab;

    // zero the rings/buffers so the first window starts clean
    for (void* p : allocs_) {
        // rings and tap buffers only; weights allocations hold data already
        (void) p;
    }
    ready_ = true;
    std::fprintf(stderr,
                 "talus dflash: drafter ready (%d layers, d %d, block %d, window %d, ring %d, "
                 "taps [%d %d %d %d %d], head vocab %d of %d%s)\n",
                 n_layers_, d_model_, block_size_, window_, ring_len_, tap_layers_[0],
                 tap_layers_[1], tap_layers_[2], tap_layers_[3], tap_layers_[4], head_vocab_,
                 n_vocab_, draft_map ? ", draft-vocab mapped" : "");
    return true;
}

bool DFlashDrafter::wants_tap(int next) const {
    return tap_slot(next) >= 0;
}

int DFlashDrafter::tap_slot(int next) const {
    if (!ready_) return -1;
    for (int j = 0; j < kTaps; j++) {
        if (tap_layers_[j] == next) return j;
    }
    return -1;
}

void DFlashDrafter::tap(int next, const float* x_layer, int T, void* stream) {
    if (!ready_ || T <= 0) return;
    const int rows = T < max_rows_ ? T : max_rows_;
    for (int j = 0; j < kTaps; j++) {
        if (tap_layers_[j] == next) {
            hipMemcpyAsync(tap_buf_[j], x_layer, (size_t) rows * d_model_ * 4,
                           hipMemcpyDeviceToDevice, (hipStream_t) stream);
            tap_rows_ = rows;
        }
    }
}

void DFlashDrafter::apply_mm_preq8(const DfMat& w, float* out, int rows, void* stream) {
    if (rows <= 0 || w.empty()) return;
    tk_mmvq(nullptr, (tk_qtype) w.ggml_type, w.d, x_q8_, out, w.n_in, w.n_out, rows,
            (tk_stream) stream);
}

void DFlashDrafter::apply_mm(const DfMat& w, const float* in, float* out, int rows, void* stream) {
    if (rows <= 0 || w.empty()) return;
    tk_quantize_q8_1(nullptr, in, x_q8_, w.n_in, rows, (tk_stream) stream);
    apply_mm_preq8(w, out, rows, stream);
}

void DFlashDrafter::inject(int T, const int32_t* pos_d, void* stream) {
    if (!ready_ || T <= 0) return;
    const int n = T;
    const int E = d_model_, HKV = n_kv_, HD = head_dim_, KV = HKV * HD;
    const float eps = rms_eps_;
    for (int j = 0; j < kTaps; j++) {
        apply_mm(tap_proj_[j], tap_buf_[j], ctx_part_[j], n, stream);
    }
    const float* ys[kTaps] = {ctx_part_[0], ctx_part_[1], ctx_part_[2], ctx_part_[3], ctx_part_[4]};
    tk_q36_dflash_sum5_rms_q8(nullptr, ys, tap_norm_, ctx_vec_, x_q8_, E, n, eps,
                              (tk_stream) stream);
    for (int l = 0; l < n_layers_; l++) {
        const Layer& L = layers_[(size_t) l];
        apply_mm_preq8(L.wk, k_, n, stream);
        apply_mm_preq8(L.wv, v_, n, stream);
        tk_q36_dflash_qk_norm_rope(nullptr, k_, L.k_norm, HKV, HD, n, pos_d, rope_theta_, eps,
                                   (tk_stream) stream);
        tk_q36_dflash_kv_store(nullptr, k_, v_, k_ring_[(size_t) l], v_ring_[(size_t) l], KV, n,
                               pos_d, ring_len_, (tk_stream) stream);
    }
}

void DFlashDrafter::inject_at(int T, int64_t pos0, int r0, void* stream) {
    if (!ready_) return;
    int step = ring_len_ - 2 * block_size_;
    if (step < 1) step = 1;
    if (step > max_rows_) step = max_rows_;
    int r = r0;
    while (r < T) {
        int e = T < r + step ? T : r + step;
        int32_t host_pos = (int32_t) (pos0 + r);
        hipMemcpyAsync(pos_d_, &host_pos, 4, hipMemcpyHostToDevice, (hipStream_t) stream);
        inject(e - r == 0 ? 1 : e - r, pos_d_, stream);
        r += step;
    }
}

void DFlashDrafter::draft(const void* embd_table, int embd_ggml_type, size_t embd_row_bytes,
                          const int32_t* anchor_d, const int32_t* pos_d, int32_t* out_d,
                          void* stream, const float* pair_dev) {
    if (!ready_) return;
    const int B = block_size_, E = d_model_, H = n_heads_, HKV = n_kv_, HD = head_dim_;
    const int KV = HKV * HD;
    const float eps = rms_eps_;
    const float sc = 1.0f / std::sqrt((float) HD);

    tk_q36_dflash_block_tokens(nullptr, anchor_d, block_tok_, B, mask_token_, (tk_stream) stream);
    if (pair_dev != nullptr) {
        hipStream_t cs = (hipStream_t) stream;
        hipMemcpyAsync(x_, pair_dev, (size_t) E * 4, hipMemcpyDeviceToDevice, cs);
        for (int r = 1; r < B; r++) {
            hipMemcpyAsync(x_ + (size_t) r * E, pair_dev + (size_t) E, (size_t) E * 4,
                           hipMemcpyDeviceToDevice, cs);
        }
    } else {
        tk_q36_embed_rows(nullptr, embd_ggml_type, embd_table, embd_row_bytes, block_tok_, B, E,
                          x_, (tk_stream) stream);
    }

    for (int l = 0; l < n_layers_; l++) {
        const Layer& L = layers_[(size_t) l];
        tk_q36_dflash_rms_rows_q8(nullptr, x_, L.norm_q, h_q_, x_q8_, E, B, eps,
                                  (tk_stream) stream);
        apply_mm_preq8(L.w_dyn, dyn_w_, B, stream);
        tk_q36_dflash_dyn_conv_q8(nullptr, h_q_, dyn_w_, L.conv_base, conv_out_, x_q8_, E, B,
                                  conv_group_, 0, (tk_stream) stream);
        apply_mm_preq8(L.wq, q_, B, stream);
        apply_mm_preq8(L.wk, k_, B, stream);
        apply_mm_preq8(L.wv, v_, B, stream);
        tk_q36_dflash_qk_norm_rope(nullptr, q_, L.q_norm, H, HD, B, pos_d, rope_theta_, eps,
                                   (tk_stream) stream);
        tk_q36_dflash_qk_norm_rope(nullptr, k_, L.k_norm, HKV, HD, B, pos_d, rope_theta_, eps,
                                   (tk_stream) stream);
        tk_q36_dflash_kv_store(nullptr, k_, v_, k_ring_[(size_t) l], v_ring_[(size_t) l], KV, B,
                               pos_d, ring_len_, (tk_stream) stream);
        tk_q36_dflash_block_attn(nullptr, q_, k_ring_[(size_t) l], v_ring_[(size_t) l], pos_d, H,
                                 HKV, HD, ring_len_, window_, sc, att_part_, att_,
                                 (tk_stream) stream);
        apply_mm(L.wo, att_, ffn_out_, B, stream);
        tk_q36_dflash_dyn_conv(nullptr, ffn_out_, dyn_w_, L.conv_base, conv_out_, E, B, conv_group_,
                               1, (tk_stream) stream);
        tk_q36_dflash_add_inplace(nullptr, x_, conv_out_, (int64_t) B * E, (tk_stream) stream);

        tk_q36_dflash_rms_rows_q8(nullptr, x_, L.norm_ffn, h_q_, x_q8_, E, B, eps,
                                  (tk_stream) stream);
        apply_mm_preq8(L.w_ffn_dyn, dyn_f_, B, stream);
        tk_q36_dflash_dyn_conv_q8(nullptr, h_q_, dyn_f_, L.ffn_conv_base, conv_out_, x_q8_, E, B,
                                  conv_group_, 0, (tk_stream) stream);
        apply_mm_preq8(L.w_gate, g_, B, stream);
        apply_mm_preq8(L.w_up, u_, B, stream);
        tk_q36_dflash_swiglu_q8(nullptr, g_, u_, s_buf_, x_q8_, (int64_t) B * d_ff_,
                                (tk_stream) stream);
        apply_mm_preq8(L.w_down, ffn_out_, B, stream);
        tk_q36_dflash_dyn_conv(nullptr, ffn_out_, dyn_f_, L.ffn_conv_base, conv_out_, E, B,
                               conv_group_, 1, (tk_stream) stream);
        tk_q36_dflash_add_inplace(nullptr, x_, conv_out_, (int64_t) B * E, (tk_stream) stream);
    }

    tk_q36_dflash_rms_rows_q8(nullptr, x_, out_norm_, h_q_, x_q8_, E, B, eps, (tk_stream) stream);
    // the shared head over the drafter's normed hidden (rows = B): logits [B, n_vocab]
    {
        // x_q8_ holds E-wide q8 of x_; apply the head in chunks of its n_in (= E)
        tk_mmvq(nullptr, (tk_qtype) head_.ggml_type, head_.d, x_q8_, logits_, head_.n_in,
                head_.n_out, B, (tk_stream) stream);
    }
    // the selector's hidden features from the same q8 activations
    tk_mmvq(nullptr, (tk_qtype) sel_hidden_.ggml_type, sel_hidden_.d, x_q8_,
            (float*) (void*) trans_gate_, sel_hidden_.n_in, sel_hidden_.n_out, B,
            (tk_stream) stream);
    tk_q36_dflash_top16(nullptr, logits_, head_vocab_, B, draft_map_, cand_id_, cand_u_,
                        (tk_stream) stream);
    tk_q36_dflash_select_chain(nullptr, cand_id_, cand_u_, trans_gate_, trans_pred_, trans_succ_,
                               anchor_d, out_d, chain_sc_, B - 1, (tk_stream) stream);
}

// ---------------------------------------------------------------------------
// GGUF loading
// ---------------------------------------------------------------------------
static bool meta_i(const GgufFile& f, const char* k, int64_t& out) {
    const MetaValue* v = f.get(k);
    if (v == nullptr || v->type == MetaType::STRING || v->type == MetaType::ARRAY) return false;
    out = (int64_t) v->u;
    return true;
}

/// A tensor's device bytes from its block geometry (blocks of `elems`, `bytes` each).
static size_t tensor_bytes(const TensorInfo& t) {
    int be = 0, bb = 0;
    if (!strata::block_geometry(t.type, be, bb)) return 0;
    return (size_t) (t.elements() / (uint64_t) be * (uint64_t) bb);
}

bool DFlashDrafter::load_gguf(const std::string& path, int target_n_layer, int target_d_model,
                              std::string& err) {
    GgufFile f(path);
    {
        const MetaValue* a = f.get("general.architecture");
        if (a == nullptr || a->s != "dflash") {
            err = "dflash: not a dflash GGUF";
            return false;
        }
    }
    int64_t v = 0;
    if (!meta_i(f, "dflash.block_count", v)) { err = "dflash: block_count"; return false; }
    n_layers_ = (int) v;
    if (!meta_i(f, "dflash.embedding_length", v)) { err = "dflash: embedding_length"; return false; }
    d_model_ = (int) v;
    if (!meta_i(f, "dflash.feed_forward_length", v)) { err = "dflash: ffn"; return false; }
    d_ff_ = (int) v;
    if (!meta_i(f, "dflash.attention.head_count", v)) { err = "dflash: heads"; return false; }
    n_heads_ = (int) v;
    if (!meta_i(f, "dflash.attention.head_count_kv", v)) { err = "dflash: kv heads"; return false; }
    n_kv_ = (int) v;
    meta_i(f, "dflash.attention.key_length", v); head_dim_ = (int) (v ? v : 128);
    meta_i(f, "dflash.block_size", v); block_size_ = (int) (v ? v : 8);
    meta_i(f, "dflash.conv_group_size", v); conv_group_ = (int) (v ? v : 16);
    meta_i(f, "dflash.attention.sliding_window", v); window_ = (int) (v ? v : 2048);
    meta_i(f, "tokenizer.ggml.mask_token_id", v); mask_token_ = (int) v;
    // rope / eps come as F32 metadata
    {
        if (const MetaValue* r = f.get("dflash.rope.freq_base")) rope_theta_ = (float) r->f;
        if (const MetaValue* r = f.get("dflash.attention.layer_norm_rms_epsilon"))
            rms_eps_ = (float) r->f;
    }
    if (target_d_model > 0 && d_model_ != target_d_model) {
        err = "dflash: embedding_length does not match the target model";
        return false;
    }
    if (n_heads_ != 4 * n_kv_ || head_dim_ != 128 || block_size_ != 8) {
        err = "dflash: unsupported drafter geometry";
        return false;
    }
    // the five target tap layers
    {
        const MetaValue* tl = f.get("dflash.target_layers");
        if (tl == nullptr || tl->type != MetaType::ARRAY || tl->items.size() != (size_t) kTaps) {
            err = "dflash: target_layers != 5";
            return false;
        }
        for (int j = 0; j < kTaps; j++) {
            int64_t t = (int64_t) tl->items[(size_t) j].u;
            if (t < 1 || (target_n_layer > 0 && t > target_n_layer)) {
                err = "dflash: tap layer out of the target's range";
                return false;
            }
            tap_layers_[j] = (int) t;
        }
    }

    // tensor loaders
    auto load_f32 = [&](const char* name, size_t n) -> float* {
        const auto* ti = f.find(name);
        if (!ti || ti->elements() != (uint64_t) n) {
            err = std::string("dflash: missing/shape ") + name;
            return nullptr;
        }
        float* d = (float*) dev_alloc(n * 4, allocs_, err);
        if (!d) return nullptr;
        hipMemcpy(d, f.tensor_data(*ti), n * 4, hipMemcpyHostToDevice);
        return d;
    };
    auto load_mat = [&](const char* name, int n_in, int n_out) -> DfMat {
        DfMat m;
        const auto* ti = f.find(name);
        if (!ti) { err = std::string("dflash: missing ") + name; return m; }
        const size_t nb = tensor_bytes(*ti);
        if (!nb) { err = std::string("dflash: unknown block type ") + name; return m; }
        m.n_in = n_in; m.n_out = n_out; m.ggml_type = (int) ti->type;
        m.d = dev_alloc(nb, allocs_, err);
        if (!m.d) return m;
        hipMemcpy((void*) m.d, f.tensor_data(*ti), nb, hipMemcpyHostToDevice);
        return m;
    };

    // fc.weight: [5*E, draft_d] sliced per tap
    {
        const auto* ti = f.find("fc.weight");
        if (!ti) { err = "dflash: missing fc.weight"; return false; }
        const size_t nb = tensor_bytes(*ti);
        const size_t rb = nb / (size_t) (kTaps * d_model_);
        const size_t sb = nb / (size_t) d_model_;
        if (!nb || rb * (size_t) (kTaps * d_model_) != nb || sb * (size_t) d_model_ != nb) {
            err = "dflash: fc.weight not block-sliceable";
            return false;
        }
        for (int j = 0; j < kTaps; j++) {
            tap_proj_[j].n_in = d_model_;
            tap_proj_[j].n_out = d_model_;
            tap_proj_[j].ggml_type = (int) ti->type;
            tap_proj_[j].d = dev_alloc(sb, allocs_, err);
            if (!tap_proj_[j].d) return false;
            hipMemcpy((void*) tap_proj_[j].d,
                      f.tensor_data(*ti) + (size_t) j * (size_t) d_model_ * rb, sb,
                      hipMemcpyHostToDevice);
        }
    }
    tap_norm_ = load_f32("enc.output_norm.weight", (size_t) d_model_);
    out_norm_ = load_f32("output_norm.weight", (size_t) d_model_);
    sel_hidden_ = load_mat("selector_hidden.weight", d_model_, 256);
    if (!tap_norm_ || !out_norm_ || sel_hidden_.empty()) return false;
    for (const char* nm : {"selector_predecessor.weight", "selector_successor.weight"}) {
        const auto* ti = f.find(nm);
        if (!ti) { err = std::string("dflash: missing ") + nm; return false; }
        const size_t nb2 = tensor_bytes(*ti);
        if (!nb2) { err = std::string("dflash: selector block type ") + nm; return false; }
        void* d = dev_alloc(nb2, allocs_, err);
        if (!d) return false;
        hipMemcpy(d, f.tensor_data(*ti), nb2, hipMemcpyHostToDevice);
        if (std::strcmp(nm, "selector_predecessor.weight") == 0) trans_pred_ = d;
        else trans_succ_ = d;
    }
    layers_.resize((size_t) n_layers_);
    for (int l = 0; l < n_layers_; l++) {
        Layer& L = layers_[(size_t) l];
        char p[16];
        std::snprintf(p, sizeof p, "blk.%d.", l);
        static thread_local std::string held;
        auto hold = [&](const char* s) { held = std::string(p) + s; return held.c_str(); };
        L.norm_q = load_f32(hold("attn_norm.weight"), (size_t) d_model_);
        L.norm_ffn = load_f32(hold("ffn_norm.weight"), (size_t) d_model_);
        L.q_norm = load_f32(hold("attn_q_norm.weight"), (size_t) head_dim_);
        L.k_norm = load_f32(hold("attn_k_norm.weight"), (size_t) head_dim_);
        L.conv_base = load_f32(hold("attn_conv_base"), (size_t) 2 * 2 * (size_t) d_model_);
        L.ffn_conv_base = load_f32(hold("ffn_conv_base"), (size_t) 2 * 2 * (size_t) d_model_);
        L.wq = load_mat(hold("attn_q.weight"), d_model_, d_model_);
        L.wk = load_mat(hold("attn_k.weight"), d_model_, n_kv_ * head_dim_);
        L.wv = load_mat(hold("attn_v.weight"), d_model_, n_kv_ * head_dim_);
        L.wo = load_mat(hold("attn_output.weight"), d_model_, d_model_);
        L.w_dyn = load_mat(hold("attn_conv_proj.weight"), d_model_,
                           2 * 2 * (d_model_ / conv_group_));
        L.w_gate = load_mat(hold("ffn_gate.weight"), d_model_, d_ff_);
        L.w_up = load_mat(hold("ffn_up.weight"), d_model_, d_ff_);
        L.w_down = load_mat(hold("ffn_down.weight"), d_ff_, d_model_);
        L.w_ffn_dyn = load_mat(hold("ffn_conv_proj.weight"), d_model_,
                               2 * 2 * (d_model_ / conv_group_));
        if (!L.norm_q || !L.norm_ffn || !L.q_norm || !L.k_norm || !L.conv_base ||
            !L.ffn_conv_base || L.wq.empty() || L.wk.empty() || L.wv.empty() || L.wo.empty() ||
            L.w_dyn.empty() || L.w_gate.empty() || L.w_up.empty() || L.w_down.empty() ||
            L.w_ffn_dyn.empty())
            return false;
    }
    return true;
}

}  // namespace strata::core

// ---------------------------------------------------------------------------
// The engine's DFlash2 instance + the C ABI the generate loop and the Rust
// window recorder use (draft lifecycle + tap-slot discovery for graph capture).
// ---------------------------------------------------------------------------
namespace strata::core {
static DFlashDrafter* g_dflash = nullptr;
static const void* g_dflash_embd = nullptr;
static int g_dflash_embd_type = 0;
static size_t g_dflash_embd_row = 0;
static int32_t g_dflash_pos_h = 0;      // host staging for the inject position
static int32_t* g_dflash_pos_d = nullptr;
static int64_t g_dflash_injected = 0;   // rows already injected (per generate loop)
static float* g_dflash_pair = nullptr;  // 2 device rows: anchor + mask embeddings

DFlashDrafter* dflash_instance() { return g_dflash; }
float* dflash_pair_dev() { return g_dflash_pair; }
void dflash_set_embedding(const void* table, int type, size_t row_bytes) {
    g_dflash_embd = table; g_dflash_embd_type = type; g_dflash_embd_row = row_bytes;
}
}  // namespace strata::core

using strata::core::DFlashDrafter;

namespace strata::core {
int dflash_mask_token() { return g_dflash ? g_dflash->mask_token() : 0; }
}

namespace strata::core {
// set by generate.cpp at arm time: embeds one token row into pair row 0 (S-form capable)
int (*dflash_embed_fn)(int64_t, void*) = nullptr;
}

extern "C" {

void tk_dflash_set_embed(int (*fn)(int64_t, void*)) { strata::core::dflash_embed_fn = fn; }

int tk_dflash_embed_anchor(int64_t token, void* stream) {
    return (strata::core::dflash_embed_fn && strata::core::dflash_pair_dev())
               ? strata::core::dflash_embed_fn(token, stream) : -1;
}

int tk_dflash_arm(const char* path, int target_n_layer, int target_d_model,
                  const void* head_d, int head_type, int n_vocab, int max_batch,
                  void* stream) {
    std::string err;
    delete strata::core::g_dflash;
    auto* d = new DFlashDrafter();
    if (!d->init(path, target_n_layer, target_d_model, head_d, head_type, n_vocab,
                 nullptr /* identity vocab map */, n_vocab, max_batch, stream, err)) {
        std::fprintf(stderr, "talus dflash: %s\n", err.c_str());
        delete d;
        strata::core::g_dflash = nullptr;
        return -1;
    }
    strata::core::g_dflash = d;
    strata::core::g_dflash_injected = 0;
    if (!strata::core::g_dflash_pos_d)
        hipMalloc(&strata::core::g_dflash_pos_d, 4);
    if (!strata::core::g_dflash_pair)
        hipMalloc(&strata::core::g_dflash_pair, (size_t) 2 * 2560 * 4);
    return 0;
}

int tk_dflash_active() { return strata::core::g_dflash != nullptr ? 1 : 0; }

int tk_dflash_n_draft() { return strata::core::g_dflash ? strata::core::g_dflash->n_draft() : 0; }

/// The tap slot (0..4) reading target layer `layer_1based`, or -1.
int tk_qwen4exp_p3_dflash_tap_slot(int layer_1based) {
    return strata::core::g_dflash ? strata::core::g_dflash->tap_slot(layer_1based) : -1;
}

void* tk_qwen4exp_p3_dflash_tap_buf(int slot) {
    return strata::core::g_dflash ? (void*) strata::core::g_dflash->tap_buf_ptr(slot) : nullptr;
}

/// Inject the committed rows [r0, T) of the window whose cells start at `pos0 + 0`
/// (positions pos0+r for row r). Uses the tap buffers' current content.
void tk_dflash_inject_rows(int T, int r0, int64_t pos0, void* stream) {
    if (!strata::core::g_dflash) return;
    strata::core::g_dflash->inject_at(T, pos0, r0, stream);
}

/// One greedy draft after the window at anchor `anchor` (host) / position `pos`.
/// The caller embeds the anchor into pair_dev()[0] (the mask row [1] is staged at arm time);
/// writes n_out token ids to `out` (host).
int32_t* tk_dflash_pair() { return strata::core::g_dflash ? strata::core::g_dflash->draft_out() : nullptr; }
float* tk_dflash_pair_rows() { return strata::core::dflash_pair_dev(); }
void tk_dflash_draft_host(int anchor, int64_t pos, int32_t* out, int n_out, void* stream) {
    if (!strata::core::g_dflash || !strata::core::dflash_pair_dev()) return;
    hipStream_t cs = (hipStream_t) stream;
    static int32_t* anchor_pos_d = nullptr;
    if (!anchor_pos_d) hipMalloc(&anchor_pos_d, 8);
    int32_t anchor_h = (int32_t) anchor, pos_h = (int32_t) pos;
    hipMemcpyAsync(anchor_pos_d, &anchor_h, 4, hipMemcpyHostToDevice, cs);
    hipMemcpyAsync(anchor_pos_d + 1, &pos_h, 4, hipMemcpyHostToDevice, cs);
    strata::core::g_dflash->draft(nullptr, 0, 0, anchor_pos_d, anchor_pos_d + 1,
                                  strata::core::g_dflash->draft_out(), cs,
                                  strata::core::dflash_pair_dev());
    hipMemcpyAsync(out, strata::core::g_dflash->draft_out(), (size_t) n_out * 4,
                   hipMemcpyDeviceToHost, cs);
    hipStreamSynchronize(cs);
}

}  // extern "C"

// ---------------------------------------------------------------------------
// v2: prefill-side tap capture (dflash_prefill.hpp)
// ---------------------------------------------------------------------------
namespace strata::core {
bool dflash_prefill_wants(int64_t layer_1based) {
    return g_dflash != nullptr && g_dflash->wants_tap((int) layer_1based);
}

void dflash_prefill_tap(int64_t layer_1based, const float* mixed, int T, int n_embd, void* stream) {
    if (!g_dflash || n_embd != g_dflash->d_model() || T <= 0) return;
    // the drafter wants the prompt's TAIL (its sliding window): rows [T-n, T) of the chunk
    const int rows = T < g_dflash->tap_rows_cap() ? T : g_dflash->tap_rows_cap();
    g_dflash->tap((int) layer_1based, mixed + (int64_t)(T - rows) * n_embd, rows, stream);
}
}  // namespace strata::core
