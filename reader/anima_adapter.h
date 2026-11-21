// Anima's prompt adapter (ComfyUI comfy/ldm/anima/model.py LLMAdapter) on the processor, for the graphics-chip
// route: the T5 word rows read the Qwen3-0.6B reader's rows through 6 blocks (self-attention, cross-attention, MLP,
// each after an RMS norm), then out_proj + RMS norm, each row times its emphasis weight. Same maths as the AI-chip
// export (gpu-engine-test/tsbuild/anima/anima_text.py) without its fixed slots: only the real rows are computed,
// which is what that export's masks amount to. Weights: tools/pack_anima_text.py (bf16, prefix removed).
#pragma once
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace anima_adapter {

static const int H = 1024, NH = 16, HD = 64, NT = 256, NB = 6;

struct Weights {
    std::map<std::string, std::vector<float>> t;
    const std::vector<float> &operator()(const std::string &n) const {
        auto it = t.find(n);
        if (it == t.end()) { fprintf(stderr, "adapter weight %s missing\n", n.c_str()); exit(1); }
        return it->second;
    }
};

// every tensor except the T5 word table (read row by row on demand)
static bool load(const char *path, Weights *w, std::vector<uint16_t> *embed) {
    gguf_init_params ip = {true, nullptr};
    gguf_context *g = gguf_init_from_file(path, ip);
    if (!g) { fprintf(stderr, "cannot read %s\n", path); return false; }
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    const size_t base = gguf_get_data_offset(g);
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        const std::string n = gguf_get_tensor_name(g, i);
        if (gguf_get_tensor_type(g, i) != GGML_TYPE_BF16) { fprintf(stderr, "%s is not bf16\n", n.c_str()); return false; }
        const size_t cnt = gguf_get_tensor_size(g, i) / 2;
        std::vector<uint16_t> raw(cnt);
        fseek(f, (long)(base + gguf_get_tensor_offset(g, i)), SEEK_SET);
        if (fread(raw.data(), 2, cnt, f) != cnt) { fprintf(stderr, "short read in %s\n", n.c_str()); return false; }
        if (n == "embed.weight") { *embed = std::move(raw); continue; }
        auto &v = w->t[n]; v.resize(cnt);
        for (size_t k = 0; k < cnt; k++) { uint32_t u = (uint32_t)raw[k] << 16; memcpy(&v[k], &u, 4); }
    }
    fclose(f); gguf_free(g);
    return true;
}

static void parallel(int threads, int n, const std::function<void(int, int)> &fn) {
    std::vector<std::thread> ts;
    const int per = (n + threads - 1) / threads;
    for (int t = 0; t < threads; t++) {
        const int a = t * per, b = std::min(n, a + per);
        if (a < b) ts.emplace_back(fn, a, b);
    }
    for (auto &t : ts) t.join();
}

// Y (T x O) = X (T x I) . W^T (+ bias); W is O x I, row-major
static std::vector<float> linear(int threads, const std::vector<float> &X, int T, int I, const std::vector<float> &W, int O,
                                 const std::vector<float> *bias = nullptr) {
    std::vector<float> Y((size_t)T * O);
    parallel(threads, O, [&](int a, int b) {
        for (int o = a; o < b; o++) {
            const float *w = &W[(size_t)o * I];
            for (int t = 0; t < T; t++) {
                const float *x = &X[(size_t)t * I];
                float s = 0; for (int k = 0; k < I; k++) s += x[k] * w[k];
                Y[(size_t)t * O + o] = s + (bias ? (*bias)[o] : 0.0f);
            }
        }
    });
    return Y;
}

static void rms(std::vector<float> &X, int T, int D, const std::vector<float> &w, float eps) {
    for (int t = 0; t < T; t++) {
        float *x = &X[(size_t)t * D];
        double ss = 0; for (int k = 0; k < D; k++) ss += (double)x[k] * x[k];
        const float r = 1.0f / sqrtf((float)(ss / D) + eps);
        for (int k = 0; k < D; k++) x[k] = x[k] * r * w[k];
    }
}

static std::vector<float> rmsCopy(const std::vector<float> &X, int T, int D, const std::vector<float> &w, float eps) {
    std::vector<float> y = X; rms(y, T, D, w, eps); return y;
}

// per head: RMS norm over 64 (q_norm / k_norm), then rope on position p: x*cos + rotate_half(x)*sin, theta 10000
static void headNormRope(std::vector<float> &X, int T, const std::vector<float> &nw) {
    for (int t = 0; t < T; t++)
        for (int h = 0; h < NH; h++) {
            float *x = &X[(size_t)t * H + h * HD];
            double ss = 0; for (int k = 0; k < HD; k++) ss += (double)x[k] * x[k];
            const float r = 1.0f / sqrtf((float)(ss / HD) + 1e-6f);
            for (int k = 0; k < HD; k++) x[k] *= r * nw[k];
            float y[HD];
            for (int k = 0; k < HD / 2; k++) {
                const float a = t * powf(10000.0f, -(2.0f * k) / HD), c = cosf(a), s = sinf(a);
                y[k] = x[k] * c - x[k + HD / 2] * s;
                y[k + HD / 2] = x[k + HD / 2] * c + x[k] * s;
            }
            memcpy(x, y, sizeof y);
        }
}

// attention of nq rows over nk rows; prefix "self_attn." or "cross_attn."
static std::vector<float> attn(int threads, const Weights &w, const std::string &p, const std::vector<float> &xq, int nq,
                               const std::vector<float> &src, int nk) {
    auto q = linear(threads, xq, nq, H, w(p + "q_proj.weight"), H);
    auto k = linear(threads, src, nk, H, w(p + "k_proj.weight"), H);
    auto v = linear(threads, src, nk, H, w(p + "v_proj.weight"), H);
    headNormRope(q, nq, w(p + "q_norm.weight"));
    headNormRope(k, nk, w(p + "k_norm.weight"));
    std::vector<float> o((size_t)nq * H, 0.0f);
    parallel(threads, NH, [&](int a, int b) {
        std::vector<float> s(nk);
        for (int h = a; h < b; h++)
            for (int i = 0; i < nq; i++) {
                float m = -1e30f;
                for (int j = 0; j < nk; j++) {
                    float d = 0; for (int c = 0; c < HD; c++) d += q[(size_t)i * H + h * HD + c] * k[(size_t)j * H + h * HD + c];
                    s[j] = d * 0.125f; m = std::max(m, s[j]);
                }
                double sum = 0; for (int j = 0; j < nk; j++) { s[j] = expf(s[j] - m); sum += s[j]; }
                float *out = &o[(size_t)i * H + h * HD];
                for (int j = 0; j < nk; j++) {
                    const float pj = (float)(s[j] / sum);
                    for (int c = 0; c < HD; c++) out[c] += pj * v[(size_t)j * H + h * HD + c];
                }
            }
    });
    return linear(threads, o, nq, H, w(p + "o_proj.weight"), H);
}

static float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678f)); }

// ctx: NT x H, rows past the T5 prompt zero (as the AI-chip route's keep)
static std::vector<float> run(int threads, const Weights &w, const std::vector<uint16_t> &embed,
                              const std::vector<float> &qwen, int tq, const std::vector<int32_t> &t5, const std::vector<float> &t5w) {
    const int tt = (int)t5.size();
    std::vector<float> y((size_t)tt * H);
    for (int t = 0; t < tt; t++)
        for (int k = 0; k < H; k++) { uint32_t u = (uint32_t)embed[(size_t)t5[t] * H + k] << 16; memcpy(&y[(size_t)t * H + k], &u, 4); }
    for (int b = 0; b < NB; b++) {
        const std::string p = "blocks." + std::to_string(b) + ".";
        auto h = rmsCopy(y, tt, H, w(p + "norm_self_attn.weight"), 1e-6f);
        auto a = attn(threads, w, p + "self_attn.", h, tt, h, tt);
        for (size_t i = 0; i < y.size(); i++) y[i] += a[i];
        h = rmsCopy(y, tt, H, w(p + "norm_cross_attn.weight"), 1e-6f);
        a = attn(threads, w, p + "cross_attn.", h, tt, qwen, tq);
        for (size_t i = 0; i < y.size(); i++) y[i] += a[i];
        h = rmsCopy(y, tt, H, w(p + "norm_mlp.weight"), 1e-6f);
        auto m = linear(threads, h, tt, H, w(p + "mlp.0.weight"), 4 * H, &w(p + "mlp.0.bias"));
        for (auto &x : m) x = gelu(x);
        a = linear(threads, m, tt, 4 * H, w(p + "mlp.2.weight"), H, &w(p + "mlp.2.bias"));
        for (size_t i = 0; i < y.size(); i++) y[i] += a[i];
    }
    auto o = linear(threads, y, tt, H, w("out_proj.weight"), H, &w("out_proj.bias"));
    rms(o, tt, H, w("norm.weight"), 1e-6f);
    std::vector<float> ctx((size_t)NT * H, 0.0f);
    for (int t = 0; t < tt; t++)
        for (int k = 0; k < H; k++) ctx[(size_t)t * H + k] = o[(size_t)t * H + k] * t5w[t];
    return ctx;
}

}  // namespace anima_adapter
