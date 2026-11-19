// Wan 2.2 TI2V 5B (width 3072, 24 heads, 30 blocks, MLP 14336) wired from the shared building blocks.
// Tokens are video patches (frame, row, column), one stream. Each block: self-attention with rope and whole-width
// q / k norms, cross-attention to the prompt (keys / values made once per video), GELU MLP; every layer has a bias.
// Modulation = one projection of the noise level plus each block's own offset. Test runner like Klein's: one
// reference step (tools/ref_wan.py) and its error, then timed loops, or DRAW=1 for a whole video latent.
#pragma once
#include "ops.h"
#include <random>

namespace wan {
using namespace ops;
static const int NB = 30, Dm = 3072, NHEADS = 24, F = 14336, CTXD = 4096, PIN = 200;   // patch input: 192 + 1 + 7

static int run(const std::string &R, int loops) {
    int N = 0, Lt = 0, T = 0, Hp = 0, Wp = 0, PW = 0, PH = 0, FR = 0;
    if (FILE *f = fopen((R + "/info.txt").c_str(), "r")) {
        if (fscanf(f, "%d %d %d %d %d %d %d %d", &N, &Lt, &T, &Hp, &Wp, &PW, &PH, &FR) != 8) N = 0;
        fclose(f);
    }
    if (!N || N % 8 || Lt % 8) { fprintf(stderr, "%s/info.txt missing, bad, or sizes not multiples of 8\n", R.c_str()); return 1; }
    BIASES = 1; GELUT = 1; QKF = 1;
    ops::init(NHEADS, Dm, std::max(N, Lt), F);
    biasInit();
    printf("%dx%d, %d frames: %d tokens (%d x %d x %d), prompt %d, attention %d heads at a time\n", PW, PH, FR, N, T, Hp, Wp, Lt, HC);

    // ---- inputs ----
    auto imgv = rdf(R + "/img.f32"), ctxv = rdf(R + "/ctx.f32");
    auto Pt = rdf(R + "/rope_P.f32"), Qt = rdf(R + "/rope_Q.f32");
    std::vector<float> ia((size_t)N * PIN, 0.0f);
    auto fillPatches = [&](const std::vector<float> &x) {
        for (int n = 0; n < N; n++) {
            std::copy(x.begin() + (size_t)n * 192, x.begin() + (size_t)(n + 1) * 192, ia.begin() + (size_t)n * PIN);
            ia[(size_t)n * PIN + 192] = 1.0f;     // the bias channel
        }
    };
    fillPatches(imgv);
    cl_mem Ximg = toX(ia, N, PIN), Xctx = toX(ctxv, Lt, CTXD);
    cl_mem bP = buf(Pt.size() * 4, Pt.data()), bQ = buf(Qt.size() * 4, Qt.data());
    // image-to-video: <ref>/first.f32 is the picture's latent frame (Hp*Wp patches). Those tokens are frame 0, kept
    // clean, and use the modulation of noise level 0 while the rest use the current one.
    std::vector<float> first;
    if (FILE *ff = fopen((R + "/first.f32").c_str(), "rb")) { fclose(ff); first = rdf(R + "/first.f32"); }
    const int F0 = first.empty() ? 0 : Hp * Wp;
    if (F0) printf("image-to-video: frame 0 (%d tokens) is the picture\n", F0);
    float sigma = 0.0f;
    { auto s = rdf(R + "/sigma.f32"); sigma = s[0]; }

    // ---- activations ----
    cl_mem res = buf((size_t)N * Dm * 4);
    cl_mem QKV = img(N, 3 * Dm / 8), O = img(N, Dm / 8), Hh = img(N, F / 8), Yout = img(N, 192 / 8);
    cl_mem bTemb = buf(256 * 4), bT1 = buf(Dm * 4), bE = buf(Dm * 4), bE0 = buf(6 * Dm * 4);
    cl_mem mAll = buf((size_t)NB * 6 * Dm * 4), mHead = buf(2 * Dm * 4);
    cl_mem mAll0 = buf((size_t)NB * 6 * Dm * 4), mHead0 = buf(2 * Dm * 4);   // noise level 0 (image-to-video)
    std::vector<float> ones(Dm, 1.0f);
    cl_mem bOnes = buf(Dm * 4, ones.data());
    auto part = [&](cl_mem m, size_t i) { return sub(m, i * Dm * 4, Dm * 4); };
    std::vector<std::array<cl_mem, 6>> mods(NB), mods0(NB);
    for (int i = 0; i < NB; i++) for (int j = 0; j < 6; j++) {
        mods[i][j] = part(mAll, (size_t)i * 6 + j); mods0[i][j] = part(mAll0, (size_t)i * 6 + j);
    }
    cl_mem hSh = part(mHead, 0), hSc = part(mHead, 1), hSh0 = part(mHead0, 0), hSc0 = part(mHead0, 1);
    // per-block modulation offsets, kept on the processor side (added to the projection every step)
    std::vector<std::vector<float>> blockMod(NB);
    std::vector<float> headMod;
    {
        auto rd = [&](const std::string &n, size_t cnt) {
            std::vector<float> v(cnt);
            CK(clEnqueueReadBuffer(q, w(n), CL_TRUE, 0, cnt * 4, v.data(), 0, 0, 0));
            return v;
        };
        for (int i = 0; i < NB; i++) blockMod[i] = rd("b" + std::to_string(i) + ".mod", 6 * Dm);
        headMod = rd("head.mod", 2 * Dm);
    }

    // ---- the prompt: text layers, then every block's cross-attention keys / values, once per video ----
    std::vector<CrossKV> ckv(NB);
    {
        auto a = std::chrono::steady_clock::now();
        cl_mem C1 = img(Lt, Dm / 8), C2 = img(Lt, Dm / 8), KV = img(Lt, 2 * Dm / 8);
        linear(1, "txt1", Xctx, "txt1", Lt, Dm, CTXD, C1);
        linear(0, "txt2", C1, "txt2", Lt, Dm, Dm, C2);
        for (int i = 0; i < NB; i++) {
            const std::string b = "b" + std::to_string(i) + ".";
            streamer.use(i, NB);
            linear(0, "ckv", C2, b + "ckv", Lt, 2 * Dm, Dm, KV);
            ckv[i] = crossKV(KV, Lt, 0, Dm / 8, w(b + "ckn"));
        }
        clFinish(q);
        clReleaseMemObject(C1); clReleaseMemObject(C2); clReleaseMemObject(KV);
        printf("prompt prepared in %.0f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
    }

    auto check = [&](const char *what, const char *file) {
        std::vector<float> raw((size_t)N * Dm), out((size_t)N * Dm);
        CK(clEnqueueReadBuffer(q, res, CL_TRUE, 0, raw.size() * 4, raw.data(), 0, 0, 0));
        for (int og = 0; og < Dm / 4; og++) for (int n = 0; n < N; n++) for (int j = 0; j < 4; j++)
            out[(size_t)n * Dm + og * 4 + j] = raw[((size_t)og * N + n) * 4 + j];
        printf("  after %-12s error %.5f\n", what, relerr(out, rdf(R + "/" + file + ".f32")));
    };
    // noise level s: [cos, sin](1000 s * 10000^(-i/128)) -> time_embedding = e -> projection e0 (6 x width);
    // every block's modulation into mA, the head's into mH
    auto modsFor = [&](float s, cl_mem mA, cl_mem mH) {
        std::vector<float> te(256);
        for (int i = 0; i < 128; i++) {
            const float a = 1000.0f * s * expf(-logf(10000.0f) * i / 128.0f);
            te[i] = cosf(a); te[128 + i] = sinf(a);
        }
        CK(clEnqueueWriteBuffer(q, bTemb, CL_FALSE, 0, 256 * 4, te.data(), 0, 0, 0));
        gemv("gemv", w("t1"), bTemb, w("t1.b"), bT1, Dm, 256, 0);
        gemv("gemv", w("t2"), bT1, w("t2.b"), bE, Dm, Dm, 1);
        gemv("gemv", w("tp"), bE, w("tp.b"), bE0, 6 * Dm, Dm, 1);
        std::vector<float> e(Dm), e0(6 * Dm), all((size_t)NB * 6 * Dm), hm(2 * Dm);
        CK(clEnqueueReadBuffer(q, bE, CL_FALSE, 0, Dm * 4, e.data(), 0, 0, 0));
        CK(clEnqueueReadBuffer(q, bE0, CL_TRUE, 0, 6 * Dm * 4, e0.data(), 0, 0, 0));
        for (int i = 0; i < NB; i++) for (int k = 0; k < 6 * Dm; k++) all[(size_t)i * 6 * Dm + k] = e0[k] + blockMod[i][k];
        for (int k = 0; k < 2 * Dm; k++) hm[k] = headMod[k] + e[k % Dm];
        CK(clEnqueueWriteBuffer(q, mA, CL_FALSE, 0, all.size() * 4, all.data(), 0, 0, 0));
        CK(clEnqueueWriteBuffer(q, mH, CL_TRUE, 0, hm.size() * 4, hm.data(), 0, 0, 0));
    };
    if (F0) modsFor(0.0f, mAll0, mHead0);
    // token ranges with their modulation: [0, F0) at noise level 0, [F0, N) at the current one
    const int nr = F0 ? 2 : 1, r0[2] = {0, F0}, r1[2] = {F0 ? F0 : N, N};
    auto step = [&](bool chk) {
        modsFor(sigma, mAll, mHead);
        linear(3, "patch", Ximg, "patch", N, Dm, PIN, res);
        for (int i = 0; i < NB; i++) {
            const std::string b = "b" + std::to_string(i) + ".";
            streamer.use(i, NB);
            // shift, scale, gate (attention); shift, scale, gate (MLP)
            auto M_ = [&](int r) -> std::array<cl_mem, 6> & { return F0 && r == 0 ? mods0[i] : mods[i]; };
            for (int r = 0; r < nr; r++)
                normLinear(0, "qkv", res, M_(r)[0], M_(r)[1], b + "qkv", N, 3 * Dm, Dm, QKV, nullptr, r0[r], r1[r]);
            attnPrep(QKV, N, 0, N, 0, Dm / 8, 2 * Dm / 8, w(b + "qn"), w(b + "kn"), bP, bQ);
            attnCore(N, O);
            for (int r = 0; r < nr; r++) linear(2, "o", O, b + "o", N, Dm, Dm, res, M_(r)[2], r0[r], r1[r]);
            normLinear(0, "cq", res, w(b + "n3b"), w(b + "n3w"), b + "cq", N, Dm, Dm, QKV);
            crossAttn(QKV, N, 0, w(b + "cqn"), ckv[i], 0.0f, O);
            linear(2, "co", O, b + "co", N, Dm, Dm, res, bOnes);
            for (int r = 0; r < nr; r++)
                normLinear(1, "ff1", res, M_(r)[3], M_(r)[4], b + "ff1", N, F, Dm, Hh, nullptr, r0[r], r1[r]);
            for (int r = 0; r < nr; r++) linear(2, "ff2", Hh, b + "ff2", N, Dm, F, res, M_(r)[5], r0[r], r1[r]);
            if (chk && i == 0) check("block 0", "x_block0");
            if (chk && i == NB / 2) check("block 15", "x_block_mid");
            if (chk && i == NB - 1) check("block 29", "x_block_last");
        }
        for (int r = 0; r < nr; r++)
            normLinear(0, "head", res, F0 && r == 0 ? hSh0 : hSh, F0 && r == 0 ? hSc0 : hSc, "head", N, 192, Dm, Yout,
                       nullptr, r0[r], r1[r]);
    };
    auto readOut = [&]() {
        std::vector<h16> y((size_t)N * 192);
        size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)N, 24, 1};
        CK(clEnqueueReadImage(q, Yout, CL_TRUE, org, reg, 0, 0, y.data(), 0, 0, 0));
        std::vector<float> out((size_t)N * 192);
        for (int k8 = 0; k8 < 24; k8++) for (int n = 0; n < N; n++) for (int j = 0; j < 8; j++)
            out[(size_t)n * 192 + k8 * 8 + j] = y[((size_t)k8 * N + n) * 8 + j];
        return out;
    };

    // DRAW=1: the whole video latent from <ref>/noise.f32 to <ref>/draw.f32. Noise levels from SIGLIST=a,b,... (0
    // appended) or <ref>/sigmas.f32. The Turbo model is a few-step distill: each step predicts the clean video and
    // re-noises it with FRESH noise to the next level (ComfyUI's "lcm" sampler). Plain Euler (EULER=1) leaves blocky,
    // smeared patches with this model - that was the "blocky clip" of 2026-11-18.
    if (getenv("DRAW")) {
        auto x = rdf(R + "/noise.f32"), sg = rdf(R + "/sigmas.f32");
        if (getenv("SIGLIST")) {
            sg.clear();
            std::string t; for (char c : std::string(getenv("SIGLIST")) + ",") { if (c == ',') { if (!t.empty()) sg.push_back(std::stof(t)); t.clear(); } else t += c; }
            sg.push_back(0.0f);
        }
        const bool euler = getenv("EULER") != nullptr;
        std::mt19937 rng(getenv("SEED") ? atoi(getenv("SEED")) : 42);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::copy(first.begin(), first.end(), x.begin());
        double total = 0;
        for (size_t si = 0; si + 1 < sg.size(); si++) {
            sigma = sg[si];
            fillPatches(x);
            clReleaseMemObject(Ximg); Ximg = toX(ia, N, PIN);
            clFinish(q); auto a = std::chrono::steady_clock::now();
            step(false);
            auto v = readOut();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
            total += ms;
            if (euler) for (size_t k = 0; k < x.size(); k++) x[k] += (sg[si + 1] - sg[si]) * v[k];
            else {
                const float s0 = sg[si], s1 = sg[si + 1];
                for (size_t k = 0; k < x.size(); k++) {
                    const float x0 = x[k] - s0 * v[k];                   // the predicted clean video
                    x[k] = s1 > 0 ? (1 - s1) * x0 + s1 * nd(rng) : x0;
                }
            }
            std::copy(first.begin(), first.end(), x.begin());     // frame 0 stays the picture
            printf("draw step %zu: sigma %.3f -> %.3f, %.1f ms\n", si, sg[si], sg[si + 1], ms); fflush(stdout);
        }
        FILE *f = fopen((R + "/draw.f32").c_str(), "wb");
        fwrite(x.data(), 4, x.size(), f); fclose(f);
        printf("drawing: %.1f s of steps, latent in %s/draw.f32\n", total / 1000, R.c_str());
        return 0;
    }
    step(true);
    printf("  final output: error %.5f\n", relerr(readOut(), rdf(R + "/out.f32")));
    tsum.clear();
    clFinish(q);
    for (int l = 0; l < loops; l++) {
        auto a = std::chrono::steady_clock::now();
        step(false);
        clFinish(q);
        printf("step %d: %.1f ms\n", l, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
    }
    if (prof) {
        double tot = 0; for (auto &kv : tsum) tot += kv.second;
        for (auto &kv : tsum) printf("  %-12s %8.1f ms  %4.1f%%\n", kv.first.c_str(), kv.second / loops, 100 * kv.second / tot);
    }
    return 0;
}

}  // namespace wan
