// Wan 2.2 TI2V 5B on Vulkan (width 3072, 24 heads, 30 blocks, MLP 14336), the same test folders as the OpenCL engine
// (model_wan.h). Tokens are video patches, one stream. Each block: self-attention with rope and whole-width q / k norms,
// cross-attention to the prompt (keys / values made once per video), tanh-GELU MLP; every layer has a bias.
// Modulation = one projection of the noise level plus each block's own offset. Image-to-video (first.f32): frame 0's
// tokens are the picture, kept clean, with the modulation of noise level 0. Attention rows are rounded up to 128 (the
// extra rows stay zero and are masked); so are the prompt's.   env: DRAW=1 (whole video latent), SIGLIST, EULER, SEED
#pragma once
#include "vkops.h"
#include <random>

namespace wan {
static const int NB = 30, Dm = 3072, HEADS = 24, F = 14336, CTXD = 4096, PIN = 200;   // patch input: 192 + 1 + 7

static int run(const std::string &Rd, int loops) {
    if (autoDeclines("b0.qkv")) return 3;
    // the blocks' tables are read from storage block by block (the prompt's "ckv" tables are used once, so stay)
    const bool stream = streamWanted();
    if (stream) streamOf = [](const std::string &n) {
        int b; char suf[32];
        return sscanf(n.c_str(), "b%d.%31s", &b, suf) == 2 && strcmp(suf, "ckv") ? b : -1;
    };
    auto t0 = std::chrono::steady_clock::now();
    int N = 0, Lt = 0, T = 0, Hp = 0, Wp = 0, PW = 0, PH = 0, FR = 0;
    if (FILE *f = fopen((Rd + "/info.txt").c_str(), "r")) {
        if (fscanf(f, "%d %d %d %d %d %d %d %d", &N, &Lt, &T, &Hp, &Wp, &PW, &PH, &FR) != 8) N = 0;
        fclose(f);
    }
    if (!N) { fprintf(stderr, "%s/info.txt missing or bad\n", Rd.c_str()); return 1; }
    const int Na = pad128(N), Lp = pad128(Lt), hc = headsPerPass(HEADS, Na), hx = headsPerPass(HEADS, std::max(Na, Lp));
    printf("%dx%d, %d frames: %d tokens (%d x %d x %d), prompt %d, attention over %d rows, %d heads at a time\n",
           PW, PH, FR, N, T, Hp, Wp, Lt, Na, hc);

    // ---- inputs ----
    auto imgv = rdf(Rd + "/img.f32"), ctxv = rdf(Rd + "/ctx.f32");
    auto Pt = rdf(Rd + "/rope_P.f32"), Qt = rdf(Rd + "/rope_Q.f32");
    if (imgv.size() != (size_t)N * 192 || ctxv.size() < (size_t)Lt * CTXD || Pt.size() < (size_t)N * 128) { fprintf(stderr, "bad input sizes\n"); return 1; }
    Buf bP = up(Pt.data(), (size_t)N * 512), bQ = up(Qt.data(), (size_t)N * 512);
    Buf Xp = vk::buf((size_t)N * PIN * 2), Xctx = vk::buf((size_t)Lt * CTXD * 2);
    auto fillPatches = [&](const std::vector<float> &x) {
        h16 *d = (h16 *)Xp.map;
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < 192; c++) d[(size_t)n * PIN + c] = (h16)x[(size_t)n * 192 + c];
            d[(size_t)n * PIN + 192] = (h16)1.0f;                     // the bias channel
            for (int c = 193; c < PIN; c++) d[(size_t)n * PIN + c] = (h16)0.0f;
        }
    };
    fillPatches(imgv);
    putH(Xctx, ctxv.data(), (size_t)Lt * CTXD);
    std::vector<float> first;
    if (fexists(Rd + "/first.f32")) first = rdf(Rd + "/first.f32");
    const int F0 = first.empty() ? 0 : Hp * Wp;
    if (F0) printf("image-to-video: frame 0 (%d tokens) is the picture\n", F0);
    float sigma = rdf(Rd + "/sigma.f32")[0];

    // ---- buffers ----
    auto zeroed = [](size_t bytes) { Buf b = vk::buf(bytes); memset(b.map, 0, bytes); return b; };
    Buf res = vk::buf((size_t)N * Dm * 4), QKV = vk::buf((size_t)N * 3 * Dm * 2), O = vk::buf((size_t)Na * Dm * 2);
    Buf Hh = vk::buf((size_t)N * F * 2), Qc = vk::buf((size_t)N * Dm * 2), Yout = vk::buf((size_t)N * 192 * 2);
    Buf RSq = vk::buf((size_t)Na * 4), RSk = vk::buf((size_t)std::max(Na, Lp) * 4);
    Buf Qh = zeroed((size_t)HEADS * Na * 256), Kh = zeroed((size_t)HEADS * Na * 256), Vh = zeroed((size_t)HEADS * Na * 256);
    Buf S = vk::buf(std::max((size_t)hc * Na * Na, (size_t)hx * Na * Lp) * 2);
    Buf C1 = vk::buf((size_t)Lt * Dm * 2), C2 = vk::buf((size_t)Lt * Dm * 2), KV = vk::buf((size_t)Lt * 2 * Dm * 2);
    std::vector<Buf> CK(NB), CV(NB);
    for (int i = 0; i < NB; i++) { CK[i] = zeroed((size_t)HEADS * Lp * 256); CV[i] = zeroed((size_t)HEADS * Lp * 256); }
    scratch(std::max(N, Lt), F);
    Buf bTemb = vk::buf(256 * 4), bT1 = vk::buf(Dm * 4), bE = vk::buf(Dm * 4), bE0 = vk::buf(6 * Dm * 4);
    Buf mAll = vk::buf((size_t)NB * 6 * Dm * 4), mHead = vk::buf(2 * Dm * 4), mAll0 = vk::buf((size_t)NB * 6 * Dm * 4), mHead0 = vk::buf(2 * Dm * 4);
    Buf bOnes = vk::buf(Dm * 4);
    for (int k = 0; k < Dm; k++) ((float *)bOnes.map)[k] = 1.0f;
    auto part = [&](Buf &m, size_t i) { return R(m, i * Dm * 4, Dm * 4); };
    std::vector<Buf> snap(3);
    for (auto &s : snap) s = vk::buf((size_t)N * Dm * 4);
    const float qscale = 1.0f / sqrtf(128.0f);

    // ---- graph 1: the prompt, once per video: text layers, then every block's cross-attention keys / values ----
    vk::Graph gctx; G = &gctx;
    rotX("txt1", R(Xctx), CTXD, CTXD, Lt);
    mulq(6, "txt1", "txt1", R(C1), R(dummy), R(dummy), Lt);
    rotX("txt2", R(C1), Dm, Dm, Lt);
    mulq(0, "txt2", "txt2", R(C2), R(dummy), R(dummy), Lt);
    rotX("b0.ckv", R(C2), Dm, Dm, Lt);   // one rounding of the prompt feeds every block's ckv (all have the same blocks)
    for (int i = 0; i < NB; i++) {
        const std::string b = "b" + std::to_string(i) + ".";
        if (q8(b + "ckv").GS != q8("b0.ckv").GS) { fprintf(stderr, "ckv layers with different scale blocks\n"); return 1; }
        mulq(0, "ckv", b + "ckv", R(KV), R(dummy), R(dummy), Lt);
        rowrms(R(KV), 2 * Dm, 0, Dm, R(RSk), Lt);
        prep(R(KV), 2 * Dm, 0, R(wf(b + "ckn")), R(dummy), R(dummy), R(CK[i]), Lt, Lp, HEADS, true, false, 1.0f, "prep ck", R(RSk));
        prep(R(KV), 2 * Dm, Dm, R(dummy), R(dummy), R(dummy), R(CV[i]), Lt, Lp, HEADS, false, false, 1.0f, "prep cv");
    }
    // ---- graph 2: the noise level's projection ----
    vk::Graph gmod; G = &gmod;
    gemv(wh("t1"), R(bTemb), R(wf("t1.b")), R(bT1), Dm, 256, 0);
    gemv(wh("t2"), R(bT1), R(wf("t2.b")), R(bE), Dm, Dm, 1);
    gemv(wh("tp"), R(bE), R(wf("tp.b")), R(bE0), 6 * Dm, Dm, 1);
    // ---- graph 3: one step ----
    // token ranges with their modulation: [0, F0) at noise level 0, [F0, N) at the current one
    const int nr = F0 ? 2 : 1, r0[2] = {0, F0}, r1[2] = {F0 ? F0 : N, N};
    const bool check = !getenv("DRAW");
    vk::Graph gstep; G = &gstep;
    mulh(3, "patch", 0, R(Xp), R(wh("patch")), R(dummy), R(res), N, Dm, PIN, PIN, PIN, Dm);
    for (int i = 0; i < NB; i++) {
        const std::string b = "b" + std::to_string(i) + ".";
        if (i) G->mark();   // segment i: block i
        auto M_ = [&](int r, int j) { return part(F0 && r == 0 ? mAll0 : mAll, (size_t)i * 6 + j); };   // shift, scale, gate x2
        auto rows = [&](int r, Buf &bf, size_t rowBytes) { return R(bf, (size_t)r0[r] * rowBytes); };
        for (int r = 0; r < nr; r++) {
            rotLn(b + "qkv", rows(r, res, Dm * 4), M_(r, 0), M_(r, 1), Dm, r1[r] - r0[r]);
            mulq(0, "qkv", b + "qkv", rows(r, QKV, 3 * Dm * 2), R(dummy), R(dummy), r1[r] - r0[r]);
        }
        rowrms(R(QKV), 3 * Dm, 0, Dm, R(RSq), N);
        rowrms(R(QKV), 3 * Dm, Dm, Dm, R(RSk), N);
        prep(R(QKV), 3 * Dm, 0, R(wf(b + "qn")), R(bP), R(bQ), R(Qh), N, Na, HEADS, true, true, qscale, "prep q", R(RSq));
        prep(R(QKV), 3 * Dm, Dm, R(wf(b + "kn")), R(bP), R(bQ), R(Kh), N, Na, HEADS, true, true, 1.0f, "prep k", R(RSk));
        prep(R(QKV), 3 * Dm, 2 * Dm, R(dummy), R(bP), R(bQ), R(Vh), N, Na, HEADS, false, false, 1.0f, "prep v");
        attend(Qh, Kh, Vh, S, R(O), Na, N, HEADS, hc, Dm);
        for (int r = 0; r < nr; r++) {
            rotX(b + "o", rows(r, O, Dm * 2), Dm, Dm, r1[r] - r0[r]);
            mulq(2, "o", b + "o", R(dummy), rows(r, res, Dm * 4), M_(r, 2), r1[r] - r0[r]);
        }
        // cross-attention: norm3 (affine), q with its whole-width norm, against the prompt's keys / values
        rotLn(b + "cq", R(res), R(wf(b + "n3b")), R(wf(b + "n3w")), Dm, N);
        mulq(0, "cq", b + "cq", R(Qc), R(dummy), R(dummy), N);
        rowrms(R(Qc), Dm, 0, Dm, R(RSq), N);
        prep(R(Qc), Dm, 0, R(wf(b + "cqn")), R(dummy), R(dummy), R(Qh), N, Na, HEADS, true, false, qscale, "prep cq", R(RSq));
        for (int h0 = 0; h0 < HEADS; h0 += hx) {
            const int nh = std::min(hx, HEADS - h0);
            mula("qk cross", 0, R(Qh, (size_t)h0 * Na * 256), R(CK[i], (size_t)h0 * Lp * 256), R(S), Na, Lp, 128, 128, 128, Lp, Na * 128, Lp * 128, Na * Lp, nh);
            softmax(R(S), Na, Lp, Lt, 0.0f, nh);
            mula("pv cross", 1, R(S), R(CV[i], (size_t)h0 * Lp * 256), R(O, (size_t)h0 * 256), Na, 128, Lp, Lp, 128, Dm, Na * Lp, Lp * 128, 128, nh);
        }
        rotX(b + "co", R(O), Dm, Dm, N);
        mulq(2, "co", b + "co", R(dummy), R(res), R(bOnes), N);
        for (int r = 0; r < nr; r++) {
            rotLn(b + "ff1", rows(r, res, Dm * 4), M_(r, 3), M_(r, 4), Dm, r1[r] - r0[r]);
            mulq(6, "ff1", b + "ff1", rows(r, Hh, F * 2), R(dummy), R(dummy), r1[r] - r0[r]);
        }
        for (int r = 0; r < nr; r++) {
            rotX(b + "ff2", rows(r, Hh, F * 2), F, F, r1[r] - r0[r]);
            mulq(2, "ff2", b + "ff2", R(dummy), rows(r, res, Dm * 4), M_(r, 5), r1[r] - r0[r]);
        }
        if (check && i == 0) copyf(R(res), R(snap[0]), (size_t)N * Dm);
        if (check && i == NB / 2) copyf(R(res), R(snap[1]), (size_t)N * Dm);
        if (check && i == NB - 1) copyf(R(res), R(snap[2]), (size_t)N * Dm);
    }
    for (int r = 0; r < nr; r++) {
        Buf &mh = F0 && r == 0 ? mHead0 : mHead;
        rotLn("head", R(res, (size_t)r0[r] * Dm * 4), part(mh, 0), part(mh, 1), Dm, r1[r] - r0[r]);
        mulq(0, "head", "head", R(Yout, (size_t)r0[r] * 192 * 2), R(dummy), R(dummy), r1[r] - r0[r]);
    }
    vk::savePipelines();
    printf("setup: %.1f s rearranging weights, %.1f s building shaders\n", tWeights, tKern);
    printf("loaded and recorded in %.1f s: %.0f MB of weights on the chip, %zu dispatches per step\n",
           secsSince(t0), chipBytes / 1e6, gstep.ops.size());
    if (stream) printf("streaming the blocks from storage: %.0f MB of slots\n", slotBytes / 1e6);

    // per-block modulation offsets, added to the projection on the processor side every step
    std::vector<const float *> blockMod(NB);
    for (int i = 0; i < NB; i++) blockMod[i] = (const float *)ft("b" + std::to_string(i) + ".mod").src;
    const float *headMod = (const float *)ft("head.mod").src;
    auto modsFor = [&](float s, Buf &mA, Buf &mH) {
        float *te = (float *)bTemb.map;
        for (int i = 0; i < 128; i++) {
            const float a = 1000.0f * s * expf(-logf(10000.0f) * i / 128.0f);
            te[i] = cosf(a); te[128 + i] = sinf(a);
        }
        gmod.run();
        const float *e = (const float *)bE.map, *e0 = (const float *)bE0.map;
        float *all = (float *)mA.map, *hm = (float *)mH.map;
        for (int i = 0; i < NB; i++) for (int k = 0; k < 6 * Dm; k++) all[(size_t)i * 6 * Dm + k] = e0[k] + blockMod[i][k];
        for (int k = 0; k < 2 * Dm; k++) hm[k] = headMod[k] + e[k % Dm];
    };
    {
        auto a = std::chrono::steady_clock::now();
        gctx.run();
        printf("prompt prepared in %.0f ms\n", secsSince(a) * 1000);
        const size_t freed = dropTables([](const std::string &n) {
            return n == "txt1" || n == "txt2" || (n.size() > 4 && !n.compare(n.size() - 4, 4, ".ckv"));
        });
        printf("freed %.0f MB used only for the prompt\n", freed / 1e6);
    }
    if (F0) modsFor(0.0f, mAll0, mHead0);
    std::vector<std::pair<std::string, double>> times;
    if (getenv("DRAW")) {
        // the Turbo model is a few-step distill: each step predicts the clean video and re-noises it with FRESH noise
        // to the next level (ComfyUI's "lcm" sampler), as model_wan.h (same random numbers for the same SEED)
        auto x = rdf(Rd + "/noise.f32"), sg = rdf(Rd + "/sigmas.f32");
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
            auto a = std::chrono::steady_clock::now();
            fillPatches(x);
            modsFor(sigma, mAll, mHead);
            runStep(gstep, &times);
            auto v = readH(Yout, (size_t)N * 192);
            const double ms = secsSince(a) * 1000;
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
            printf("draw step %zu: sigma %.3f -> %.3f, %.1f ms\n", si, sg[si], sg[si + 1], ms);
        }
        wrf(Rd + "/draw.f32", x.data(), x.size());
        printf("drawing: %.1f s of steps, latent in %s/draw.f32\n", total / 1000, Rd.c_str());
        if (stream) printf("reading blocks from storage: %.1f s in all (alongside the chip); the chip waited for it %zu times\n", tFill, vk::Graph::stalls);
    } else {
        modsFor(sigma, mAll, mHead);
        runStep(gstep);
        const char *files[3] = {"x_block0", "x_block_mid", "x_block_last"}, *what[3] = {"block 0", "block 15", "block 29"};
        for (int s = 0; s < 3; s++) {
            if (!fexists(Rd + "/" + files[s] + ".f32")) continue;
            auto ref = rdf(Rd + "/" + files[s] + ".f32");
            printf("  after %-12s error %.5f\n", what[s], relerr((float *)snap[s].map, ref.data(), (size_t)N * Dm));
        }
        auto y = readH(Yout, (size_t)N * 192);
        auto ref = rdf(Rd + "/out.f32");
        printf("  final output: error %.5f\n", relerr(y.data(), ref.data(), y.size()));
        for (int l = 0; l < loops; l++) {
            auto a = std::chrono::steady_clock::now();
            runStep(gstep, &times);
            printf("step %d: %.1f ms\n", l, secsSince(a) * 1000);
        }
    }
    printTimes(times);
    (void)T;
    return 0;
}

}  // namespace wan
