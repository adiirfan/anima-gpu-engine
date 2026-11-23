// Krea 2 Turbo on Vulkan (width 6144, 48 query heads sharing 12 key / value heads, 28 blocks, SwiGLU 16384), the same
// test folders as the OpenCL engine (model_krea.h), with the file in the Vulkan layout (tools/pack_krea.py, BITS=4
// Q4K=1, GS 256: 4-bit drawing blocks with two-level scales, 8-bit everything else in blocks of 256 inputs).
// Single stream: picture and prompt rows go through every block together, rows padded to Nr (a multiple of 128) and
// keys from Ni + Lt on masked. Only attention runs over the padding: every other operation runs the Lv = Ni + Lt real rows
// (the padded query / key / value rows stay zero). Each query head h attends with key / value head h / 4.
// Once per picture the text fusion turns the reader's 12 tapped layers per prompt token (ctx.f32: Lt x 12 x 2560)
// into the prompt rows (or txt.f32 is read as they are): two blocks inside each token's 12 layers, the projector
// 12 -> 1 on the host, two blocks across the prompt, then RMS norm, linear + GELU, linear.
// In <ref>: engine.txt "Ni Lt Lp W H" (Lp unused here), ctx.f32 or txt.f32, sigmas.f32; DRAW=1 noise.f32 (16 x h x w)
// -> draw.f32 (h x w x 16); else img0.f32 / out0p.f32 check the first step.
#pragma once
#include "vkops.h"
#include "gen/grpattn.spv.h"
#include "gen/sgate.spv.h"

namespace krea {
static const int NB = 28, Dm = 6144, NHEADS = 48, KVH = 12, KVD = KVH * 128, F = 16384, QKVG = Dm + 2 * KVD + Dm;
static const int TD = 2560, TH = 20, TF = 6912, NL = 12, TQ = 4 * TD;

// attention inside groups of grp rows (rows [0, n)), q / k / v of the fp16 rows X (stride ld) at columns qc / kc / vc
static void grpattn(Ref X, int n, int grp, int ld, int qc, int kc, int vc, Ref qw, Ref kw, Ref O, int ldo, int heads) {
    struct { uint32_t n1, grp, ld, qc, kc, vc, ldo; } pc{(uint32_t)n, (uint32_t)grp, (uint32_t)ld, (uint32_t)qc, (uint32_t)kc, (uint32_t)vc, (uint32_t)ldo};
    G->add(KERN(grpattn, 4, sizeof(pc), {}), {X, qw, kw, O}, pc, (uint32_t)n, (uint32_t)heads, 1, "grpattn");
}
// Y (fp16 [n][D]) = O (stride ldo) * sigmoid(Gt[gc..], stride ldg)
static void sgate(Ref O, int ldo, Ref Gt, int ldg, int gc, Ref Y, int D, int n) {
    struct { uint32_t D, ldo, ldg, gc; } pc{(uint32_t)D, (uint32_t)ldo, (uint32_t)ldg, (uint32_t)gc};
    G->add(KERN(sgate, 3, sizeof(pc), {}), {O, Gt, Y}, pc, (uint32_t)(D + 255) / 256, (uint32_t)n, 1, "sgate");
}

// the text fusion on the chip: ctx (Lt x 12 x 2560) -> the prompt rows (Lt x 6144)
static std::vector<float> fuse(const std::vector<float> &ctx, int Lt) {
    const int n1 = Lt * NL;
    Buf res = vk::buf((size_t)n1 * TD * 4), QKV = vk::buf((size_t)n1 * TQ * 2), O = vk::buf((size_t)n1 * TD * 2);
    Buf Og = vk::buf((size_t)n1 * TD * 2), H = vk::buf((size_t)n1 * TF * 2);
    Buf zero = vk::buf(TD * 4), ones = vk::buf(TD * 4), Y1 = vk::buf((size_t)Lt * Dm * 2), Y3 = vk::buf((size_t)Lt * Dm * 2);
    memset(zero.map, 0, zero.size);
    for (int i = 0; i < TD; i++) ((float *)ones.map)[i] = 1.0f;
    memcpy(res.map, ctx.data(), (size_t)n1 * TD * 4);   // row t * 12 + l = token t's reader layer l
    auto block = [&](const std::string &b, int grp, int n) {
        rotLn(b + "qkvg", R(res), R(zero), R(wf(b + "n1")), TD, n, 1e-5f, true);
        mulq(0, "t qkvg", b + "qkvg", R(QKV), R(dummy), R(dummy), n);
        grpattn(R(QKV), n, grp, TQ, 0, TD, 2 * TD, R(wf(b + "qn")), R(wf(b + "kn")), R(O), TD, TH);
        sgate(R(O), TD, R(QKV), TQ, 3 * TD, R(Og), TD, n);
        rotX(b + "wo", R(Og), TD, TD, n);
        mulq(2, "t wo", b + "wo", R(dummy), R(res), R(ones), n);
        rotLn(b + "mlp_in", R(res), R(zero), R(wf(b + "n2")), TD, n, 1e-5f, true);
        mulq(4, "t mlp_in", b + "mlp_in", R(H), R(dummy), R(dummy), n);
        rotX(b + "down", R(H), TF, TF, n);
        mulq(2, "t down", b + "down", R(dummy), R(res), R(ones), n);
    };
    vk::Graph g1; G = &g1;
    block("tl0.", NL, n1);
    block("tl1.", NL, n1);
    vk::Graph g2; G = &g2;
    block("tr0.", Lt, Lt);
    block("tr1.", Lt, Lt);
    rotLn("txt.l1", R(res), R(zero), R(wf("txt.n")), TD, Lt, 1e-5f, true);
    mulq(6, "txt.l1", "txt.l1", R(Y1), R(dummy), R(dummy), Lt);   // + bias, then GELU
    rotX("txt.l2", R(Y1), Dm, Dm, Lt);
    mulq(0, "txt.l2", "txt.l2", R(Y3), R(dummy), R(dummy), Lt);   // + bias
    g1.run();
    {   // projector: each token's 12 layer rows -> one row, into rows [0, Lt)
        const float *pw = (const float *)ft("tf.proj").src;
        float *r = (float *)res.map;
        std::vector<float> y((size_t)Lt * TD, 0.0f);
        for (int t = 0; t < Lt; t++) for (int l = 0; l < NL; l++) {
            const float *s = r + ((size_t)t * NL + l) * TD;
            for (int d = 0; d < TD; d++) y[(size_t)t * TD + d] += pw[l] * s[d];
        }
        memcpy(r, y.data(), y.size() * 4);
    }
    g2.run();
    auto out = readH(Y3, (size_t)Lt * Dm);
    for (Buf *b : {&res, &QKV, &O, &Og, &H, &zero, &ones, &Y1, &Y3}) vk::freeBuf(*b);
    return out;
}

static int run(const std::string &Rd, int loops) {
    if (autoDeclines("b0.qkvg")) return 3;
    auto t0 = std::chrono::steady_clock::now();
    int Ni = 0, Lt = 0, Lp = 0, PW = 0, PH = 0;
    if (FILE *f = fopen((Rd + "/engine.txt").c_str(), "r")) {
        if (fscanf(f, "%d %d %d %d %d", &Ni, &Lt, &Lp, &PW, &PH) != 5) Ni = 0;
        fclose(f);
    }
    if (!Ni) { fprintf(stderr, "%s/engine.txt missing or bad\n", Rd.c_str()); return 1; }
    const int hp = PH / 16, wp = PW / 16, Lv = Ni + Lt, Nr = pad128(Lv);
    if (hp * wp != Ni) { fprintf(stderr, "engine.txt: %d patches but %dx%d\n", Ni, PW, PH); return 1; }
    const bool stream = streamWanted();
    if (stream) streamOf = [](const std::string &n) { int b; char suf[32]; return sscanf(n.c_str(), "b%d.%31s", &b, suf) == 2 ? b : -1; };
    scratch(std::max(Nr, Lt * NL), F);

    // ---- the prompt rows: from the reader's layers (ctx.f32) when there is no txt.f32 or KTEXT=1 ----
    std::vector<float> txtv;
    const bool haveTxt = fexists(Rd + "/txt.f32");
    if (fexists(Rd + "/ctx.f32") && (!haveTxt || getenv("KTEXT"))) {
        auto ctx = rdf(Rd + "/ctx.f32");
        if (ctx.size() != (size_t)Lt * NL * TD) { fprintf(stderr, "ctx.f32: %zu values, not %d x 12 x 2560\n", ctx.size(), Lt); return 1; }
        auto a = std::chrono::steady_clock::now();
        txtv = fuse(ctx, Lt);
        printf("prompt: %d tokens through the text fusion, %.0f ms", Lt, secsSince(a) * 1000);
        if (haveTxt) { auto ref = rdf(Rd + "/txt.f32"); printf(", error %.5f", relerr(txtv.data(), ref.data(), txtv.size())); }
        printf("\n");
        const size_t freed = dropTables([](const std::string &n) { return n.rfind("tl", 0) == 0 || n.rfind("tr", 0) == 0 || n.rfind("txt.", 0) == 0; });
        printf("freed %.0f MB of prompt tables\n", freed / 1e6);
    } else txtv = rdf(Rd + "/txt.f32");
    if (txtv.size() != (size_t)Lt * Dm) { fprintf(stderr, "prompt rows: %zu values, not %d x %d\n", txtv.size(), Lt, Dm); return 1; }
    printf("%dx%d: %d picture + %d prompt rows, %d with padding\n", PW, PH, Ni, Lt, Nr);

    // ---- rotations (axes 32 / 48 / 48 of each head's 64 pairs, theta 1000): picture (0, y, x), prompt and padding
    // at 0, rotate-half form (the packer reorders q / k channels): P = [cos, cos], Q = [-sin, sin] ----
    std::vector<float> Pt((size_t)Nr * 128), Qt((size_t)Nr * 128);
    for (int n = 0; n < Nr; n++) for (int i = 0; i < 64; i++) {
        double ang = 0;
        if (n < Ni) {
            const int py = n / wp, px = n % wp;
            if (i >= 16 && i < 40) ang = py * pow(1000.0, -2.0 * (i - 16) / 48);
            else if (i >= 40) ang = px * pow(1000.0, -2.0 * (i - 40) / 48);
        }
        Pt[(size_t)n * 128 + i] = Pt[(size_t)n * 128 + 64 + i] = (float)cos(ang);
        Qt[(size_t)n * 128 + i] = (float)-sin(ang); Qt[(size_t)n * 128 + 64 + i] = (float)sin(ang);
    }
    Buf bP = up(Pt.data(), Pt.size() * 4), bQ = up(Qt.data(), Qt.size() * 4);

    // ---- work buffers; the stream starts each step from res0 (prompt rows, zero padding; picture rows are set by
    // the first layer) ----
    auto zeroed = [](size_t bytes) { Buf b = vk::buf(bytes); memset(b.map, 0, bytes); return b; };
    Buf res0 = zeroed((size_t)Nr * Dm * 4), res = vk::buf((size_t)Nr * Dm * 4);
    memcpy((float *)res0.map + (size_t)Ni * Dm, txtv.data(), txtv.size() * 4);
    Buf QKV = vk::buf((size_t)Nr * QKVG * 2), O = vk::buf((size_t)Nr * Dm * 2), Og = vk::buf((size_t)Nr * Dm * 2);
    Buf Hh = vk::buf((size_t)Nr * F * 2), Ximg = zeroed((size_t)Ni * 128 * 2), Yout = vk::buf((size_t)Ni * 128 * 2);
    Buf Qh = zeroed((size_t)NHEADS * Nr * 128 * 2), Kh = zeroed((size_t)KVH * Nr * 128 * 2), Vh = zeroed((size_t)KVH * Nr * 128 * 2);
    const int QPK = NHEADS / KVH;   // query heads per key / value head
    Buf S = vk::buf((size_t)QPK * Nr * Nr * 2);
    Buf bTemb = vk::buf(256 * 4), bT1 = vk::buf(Dm * 4), bT = vk::buf(Dm * 4), bVec = vk::buf(6 * Dm * 4);
    Buf mods = vk::buf((size_t)NB * 6 * Dm * 4), fin = vk::buf(2 * Dm * 4);
    auto part = [&](Buf &m, int i) { return R(m, (size_t)i * Dm * 4, Dm * 4); };
    {   // the output layer's bias padded to its 128 rows, added on the chip
        std::vector<float> lb(128, 0.0f);
        memcpy(lb.data(), ft("last.b").src, 64 * 4);
        Wf["last.b"] = up(lb.data(), lb.size() * 4);
    }

    // ---- graph 1: t and the shared modulation from the noise level ----
    vk::Graph gmod; G = &gmod;
    gemv(wh("t1"), R(bTemb), R(wf("t1.b")), R(bT1), Dm, 256, 0);
    gemv(wh("t2"), R(bT1), R(wf("t2.b")), R(bT), Dm, Dm, 2);
    gemv(wh("tproj"), R(bT), R(wf("tproj.b")), R(bVec), 6 * Dm, Dm, 2);

    // ---- graph 2: one step over all rows ----
    vk::Graph gstep; G = &gstep;
    copyf(R(res0), R(res), (size_t)Lv * Dm);
    {   // picture rows = img_in(x) + bias (fp16 multiply, 128 inputs of which 64 used)
        PCh pc{(uint32_t)Ni, (uint32_t)Dm, 128, 128, 128, (uint32_t)Dm, 0, 0, 0, 1};
        G->add(KERN(gemm_h, 6, sizeof(PCh), {64, 64, 16, 4, 4, 0, 3, 256}), {R(Ximg), R(wh("first")), R(dummy), R(res), R(dummy), R(wf("first.b"))},
               pc, (Dm + 63) / 64, (Ni + 63) / 64, 1, "first");
    }
    const float qscale = 1.0f / sqrtf(128.0f);
    for (int b = 0; b < NB; b++) {
        const std::string p = "b" + std::to_string(b) + ".";
        if (b) G->mark();   // segment b: block b
        const int m0 = b * 6;
        rotLn(p + "qkvg", R(res), part(mods, m0 + 1), part(mods, m0), Dm, Lv, 1e-5f, true);
        mulq(0, "qkvg", p + "qkvg", R(QKV), R(dummy), R(dummy), Lv);
        prep(R(QKV), QKVG, 0, R(wf(p + "qn")), R(bP), R(bQ), R(Qh), Lv, Nr, NHEADS, true, true, qscale, "prep q");
        prep(R(QKV), QKVG, Dm, R(wf(p + "kn")), R(bP), R(bQ), R(Kh), Lv, Nr, KVH, true, true, 1.0f, "prep k");
        prep(R(QKV), QKVG, Dm + KVD, R(dummy), R(bP), R(bQ), R(Vh), Lv, Nr, KVH, false, false, 1.0f, "prep v");
        for (int g = 0; g < KVH; g++) {   // the QPK query heads of key / value head g in one batch (key stride 0)
            const size_t qo = (size_t)g * QPK * Nr * 128 * 2, ko = (size_t)g * Nr * 128 * 2;
            mula("qk", 0, R(Qh, qo), R(Kh, ko), R(S), Nr, Nr, 128, 128, 128, Nr, Nr * 128, 0, Nr * Nr, QPK);
            softmax(R(S), Nr, Nr, Lv, 0.0f, QPK);
            mula("pv", 1, R(S), R(Vh, ko), R(O, (size_t)g * QPK * 128 * 2), Nr, 128, Nr, Nr, 128, Dm, Nr * Nr, 0, 128, QPK);
        }
        sgate(R(O), Dm, R(QKV), QKVG, Dm + 2 * KVD, R(Og), Dm, Lv);
        rotX(p + "wo", R(Og), Dm, Dm, Lv);
        mulq(2, "wo", p + "wo", R(dummy), R(res), part(mods, m0 + 2), Lv);
        rotLn(p + "mlp_in", R(res), part(mods, m0 + 4), part(mods, m0 + 3), Dm, Lv, 1e-5f, true);
        mulq(4, "mlp_in", p + "mlp_in", R(Hh), R(dummy), R(dummy), Lv);
        rotX(p + "down", R(Hh), F, F, Lv);
        mulq(2, "down", p + "down", R(dummy), R(res), part(mods, m0 + 5), Lv);
    }
    rotLn("last", R(res), R(fin, Dm * 4, Dm * 4), R(fin, 0, Dm * 4), Dm, Ni, 1e-5f, true);
    mulq(0, "last", "last", R(Yout), R(dummy), R(dummy), Ni);
    vk::savePipelines();
    printf("setup: %.1f s rearranging weights, %.1f s building shaders\n", tWeights, tKern);
    printf("loaded and recorded in %.1f s: %.0f MB of weights on the chip, %zu dispatches per step\n",
           secsSince(t0), chipBytes / 1e6, gstep.ops.size());
    if (stream) printf("streaming the blocks from storage: %.0f MB of slots\n", slotBytes / 1e6);

    // the per-block tables the modulation is made from (fp32, read from the file each picture)
    auto host = [](const std::string &n) { const T &t = ft(n); return std::vector<float>((const float *)t.src, (const float *)t.src + t.M * t.K); };
    std::vector<std::vector<float>> lin(NB), n1(NB), n2(NB);
    for (int i = 0; i < NB; i++) {
        const std::string m = "m" + std::to_string(i) + ".";
        lin[i] = host(m + "lin"); n1[i] = host(m + "n1"); n2[i] = host(m + "n2");
    }
    const auto lastLin = host("last.lin"), lastN = host("last.n");
    auto modsFor = [&](float sigma) {
        float *te = (float *)bTemb.map;
        for (int i = 0; i < 128; i++) {
            const float a = 1000.0f * sigma * expf(-logf(10000.0f) * i / 128.0f);
            te[i] = cosf(a); te[128 + i] = sinf(a);
        }
        gmod.run();
        const float *hT = (const float *)bT.map, *hVec = (const float *)bVec.map;
        float *mm = (float *)mods.map;
        for (int i = 0; i < NB; i++) {
            float *m = mm + (size_t)i * 6 * Dm;
            const float *l = lin[i].data();
            for (int c = 0; c < Dm; c++) {
                // prescale, preshift, pregate, postscale, postshift, postgate; (1 + norm) folded into the scales
                m[c] = (1 + n1[i][c]) * (1 + hVec[c] + l[c]) - 1;
                m[Dm + c] = hVec[Dm + c] + l[Dm + c];
                m[2 * Dm + c] = hVec[2 * Dm + c] + l[2 * Dm + c];
                m[3 * Dm + c] = (1 + n2[i][c]) * (1 + hVec[3 * Dm + c] + l[3 * Dm + c]) - 1;
                m[4 * Dm + c] = hVec[4 * Dm + c] + l[4 * Dm + c];
                m[5 * Dm + c] = hVec[5 * Dm + c] + l[5 * Dm + c];
            }
        }
        float *f = (float *)fin.map;   // output layer: scale, shift from t itself
        for (int c = 0; c < Dm; c++) {
            f[c] = (1 + lastN[c]) * (1 + hT[c] + lastLin[c]) - 1;
            f[Dm + c] = hT[c] + lastLin[Dm + c];
        }
    };
    std::vector<std::pair<std::string, double>> times;
    auto step = [&](float sigma, const std::vector<float> &x) {   // x: Ni x 64 patches -> the velocity, Ni x 64
        h16 *xi = (h16 *)Ximg.map;
        for (int n = 0; n < Ni; n++) for (int k = 0; k < 64; k++) xi[(size_t)n * 128 + k] = (h16)x[(size_t)n * 64 + k];
        modsFor(sigma);
        runStep(gstep, &times);
        std::vector<float> v((size_t)Ni * 64);
        const h16 *y = (const h16 *)Yout.map;
        for (int n = 0; n < Ni; n++) for (int k = 0; k < 64; k++) v[(size_t)n * 64 + k] = (float)y[(size_t)n * 128 + k];
        return v;
    };
    auto sg = rdf(Rd + "/sigmas.f32");

    if (getenv("DRAW")) {
        auto nz = rdf(Rd + "/noise.f32");
        const int h = 2 * hp, wd = 2 * wp;
        if (nz.size() != (size_t)16 * h * wd) { fprintf(stderr, "noise.f32: %zu values, not 16 x %d x %d\n", nz.size(), h, wd); return 1; }
        std::vector<float> x((size_t)Ni * 64);   // patches: feature c * 4 + dy * 2 + dx of patch (py, px)
        auto pidx = [&](int c, int y, int xx) { return ((size_t)(y / 2) * wp + xx / 2) * 64 + c * 4 + (y % 2) * 2 + xx % 2; };
        for (int c = 0; c < 16; c++) for (int y = 0; y < h; y++) for (int xx = 0; xx < wd; xx++)
            x[pidx(c, y, xx)] = nz[((size_t)c * h + y) * wd + xx];
        double total = 0;
        for (size_t si = 0; si + 1 < sg.size(); si++) {
            auto a = std::chrono::steady_clock::now();
            auto v = step(sg[si], x);
            const double ms = secsSince(a) * 1000;
            total += ms;
            for (size_t k = 0; k < x.size(); k++) x[k] += (sg[si + 1] - sg[si]) * v[k];
            printf("draw step %zu: sigma %.3f -> %.3f, %.1f ms\n", si, sg[si], sg[si + 1], ms);
        }
        std::vector<float> lat((size_t)h * wd * 16);
        for (int c = 0; c < 16; c++) for (int y = 0; y < h; y++) for (int xx = 0; xx < wd; xx++)
            lat[((size_t)y * wd + xx) * 16 + c] = x[pidx(c, y, xx)];
        wrf(Rd + "/draw.f32", lat.data(), lat.size());
        printf("drawing: %.1f s of steps, latent in %s/draw.f32\n", total / 1000, Rd.c_str());
        if (stream) printf("reading blocks from storage: %.1f s in all (alongside the chip); the chip waited for it %zu times\n", tFill, vk::Graph::stalls);
    } else {
        auto img0 = rdf(Rd + "/img0.f32");
        auto v = step(sg[0], img0);
        if (fexists(Rd + "/out0p.f32")) {
            auto ref = rdf(Rd + "/out0p.f32");
            printf("  final output: error %.5f\n", relerr(v.data(), ref.data(), v.size()));
        }
        times.clear();
        for (int l = 0; l < loops; l++) {
            auto a = std::chrono::steady_clock::now();
            step(sg[0], img0);
            printf("step %d: %.1f ms\n", l, secsSince(a) * 1000);
        }
    }
    printTimes(times);
    return 0;
}

}  // namespace krea
