// MOSS-Image (+ the merged 8-step add-on) on Vulkan, the same test folders as the OpenCL engine (model_moss.h).
// Width 3072, 24 heads, 42 blocks, MLP 8192 with MOSS's paired activation (EPI 5), RMS norms, biases on the prompt
// side. The prompt's own 2 blocks run once per picture; every step then runs [picture rows; prompt rows] through all
// 42 blocks with one shared modulation (each block's norm weights folded in on the host). Attention runs over the
// rows rounded up to 128 (the extra rows stay zero and are masked).   env: DRAW=1 (whole picture)
#pragma once
#include "vkops.h"

namespace moss {
static const int NB = 42, Dm = 3072, HEADS = 24, F = 8192, CTXD = 4096, VIN = 256;   // latent 128, padded to 256

// rotations in the engine's form (rotate-half): P = [cos, cos], Q = [-sin, sin]; 4 axes of 32, theta 10000
static void rope(const std::vector<float> &ids4, int n, std::vector<float> &P, std::vector<float> &Q) {
    P.assign((size_t)n * 128, 0.0f); Q.assign((size_t)n * 128, 0.0f);
    for (int r = 0; r < n; r++) {
        int k = 0;
        for (int a = 0; a < 4; a++)
            for (int i = 0; i < 16; i++, k++) {
                const double ang = ids4[(size_t)r * 4 + a] * std::pow(10000.0, -2.0 * i / 32);
                P[(size_t)r * 128 + k] = P[(size_t)r * 128 + 64 + k] = (float)std::cos(ang);
                Q[(size_t)r * 128 + k] = -(float)std::sin(ang); Q[(size_t)r * 128 + 64 + k] = (float)std::sin(ang);
            }
    }
}

static int run(const std::string &Rd, int loops) {
    if (autoDeclines("b0.qkv")) return 3;
    // the blocks' tables are read from storage block by block (the prompt's own blocks "tb0", "tb1" are not blocks here)
    const bool stream = streamWanted();
    if (stream) streamOf = [](const std::string &n) { int b; char suf[32]; return sscanf(n.c_str(), "b%d.%31s", &b, suf) == 2 ? b : -1; };
    auto t0 = std::chrono::steady_clock::now();
    int nc = 0, PW = 0, PH = 0;
    if (FILE *f = fopen((Rd + "/info.txt").c_str(), "r")) { if (fscanf(f, "%d %d %d", &nc, &PW, &PH) != 3) nc = 0; fclose(f); }
    if (!nc) { fprintf(stderr, "%s/info.txt missing or bad\n", Rd.c_str()); return 1; }
    const int gh = PH / 16, gw = PW / 16, Ni = gh * gw, Nt = Ni + nc, Na = pad128(Nt), Ta = pad128(nc);
    const int hc = headsPerPass(HEADS, Na);
    printf("%dx%d: %d picture + %d prompt rows, attention over %d rows, %d heads at a time\n", PW, PH, Ni, nc, Na, hc);

    // ---- rotations: picture rows (0, nc, y - (h-1)/2, x - (w-1)/2), prompt rows (0, i, 0, 0) ----
    std::vector<float> ids((size_t)Nt * 4, 0.0f), idt((size_t)nc * 4, 0.0f), P, Q, Pt, Qt;
    for (int y = 0; y < gh; y++) for (int x = 0; x < gw; x++) {
        float *p = &ids[((size_t)y * gw + x) * 4];
        p[1] = (float)nc; p[2] = y - (gh - 1) / 2.0f; p[3] = x - (gw - 1) / 2.0f;
    }
    for (int i = 0; i < nc; i++) { ids[((size_t)Ni + i) * 4 + 1] = (float)i; idt[(size_t)i * 4 + 1] = (float)i; }
    rope(ids, Nt, P, Q); rope(idt, nc, Pt, Qt);
    Buf bP = up(P.data(), P.size() * 4), bQ = up(Q.data(), Q.size() * 4), bPt = up(Pt.data(), Pt.size() * 4), bQt = up(Qt.data(), Qt.size() * 4);

    // ---- buffers ----
    Buf res = vk::buf((size_t)Nt * Dm * 4), resT = vk::buf((size_t)nc * Dm * 4);
    Buf QKV = vk::buf((size_t)Nt * 3 * Dm * 2), O = vk::buf((size_t)Na * Dm * 2), Hh = vk::buf((size_t)Nt * F * 2);
    Buf Yout = vk::buf((size_t)Ni * 128 * 2), Xin = vk::buf((size_t)Ni * VIN * 2), Xc = vk::buf((size_t)nc * CTXD * 2);
    memset(Xin.map, 0, Xin.size);
    auto zeroed = [](size_t bytes) { Buf b = vk::buf(bytes); memset(b.map, 0, bytes); return b; };   // attention rows past the last stay zero
    Buf Qh = zeroed((size_t)HEADS * Na * 256), Kh = zeroed((size_t)HEADS * Na * 256), Vh = zeroed((size_t)HEADS * Na * 256);
    Buf QhT = zeroed((size_t)HEADS * Ta * 256), KhT = zeroed((size_t)HEADS * Ta * 256), VhT = zeroed((size_t)HEADS * Ta * 256);
    Buf S = vk::buf(std::max((size_t)hc * Na * Na, (size_t)HEADS * Ta * Ta) * 2);
    scratch(Nt, F);
    Buf bTemb = vk::buf(256 * 4), bT1 = vk::buf(Dm * 4), bE = vk::buf(Dm * 4), bMod = vk::buf(6 * Dm * 4), bHm = vk::buf(2 * Dm * 4);
    Buf mAll = vk::buf((size_t)NB * 6 * Dm * 4), mHead = vk::buf(2 * Dm * 4), bZero = zeroed(Dm * 4), bOnes = vk::buf(Dm * 4);
    for (int k = 0; k < Dm; k++) ((float *)bOnes.map)[k] = 1.0f;
    auto part = [&](Buf &m, size_t i) { return R(m, i * Dm * 4, Dm * 4); };
    auto vec = [&](const std::string &n) { const float *s = (const float *)ft(n).src; return std::vector<float>(s, s + Dm); };
    std::vector<std::vector<float>> n1(NB), n2(NB);
    for (int b = 0; b < NB; b++) { n1[b] = vec("b" + std::to_string(b) + ".n1"); n2[b] = vec("b" + std::to_string(b) + ".n2"); }
    const auto fnorm = vec("fnorm");
    const float qscale = 1.0f / sqrtf(128.0f);
    auto qkvPrep = [&](const std::string &p, Buf &Qo, Buf &Ko, Buf &Vo, Buf &Pb, Buf &Qb, int n, int outN) {
        prep(R(QKV), 3 * Dm, 0, R(wf(p + "qn")), R(Pb), R(Qb), R(Qo), n, outN, HEADS, true, true, qscale, "prep q");
        prep(R(QKV), 3 * Dm, Dm, R(wf(p + "kn")), R(Pb), R(Qb), R(Ko), n, outN, HEADS, true, true, 1.0f, "prep k");
        prep(R(QKV), 3 * Dm, 2 * Dm, R(dummy), R(Pb), R(Qb), R(Vo), n, outN, HEADS, false, false, 1.0f, "prep v");
    };

    // ---- graph 1: the prompt's own 2 blocks, once per picture (norm weight only: sc = n - 1, gates 1) ----
    vk::Graph gtxt; G = &gtxt;
    std::vector<Buf> tsc;
    for (int j = 0; j < 2; j++) for (const char *nn : {"n1", "n2"}) {
        auto a = vec("tb" + std::to_string(j) + "." + nn);
        for (auto &v : a) v -= 1.0f;
        tsc.push_back(up(a.data(), Dm * 4));
    }
    rotX("tproj", R(Xc), CTXD, CTXD, nc);
    mulq(3, "tproj", "tproj", R(dummy), R(resT), R(dummy), nc);
    for (int j = 0; j < 2; j++) {
        const std::string p = "tb" + std::to_string(j) + ".";
        rotLn(p + "qkv", R(resT), R(bZero), R(tsc[j * 2]), Dm, nc, 1e-6f, true);
        mulq(0, "qkv", p + "qkv", R(QKV), R(dummy), R(dummy), nc);
        qkvPrep(p, QhT, KhT, VhT, bPt, bQt, nc, Ta);
        attend(QhT, KhT, VhT, S, R(O), Ta, nc, HEADS, HEADS, Dm);
        rotX(p + "o", R(O), Dm, Dm, nc);
        mulq(2, "o", p + "o", R(dummy), R(resT), R(bOnes), nc);
        rotLn(p + "mlp_in", R(resT), R(bZero), R(tsc[j * 2 + 1]), Dm, nc, 1e-6f, true);
        mulq(5, "mlp_in", p + "mlp_in", R(Hh), R(dummy), R(dummy), nc);
        rotX(p + "down", R(Hh), F, F, nc);
        mulq(2, "down", p + "down", R(dummy), R(resT), R(bOnes), nc);
    }
    // ---- graph 2: the shared modulation's raw vectors (the host folds in the norms) ----
    vk::Graph gmod; G = &gmod;
    gemv(wh("t1"), R(bTemb), R(wf("t1.b")), R(bT1), Dm, 256, 0);
    gemv(wh("t2"), R(bT1), R(wf("t2.b")), R(bE), Dm, Dm, 1);
    gemv(wh("mod"), R(bE), R(wf("mod.b")), R(bMod), 6 * Dm, Dm, 0);
    gemv(wh("hmod"), R(bE), R(wf("hmod.b")), R(bHm), 2 * Dm, Dm, 0);
    // ---- graph 3: one step ----
    vk::Graph gstep; G = &gstep;
    rotX("vin", R(Xin), VIN, VIN, Ni);
    mulq(3, "vin", "vin", R(dummy), R(res), R(dummy), Ni);
    copyf(R(resT), R(res, (size_t)Ni * Dm * 4), (size_t)nc * Dm);   // the prompt rows, fresh every step
    for (int b = 0; b < NB; b++) {
        const std::string p = "b" + std::to_string(b) + ".";
        if (b) G->mark();   // segment b: block b
        const size_t m = (size_t)b * 6;
        rotLn(p + "qkv", R(res), part(mAll, m), part(mAll, m + 1), Dm, Nt, 1e-6f, true);
        mulq(0, "qkv", p + "qkv", R(QKV), R(dummy), R(dummy), Nt);
        qkvPrep(p, Qh, Kh, Vh, bP, bQ, Nt, Na);
        attend(Qh, Kh, Vh, S, R(O), Na, Nt, HEADS, hc, Dm);
        rotX(p + "o", R(O), Dm, Dm, Nt);
        mulq(2, "o", p + "o", R(dummy), R(res), part(mAll, m + 2), Nt);
        rotLn(p + "mlp_in", R(res), part(mAll, m + 3), part(mAll, m + 4), Dm, Nt, 1e-6f, true);
        mulq(5, "mlp_in", p + "mlp_in", R(Hh), R(dummy), R(dummy), Nt);
        rotX(p + "down", R(Hh), F, F, Nt);
        mulq(2, "down", p + "down", R(dummy), R(res), part(mAll, m + 5), Nt);
    }
    rotLn("fout", R(res), part(mHead, 0), part(mHead, 1), Dm, Ni, 1e-6f, true);
    mulq(0, "fout", "fout", R(Yout), R(dummy), R(dummy), Ni);
    vk::savePipelines();
    printf("setup: %.1f s rearranging weights, %.1f s building shaders\n", tWeights, tKern);
    printf("loaded and recorded in %.1f s: %.0f MB of weights on the chip, %zu dispatches per step\n",
           secsSince(t0), chipBytes / 1e6, gstep.ops.size());
    if (stream) printf("streaming the blocks from storage: %.0f MB of slots\n", slotBytes / 1e6);

    {   // the prompt pass
        auto a = std::chrono::steady_clock::now();
        auto ctx = rdf(Rd + "/ctx.f32");
        if (ctx.size() < (size_t)nc * CTXD) { fprintf(stderr, "ctx.f32 has fewer than %d rows\n", nc); return 1; }
        putH(Xc, ctx.data(), (size_t)nc * CTXD);
        gtxt.run();
        printf("prompt blocks %.0f ms\n", secsSince(a) * 1000);
        const size_t freed = dropTables([](const std::string &n) { return n == "tproj" || !n.compare(0, 2, "tb"); });
        printf("freed %.0f MB used only for the prompt\n", freed / 1e6);
    }
    // noise level s -> every block's [shift1, scale1', gate1, shift2, scale2', gate2] (scale' = n (1 + scale) - 1) and
    // the head's shift / scale'
    auto setMod = [&](float s) {
        float *te = (float *)bTemb.map;
        for (int i = 0; i < 128; i++) {
            const float a = 1000.0f * s * expf(-logf(10000.0f) * i / 128.0f);
            te[i] = cosf(a); te[128 + i] = sinf(a);
        }
        gmod.run();
        const float *mm = (const float *)bMod.map, *hm = (const float *)bHm.map;
        float *all = (float *)mAll.map, *hd = (float *)mHead.map;
        for (int b = 0; b < NB; b++) {
            float *a = all + (size_t)b * 6 * Dm;
            for (int k = 0; k < Dm; k++) {
                a[k] = mm[k]; a[2 * Dm + k] = mm[2 * Dm + k]; a[3 * Dm + k] = mm[3 * Dm + k]; a[5 * Dm + k] = mm[5 * Dm + k];
                a[Dm + k] = n1[b][k] * (1.0f + mm[Dm + k]) - 1.0f;
                a[4 * Dm + k] = n2[b][k] * (1.0f + mm[4 * Dm + k]) - 1.0f;
            }
        }
        for (int k = 0; k < Dm; k++) { hd[k] = hm[k]; hd[Dm + k] = fnorm[k] * (1.0f + hm[Dm + k]) - 1.0f; }
    };
    auto putLatent = [&](const std::vector<float> &x) {
        h16 *d = (h16 *)Xin.map;
        for (int r = 0; r < Ni; r++) for (int c = 0; c < 128; c++) d[(size_t)r * VIN + c] = (h16)x[(size_t)r * 128 + c];
    };
    auto x = rdf(Rd + "/noise.f32"), sg = rdf(Rd + "/sigmas.f32");
    sg.push_back(0.0f);
    std::vector<std::pair<std::string, double>> times;
    if (getenv("DRAW")) {
        double total = 0;
        for (size_t i = 0; i + 1 < sg.size(); i++) {
            auto a = std::chrono::steady_clock::now();
            setMod(sg[i]);
            putLatent(x);
            runStep(gstep, &times);
            auto v = readH(Yout, (size_t)Ni * 128);
            const double ms = secsSince(a) * 1000;
            total += ms;
            for (size_t k = 0; k < x.size(); k++) x[k] += (sg[i + 1] - sg[i]) * v[k];
            printf("draw step %zu: sigma %.3f -> %.3f, %.1f ms\n", i, sg[i], sg[i + 1], ms);
        }
        wrf(Rd + "/draw.f32", x.data(), x.size());
        printf("drawing: %.1f s of steps, latent in %s/draw.f32\n", total / 1000, Rd.c_str());
        if (stream) printf("reading blocks from storage: %.1f s in all (alongside the chip); the chip waited for it %zu times\n", tFill, vk::Graph::stalls);
    } else {
        setMod(sg[0]);
        putLatent(x);
        runStep(gstep);
        if (fexists(Rd + "/out.f32")) {
            auto v = readH(Yout, (size_t)Ni * 128);
            auto ref = rdf(Rd + "/out.f32");
            printf("  first step     error %.5f\n", relerr(v.data(), ref.data(), v.size()));
        }
        for (int l = 0; l < loops; l++) {
            auto a = std::chrono::steady_clock::now();
            runStep(gstep, &times);
            printf("step %d: %.1f ms\n", l, secsSince(a) * 1000);
        }
    }
    printTimes(times);
    return 0;
}

}  // namespace moss
