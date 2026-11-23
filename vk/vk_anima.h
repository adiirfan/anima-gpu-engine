// Anima on Vulkan: the same model file, test folders and numbers as the OpenCL engine (model_anima.h). A whole step is
// recorded once and replayed.   env: CEN=<table>.cen, DRAW=1 (whole picture)
#pragma once
#include "vkops.h"

namespace anima {

static void ropeTables(int hp, int wp, std::vector<float> &P, std::vector<float> &Q) {   // as model_anima.h
    const int D = 128, dh = D / 6 * 2, dt = D - 2 * dh;
    const double th = 10000.0 * pow(4.0, (double)dh / (dh - 2));
    P.assign((size_t)hp * wp * D, 1.0f); Q.assign((size_t)hp * wp * D, 0.0f);
    for (int y = 0; y < hp; y++) for (int x = 0; x < wp; x++) {
        const size_t r = ((size_t)y * wp + x) * D;
        for (int i = 0; i < dh / 2; i++) {
            const double f = 1.0 / pow(th, (2.0 * i) / dh);
            const int ky = dt / 2 + i, kx = dt / 2 + dh / 2 + i;
            const double ay = y * f, ax = x * f;
            P[r + ky] = P[r + 64 + ky] = (float)cos(ay); Q[r + ky] = (float)-sin(ay); Q[r + 64 + ky] = (float)sin(ay);
            P[r + kx] = P[r + 64 + kx] = (float)cos(ax); Q[r + kx] = (float)-sin(ax); Q[r + 64 + kx] = (float)sin(ax);
        }
    }
}

static int run(const std::string &Rd, int loops, int PW, int PH) {
    if (autoDeclines("0.sa_qkv")) return 3;
    auto t0 = std::chrono::steady_clock::now();
    const int NB = 28, Dm = 2048, CTXD = 1024, LK = 256, HEADS = 16;
    const int N = (PW / 16) * (PH / 16);
    // ---- inputs ----
    auto exists = [&](const char *n) { return fexists(Rd + "/" + n); };
    std::vector<float> patches = rdf(Rd + "/patches.f32"), ctxv = rdf(Rd + "/ctx.f32");
    std::vector<float> embraw = exists("emb_raw.f32") ? rdf(Rd + "/emb_raw.f32") : std::vector<float>(Dm, 0.0f);
    std::vector<float> Pt, Qt;
    if (exists("rope_P.f32")) { Pt = rdf(Rd + "/rope_P.f32"); Qt = rdf(Rd + "/rope_Q.f32"); } else ropeTables(PH / 16, PW / 16, Pt, Qt);
    if (patches.size() != (size_t)N * 68 || ctxv.size() != (size_t)LK * CTXD) { fprintf(stderr, "bad input sizes\n"); return 1; }
    Buf bP = up(Pt.data(), Pt.size() * 4), bQ = up(Qt.data(), Qt.size() * 4);
    Buf Xp = vk::buf((size_t)N * 72 * 2), Xc = vk::buf((size_t)LK * CTXD * 2);
    auto putPatches = [&]() {
        h16 *x = (h16 *)Xp.map;
        for (int n = 0; n < N; n++) for (int c = 0; c < 72; c++) x[(size_t)n * 72 + c] = (h16)(c < 68 ? patches[(size_t)n * 68 + c] : 0.0f);
    };
    putPatches();
    putH(Xc, ctxv.data(), ctxv.size());

    // ---- work buffers ----
    Buf res = vk::buf((size_t)N * Dm * 4), QKV = vk::buf((size_t)N * 3 * Dm * 2), O = vk::buf((size_t)N * Dm * 2);
    Buf Hb = vk::buf((size_t)N * 4 * Dm * 2), Qc = vk::buf((size_t)N * Dm * 2), Xa = vk::buf((size_t)N * Dm * 2);
    Buf Qh = vk::buf((size_t)HEADS * N * 128 * 2), Kh = vk::buf((size_t)HEADS * N * 128 * 2), Vh = vk::buf((size_t)HEADS * N * 128 * 2);
    scratch(N, 4 * Dm);
    const int hc = headsPerPass(HEADS, N);   // heads per pass, so the score table stays under 64 MB
    Buf S = vk::buf(std::max((size_t)hc * N * N, (size_t)HEADS * N * LK) * 2);
    Buf KVc = vk::buf((size_t)LK * 2 * Dm * 2);
    std::vector<Buf> CK(NB), CV(NB);
    for (int i = 0; i < NB; i++) { CK[i] = vk::buf((size_t)HEADS * LK * 128 * 2); CV[i] = vk::buf((size_t)HEADS * LK * 128 * 2); }
    Buf embRaw = vk::buf(Dm * 4), emb = vk::buf(Dm * 4), T1 = vk::buf(Dm * 4), lora = vk::buf(3 * Dm * 4), hid = vk::buf(256 * 4);
    std::vector<Buf> mod(NB * 3);
    for (auto &m : mod) m = vk::buf(3 * Dm * 4);
    Buf fmod = vk::buf(2 * Dm * 4), Yout = vk::buf((size_t)N * 64 * 2);
    std::vector<Buf> snap(3);
    for (auto &s : snap) s = vk::buf((size_t)N * Dm * 4);
    const float cbias = logf(512.0f - LK + 1.0f), qscale = 1.0f / sqrtf(128.0f);

    // ---- centring (CEN=<file>): mu per input channel and W.mu per output, per noise level, blended per step ----
    std::vector<float> cenSig, cenTab; size_t cenStep = 0; Buf cenBuf;
    if (getenv("CEN")) {
        FILE *f = fopen(getenv("CEN"), "rb");
        if (!f) { perror("centring table"); return 1; }
        int hdr[2];
        if (fread(hdr, 4, 2, f) != 2 || hdr[1] != NB) { fprintf(stderr, "bad centring table\n"); return 1; }
        const char *kinds[6] = {"sa_qkv", "sa_o", "ca_q", "ca_o", "l1", "l2"};
        const int kK[6] = {1, 1, 1, 1, 1, 4}, kM[6] = {3, 1, 1, 1, 4, 1};
        for (int b = 0; b < NB; b++) for (int l = 0; l < 6; l++) cenStep += (size_t)(kK[l] + kM[l]) * Dm;
        cenSig.resize(hdr[0]); cenTab.resize(cenStep * hdr[0]);
        if (fread(cenSig.data(), 4, hdr[0], f) != (size_t)hdr[0] || fread(cenTab.data(), 4, cenTab.size(), f) != cenTab.size()) { fprintf(stderr, "short centring table\n"); return 1; }
        fclose(f);
        cenBuf = vk::buf(cenStep * 4);
        size_t o = 0;
        for (int b = 0; b < NB; b++) for (int l = 0; l < 6; l++) {
            const std::string nm = std::to_string(b) + "." + kinds[l];
            Ref mu = R(cenBuf, o * 4, (size_t)kK[l] * Dm * 4); o += (size_t)kK[l] * Dm;
            Ref wb = R(cenBuf, o * 4, (size_t)kM[l] * Dm * 4); o += (size_t)kM[l] * Dm;
            CENR[nm] = {mu, wb};
        }
        printf("centring: %d noise levels\n", hdr[0]);
    }
    float sigma = 0.6f;
    auto cenSet = [&](float sg) {
        if (cenSig.empty()) return;
        const int Sn = (int)cenSig.size();
        int a = 0; float t = 0;
        if (sg <= cenSig[Sn - 1]) a = Sn - 1;
        else if (sg < cenSig[0]) {
            while (a + 1 < Sn && cenSig[a + 1] > sg) a++;
            if (a + 1 < Sn) t = (cenSig[a] - sg) / (cenSig[a] - cenSig[a + 1]);
        }
        float *v = (float *)cenBuf.map;
        for (size_t i = 0; i < cenStep; i++) v[i] = cenTab[a * cenStep + i] + (t > 0 ? t * (cenTab[(a + 1) * cenStep + i] - cenTab[a * cenStep + i]) : 0.0f);
    };

    // ---- graph 1: the prompt's keys / values (once per picture) ----
    vk::Graph gctx; G = &gctx;
    for (int i = 0; i < NB; i++) {
        const std::string b = std::to_string(i) + ".";
        mulh(0, "gemm ctx", 0, R(Xc), R(wh(b + "ca_kv")), R(KVc), R(dummy), LK, 2 * Dm, CTXD, CTXD, CTXD, 2 * Dm);
        prep(R(KVc), 2 * Dm, 0, R(wf(b + "ca_kn")), R(bP), R(bQ), R(CK[i]), LK, LK, HEADS, true, false, 1.0f, "prep ck");
        prep(R(KVc), 2 * Dm, Dm, R(dummy), R(bP), R(bQ), R(CV[i]), LK, LK, HEADS, false, false, 1.0f, "prep cv");
    }
    // ---- graph 2: modulation (whenever the noise level changes) ----
    vk::Graph gmod; G = &gmod;
    gemv(wh("t1"), R(embRaw), R(dummy), R(T1), Dm, Dm, 0);
    gemv(wh("t2"), R(T1), R(dummy), R(lora), 3 * Dm, Dm, 1);
    const char *mn[3] = {"msa", "mca", "mmlp"};
    for (int i = 0; i < NB; i++) for (int j = 0; j < 3; j++) {
        const std::string b = std::to_string(i) + "." + mn[j];
        gemv(wh(b + "1"), R(emb), R(dummy), R(hid), 256, Dm, 1);
        gemv(wh(b + "2"), R(hid), R(lora), R(mod[i * 3 + j]), 3 * Dm, 256, 0);
    }
    gemv(wh("fmod1"), R(emb), R(dummy), R(hid), 256, Dm, 1);
    gemv(wh("fmod2"), R(hid), R(lora, 0, 2 * Dm * 4), R(fmod), 2 * Dm, 256, 0);
    // ---- graph 3: one step ----
    const bool check = !getenv("DRAW");
    vk::Graph gstep; G = &gstep;
    mulh(3, "gemm embed", 0, R(Xp), R(wh("x_embedder")), R(dummy), R(res), N, Dm, 72, 72, 72, Dm);
    for (int i = 0; i < NB; i++) {
        const std::string b = std::to_string(i) + ".";
        Buf &m0 = mod[i * 3], &m1 = mod[i * 3 + 1], &m2 = mod[i * 3 + 2];
        auto sh = [&](Buf &m) { return R(m, 0, Dm * 4); };
        auto sc = [&](Buf &m) { return R(m, Dm * 4, Dm * 4); };
        auto gt = [&](Buf &m) { return R(m, 2 * Dm * 4, Dm * 4); };
        // self-attention
        rotLn(b + "sa_qkv", R(res), sh(m0), sc(m0), Dm, N);
        mulq(0, "qkv", b + "sa_qkv", R(QKV), R(dummy), R(dummy), N);
        prep(R(QKV), 3 * Dm, 0, R(wf(b + "sa_qn")), R(bP), R(bQ), R(Qh), N, N, HEADS, true, true, qscale, "prep q");
        prep(R(QKV), 3 * Dm, Dm, R(wf(b + "sa_kn")), R(bP), R(bQ), R(Kh), N, N, HEADS, true, true, 1.0f, "prep k");
        prep(R(QKV), 3 * Dm, 2 * Dm, R(dummy), R(bP), R(bQ), R(Vh), N, N, HEADS, false, false, 1.0f, "prep v");
        attend(Qh, Kh, Vh, S, R(O), N, N, HEADS, hc, Dm);
        rotX(b + "sa_o", R(O), Dm, Dm, N);
        mulq(2, "out", b + "sa_o", R(dummy), R(res), gt(m0), N);
        // cross-attention
        rotLn(b + "ca_q", R(res), sh(m1), sc(m1), Dm, N);
        mulq(0, "q", b + "ca_q", R(Qc), R(dummy), R(dummy), N);
        prep(R(Qc), Dm, 0, R(wf(b + "ca_qn")), R(bP), R(bQ), R(Qh), N, N, HEADS, true, false, qscale, "prep q");
        mula("qk cross", 0, R(Qh), R(CK[i]), R(S), N, LK, 128, 128, 128, LK, N * 128, LK * 128, N * LK, HEADS);
        softmax(R(S), N, LK, LK, cbias, HEADS);
        mula("pv cross", 1, R(S), R(CV[i]), R(O), N, 128, LK, LK, 128, Dm, N * LK, LK * 128, 128, HEADS);
        rotX(b + "ca_o", R(O), Dm, Dm, N);
        mulq(2, "out", b + "ca_o", R(dummy), R(res), gt(m1), N);
        // MLP
        rotLn(b + "l1", R(res), sh(m2), sc(m2), Dm, N);
        mulq(1, "mlp1", b + "l1", R(Hb), R(dummy), R(dummy), N);
        rotX(b + "l2", R(Hb), 4 * Dm, 4 * Dm, N);
        mulq(2, "mlp2", b + "l2", R(dummy), R(res), gt(m2), N);
        if (check && (i == 0 || i == 1 || i == NB - 1)) copyf(R(res), R(snap[i == 0 ? 0 : i == 1 ? 1 : 2]), (size_t)N * Dm);
    }
    lnmod(R(res), R(fmod, 0, Dm * 4), R(fmod, Dm * 4, Dm * 4), R(Xa), N, Dm);
    mulh(0, "gemm final", 0, R(Xa), R(wh("final_linear")), R(Yout), R(dummy), N, 64, Dm, Dm, Dm, 64);
    vk::savePipelines();
    printf("setup: %.1f s rearranging weights, %.1f s building shaders\n", tWeights, tKern);
    printf("loaded and recorded in %.1f s: %.0f MB of weights on the chip, %zu dispatches per step, %d heads per attention pass\n",
           secsSince(t0), chipBytes / 1e6, gstep.ops.size(), hc);

    // ---- run ----
    std::vector<float> tn(Dm);
    memcpy(tn.data(), ft("t_norm").src, Dm * 4);
    if (FILE *sf = fopen((Rd + "/sigma.f32").c_str(), "rb")) { if (fread(&sigma, 4, 1, sf) != 1) sigma = 0.6f; fclose(sf); }
    auto setMod = [&]() {
        double ss = 0; for (int c = 0; c < Dm; c++) ss += embraw[c] * (double)embraw[c];
        const float rs = 1.0f / sqrtf(ss / Dm + 1e-6f);
        for (int c = 0; c < Dm; c++) { ((float *)embRaw.map)[c] = embraw[c]; ((float *)emb.map)[c] = embraw[c] * rs * tn[c]; }
        gmod.run();
        cenSet(sigma);
    };
    {
        auto a = std::chrono::steady_clock::now();
        gctx.run();
        printf("prompt keys: %.2f s (with recording)\n", secsSince(a));
    }
    std::vector<std::pair<std::string, double>> times;
    if (getenv("DRAW")) {   // a whole picture: the test folder's noise, the app's schedule from sigmas.f32 (or DRAW steps)
        std::vector<float> sg = exists("sigmas.f32") ? rdf(Rd + "/sigmas.f32") : std::vector<float>();
        const int steps = sg.size() > 1 ? (int)sg.size() - 1 : atoi(getenv("DRAW"));
        auto shf = [](double t) { return 3 * t / (1 + 2 * t); };
        double total = 0;
        for (int si = 0; si < steps; si++) {
            const float s0 = sg.empty() ? (float)shf(1.0 - (double)si / steps) : sg[si],
                        s1 = sg.empty() ? (si + 1 < steps ? (float)shf(1.0 - (double)(si + 1) / steps) : 0.0f) : sg[si + 1];
            sigma = s0;
            for (int k = 0; k < Dm / 2; k++) {
                const float e = expf(-logf(10000.0f) * k / (Dm / 2)), a = s0 * e;
                embraw[k] = cosf(a); embraw[Dm / 2 + k] = sinf(a);
            }
            auto a = std::chrono::steady_clock::now();
            putPatches();
            setMod();
            gstep.run(&times);
            auto y = readH(Yout, (size_t)N * 64);
            const double ms = secsSince(a) * 1000;
            total += ms;
            for (int n = 0; n < N; n++) for (int Cc = 0; Cc < 16; Cc++) for (int pp = 0; pp < 4; pp++)
                patches[(size_t)n * 68 + Cc * 4 + pp] += (s1 - s0) * y[(size_t)n * 64 + pp * 16 + Cc];
            printf("draw step %d: sigma %.3f -> %.3f, %.1f ms\n", si, s0, s1, ms);
        }
        wrf(Rd + "/draw.f32", patches.data(), patches.size());
        printf("drawing: %.1f s of steps, latent in %s/draw.f32\n", total / 1000, Rd.c_str());
    } else {
        setMod();
        gstep.run();
        const char *names[3] = {"0", "1", "27"};
        for (int s = 0; s < 3; s++) {
            const std::string f = "x_after" + std::string(names[s]) + ".f32";
            if (!exists(f.c_str())) continue;
            auto ref = rdf(Rd + "/" + f);
            printf("  after block %2s: error %.5f\n", names[s], relerr((float *)snap[s].map, ref.data(), ref.size()));
        }
        auto y = readH(Yout, (size_t)N * 64);
        auto ref = rdf(Rd + "/out.f32");
        printf("  final output: error %.5f\n", relerr(y.data(), ref.data(), y.size()));
        for (int l = 0; l < loops; l++) {
            auto a = std::chrono::steady_clock::now();
            gstep.run(&times);
            printf("step %d: %.1f ms\n", l, secsSince(a) * 1000);
        }
    }
    printTimes(times);
    return 0;
}

}  // namespace anima
