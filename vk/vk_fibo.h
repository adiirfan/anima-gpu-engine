// Fibo 2.0 edit-turbo on Vulkan (width 3584, 28 heads, 2 refiner + 44 main blocks b0..b45, SwiGLU 9728), the same
// test folders and file as the OpenCL engine (model_fibo.h), with the 4-bit two-level-scale file (tools/pack_fibo.py,
// BITS=4 Q4K=1) on the matrix hardware. Rows every step: [picture Ni | edit picture Ni | gist G]; only picture rows
// carry the step's modulation, the others the t = 0 one (folded into their norm weights by the packer). Every layer's
// output is RMS-normed, scaled and added (rmsadd). The refiner (b0, b1) masks the gist keys; the gist rows are put
// back after it. Injection blocks (b8, b16, ..., b40) also attend to their prompt bundle. Once per edit: the
// Perceiver (2 layers: cross to bundle 0, self, SwiGLU) makes the G gist rows; the injection blocks' prompt keys /
// values are made once too.
//   env: DRAW=1 (whole edit from noise.f32 along sigmas.f32 into draw.f32; ref.f32 = the picture to edit)
#pragma once
#include "vkops.h"

namespace fibo {
static const int NB = 46, Dm = 3584, HEADS = 28, F = 9728, CAP = 7680, NBUN = 6, MAXG = 512, VIN = 256;
static const int INJB[5] = {8, 16, 24, 32, 40};

static void rope(const std::vector<float> &ids3, int n, std::vector<float> &P, std::vector<float> &Q) {
    const int dims[3] = {32, 48, 48};
    P.assign((size_t)n * 128, 0.0f); Q.assign((size_t)n * 128, 0.0f);
    for (int r = 0; r < n; r++) {
        int k = 0;
        for (int a = 0; a < 3; a++)
            for (int i = 0; i < dims[a] / 2; i++, k++) {
                const double ang = ids3[(size_t)r * 3 + a] * std::pow(256.0, -2.0 * i / dims[a]);
                P[(size_t)r * 128 + k] = P[(size_t)r * 128 + 64 + k] = (float)std::cos(ang);
                Q[(size_t)r * 128 + k] = -(float)std::sin(ang); Q[(size_t)r * 128 + 64 + k] = (float)std::sin(ang);
            }
    }
}

static int run(const std::string &Rd, int loops) {
    if (autoDeclines("b2.qkv")) return 3;
    // the blocks' big tables are read from storage block by block (the prompt tables "ckv" are used once, so stay)
    const bool stream = streamWanted();
    if (stream) streamOf = [](const std::string &n) {
        int b; char suf[32];
        return sscanf(n.c_str(), "b%d.%31s", &b, suf) == 2 && strcmp(suf, "ckv") ? b : -1;
    };
    auto t0 = std::chrono::steady_clock::now();
    int T = 0, Gn = 0, h = 0, wd = 0;
    if (FILE *f = fopen((Rd + "/info.txt").c_str(), "r")) { if (fscanf(f, "%d %d %d %d", &T, &Gn, &h, &wd) != 4) T = 0; fclose(f); }
    if (!T) { fprintf(stderr, "%s/info.txt missing or bad\n", Rd.c_str()); return 1; }
    const int Ni = h * wd, NA = 2 * Ni + Gn, Tp = (T + 7) / 8 * 8, Na = pad128(NA), Lk = pad128(Tp), Gp = pad128(Gn);
    if (Ni % 8 || Gn % 8 || Gn > MAXG) { fprintf(stderr, "row counts must be multiples of 8\n"); return 1; }
    const int hc = headsPerPass(HEADS, Na), hcx = headsPerPass(HEADS, std::max(Na, Lk));
    printf("%dx%d edit: %d picture + %d edit-picture + %d gist rows, prompt %d, attention over %d rows, %d heads at a time\n",
           wd * 16, h * 16, Ni, Ni, Gn, T, Na, hc);

    // ---- rotations: picture (MAXG + 1, y, x), edit picture (MAXG + 2, y, x), gist (1 .. G, 0, 0) ----
    std::vector<float> ids((size_t)NA * 3, 0.0f), P, Q;
    for (int y = 0; y < h; y++) for (int x = 0; x < wd; x++) {
        const size_t r = (size_t)y * wd + x;
        ids[r * 3] = MAXG + 1; ids[r * 3 + 1] = (float)y; ids[r * 3 + 2] = (float)x;
        ids[(Ni + r) * 3] = MAXG + 2; ids[(Ni + r) * 3 + 1] = (float)y; ids[(Ni + r) * 3 + 2] = (float)x;
    }
    for (int i = 0; i < Gn; i++) ids[((size_t)2 * Ni + i) * 3] = (float)(i + 1);
    rope(ids, NA, P, Q);
    Buf bP = up(P.data(), P.size() * 4), bQ = up(Q.data(), Q.size() * 4);

    // ---- work buffers ----
    auto zeroed = [](size_t bytes) { Buf b = vk::buf(bytes); memset(b.map, 0, bytes); return b; };
    const int NR = std::max(NA, Tp);
    Buf res = vk::buf((size_t)NA * Dm * 4), resG = vk::buf((size_t)Gn * Dm * 4);
    Buf QKV = vk::buf((size_t)NR * 3 * Dm * 2), O = vk::buf((size_t)std::max(Na, Gp) * Dm * 2), A = vk::buf((size_t)NA * Dm * 2);
    Buf Hh = vk::buf((size_t)NR * F * 2), Yout = vk::buf((size_t)Ni * 128 * 2), Xin = zeroed((size_t)NA * VIN * 2);
    Buf KVb = vk::buf((size_t)std::max(Tp, Gn) * 2 * Dm * 2);
    Buf Qh = zeroed((size_t)HEADS * Na * 128 * 2), Kh = zeroed((size_t)HEADS * Na * 128 * 2), Vh = zeroed((size_t)HEADS * Na * 128 * 2);
    Buf Qg = zeroed((size_t)HEADS * Gp * 128 * 2), Kp = zeroed((size_t)HEADS * Lk * 128 * 2), Vp = zeroed((size_t)HEADS * Lk * 128 * 2);
    Buf Ks = zeroed((size_t)HEADS * Gp * 128 * 2), Vs = zeroed((size_t)HEADS * Gp * 128 * 2);
    std::vector<Buf> Kc(5), Vc(5);
    for (int i = 0; i < 5; i++) { Kc[i] = zeroed((size_t)HEADS * Lk * 128 * 2); Vc[i] = zeroed((size_t)HEADS * Lk * 128 * 2); }
    Buf S = vk::buf(std::max(std::max((size_t)hc * Na * Na, (size_t)hcx * Na * Lk), (size_t)hcx * Gp * std::max(Lk, Gp)) * 2);
    scratch(NR, std::max(F, CAP));
    Buf bTs = vk::buf(256 * 4), bT1 = vk::buf(1024 * 4), bTe = vk::buf(256 * 4), bOs = vk::buf(Dm * 4);
    Buf mraw = vk::buf((size_t)NB * 4 * Dm * 4), mAll = vk::buf((size_t)NB * 8 * Dm * 4);
    Buf zero = zeroed(Dm * 4), ones = vk::buf(Dm * 4);
    for (int k = 0; k < Dm; k++) ((float *)ones.map)[k] = 1.0f;
    auto mv = [&](int b, int j) { return R(mAll, ((size_t)b * 8 + j) * Dm * 4, Dm * 4); };
    auto vec = [&](const std::string &n) { return (const float *)ft(n).src; };
    auto minus1 = [&](const std::string &n) {
        Buf b = vk::buf(Dm * 4); const float *v = vec(n);
        for (int k = 0; k < Dm; k++) ((float *)b.map)[k] = v[k] - 1.0f;
        return b;
    };
    const float qscale = 1.0f / sqrtf(128.0f);
    Ref none = R(dummy);

    // ---- the prompt bundles (fp16, padded rows zero) ----
    std::vector<Buf> Xb(NBUN);
    {
        auto ctx = rdf(Rd + "/ctx.f32");   // T x (6 x 7680)
        if (ctx.size() < (size_t)T * NBUN * CAP) { fprintf(stderr, "ctx.f32 needs %d x %d numbers\n", T, NBUN * CAP); return 1; }
        std::vector<float> bv((size_t)Tp * CAP);
        for (int k = 0; k < NBUN; k++) {
            std::fill(bv.begin(), bv.end(), 0.0f);
            for (int t = 0; t < T; t++)
                std::copy(ctx.begin() + (size_t)t * NBUN * CAP + (size_t)k * CAP, ctx.begin() + (size_t)t * NBUN * CAP + (size_t)(k + 1) * CAP,
                          bv.begin() + (size_t)t * CAP);
            Xb[k] = vk::buf((size_t)Tp * CAP * 2);
            putH(Xb[k], bv.data(), bv.size());
        }
    }
    // key / value prep of n rows of KVb ([n][2 Dm]: keys then values) into K / V ([heads][outN][128]), keys normed
    auto kvPrep = [&](const std::string &kn, int n, Buf &K, Buf &V, int outN) {
        prep(R(KVb), 2 * Dm, 0, R(wf(kn)), none, none, R(K), n, outN, HEADS, true, false, 1.0f, "prep k");
        prep(R(KVb), 2 * Dm, Dm, none, none, none, R(V), n, outN, HEADS, false, false, 1.0f, "prep v");
    };

    // ---- graph 1: the Perceiver and the injection blocks' prompt keys (once per edit) ----
    vk::Graph gonce; G = &gonce;
    {
        std::vector<Buf> pm;   // the Perceiver's norm scales minus 1: [cn, sn, fn] per layer
        for (int j = 0; j < 2; j++)
            for (const char *n : {"cn", "sn", "fn"}) pm.push_back(minus1("p" + std::to_string(j) + "." + n));
        for (int j = 0; j < 2; j++) {
            const std::string p = "p" + std::to_string(j) + ".";
            Buf &cn = pm[j * 3], &sn = pm[j * 3 + 1], &fn = pm[j * 3 + 2];
            rotX(p + "ckv", R(Xb[0]), CAP, CAP, Tp);
            mulq(0, "ckv", p + "ckv", R(KVb), none, none, Tp);
            kvPrep(p + "ckn", Tp, Kp, Vp, Lk);
            rotLn(p + "cq", R(resG), R(zero), R(cn), Dm, Gn, 1e-6f, true);
            mulq(0, "cq", p + "cq", R(QKV), none, none, Gn);
            prep(R(QKV), Dm, 0, R(wf(p + "cqn")), none, none, R(Qg), Gn, Gp, HEADS, true, false, qscale, "prep q");
            attendX(Qg, Gp, Kp, Vp, Lk, T, S, R(O), HEADS, hcx, Dm);
            rotX(p + "co", R(O), Dm, Dm, Gn);
            mulq(2, "co", p + "co", none, R(resG), R(ones), Gn);
            rotLn(p + "skv", R(resG), R(zero), R(sn), Dm, Gn, 1e-6f, true);
            mulq(0, "skv", p + "skv", R(KVb), none, none, Gn);
            kvPrep(p + "skn", Gn, Ks, Vs, Gp);
            rotLn(p + "sq", R(resG), R(zero), R(sn), Dm, Gn, 1e-6f, true);
            mulq(0, "sq", p + "sq", R(QKV), none, none, Gn);
            prep(R(QKV), Dm, 0, R(wf(p + "sqn")), none, none, R(Qg), Gn, Gp, HEADS, true, false, qscale, "prep q");
            attendX(Qg, Gp, Ks, Vs, Gp, Gn, S, R(O), HEADS, hcx, Dm);
            rotX(p + "so", R(O), Dm, Dm, Gn);
            mulq(2, "so", p + "so", none, R(resG), R(ones), Gn);
            rotLn(p + "mlp_in", R(resG), R(zero), R(fn), Dm, Gn, 1e-6f, true);
            mulq(4, "mlp_in", p + "mlp_in", R(Hh), none, none, Gn);
            rotX(p + "down", R(Hh), F, F, Gn);
            mulq(2, "down", p + "down", none, R(resG), R(ones), Gn);
        }
        for (int i = 0; i < 5; i++) {
            const std::string p = "b" + std::to_string(INJB[i]) + ".";
            rotX(p + "ckv", R(Xb[1 + i]), CAP, CAP, Tp);
            mulq(0, "ckv", p + "ckv", R(KVb), none, none, Tp);
            kvPrep(p + "ckn", Tp, Kc[i], Vc[i], Lk);
        }
    }

    // ---- graph 2: the modulations from the noise level ----
    vk::Graph gmod; G = &gmod;
    gemv(wh("t1"), R(bTs), R(wf("t1.b")), R(bT1), 1024, 256, 0);
    gemv(wh("t2"), R(bT1), R(wf("t2.b")), R(bTe), 256, 1024, 1);
    gemv(wh("out_scale"), R(bTe), R(wf("out_scale.b")), R(bOs), Dm, 256, 1);
    for (int b = 0; b < NB; b++) {
        const std::string p = "b" + std::to_string(b) + ".";
        gemv(wh(p + "mod"), R(bTe), R(wf(p + "mod.b")), R(mraw, (size_t)b * 4 * Dm * 4, 4 * Dm * 4), 4 * Dm, 256, 0);
    }

    // ---- graph 3: one step ----
    std::vector<Buf> cn1(5);
    for (int i = 0; i < 5; i++) cn1[i] = minus1("b" + std::to_string(INJB[i]) + ".cn1");
    vk::Graph gstep; G = &gstep;
    const size_t rowD4 = (size_t)Dm * 4, rowD2 = (size_t)Dm * 2;
    auto gistIn = [&]() { copyf(R(resG), R(res, (size_t)2 * Ni * rowD4), (size_t)Gn * Dm); };
    rotX("x_in", R(Xin), VIN, VIN, 2 * Ni);
    mulq(3, "x_in", "x_in", none, R(res), none, 2 * Ni);
    gistIn();
    for (int b = 0; b < NB; b++) {
        const std::string p = "b" + std::to_string(b) + ".";
        if (b) G->mark();   // segment b: block b (the first also the inputs, the last also the output layer)
        // picture rows with this step's modulation, the rest with t = 0's
        rotLn(p + "qkv", R(res), R(zero), mv(b, 0), Dm, Ni, 1e-6f, true);
        mulq(0, "qkv", p + "qkv", R(QKV), none, none, Ni);
        rotLn(p + "qkv", R(res, Ni * rowD4), R(zero), mv(b, 4), Dm, NA - Ni, 1e-6f, true);
        mulq(0, "qkv", p + "qkv", R(QKV, (size_t)Ni * 3 * rowD2), none, none, NA - Ni);
        prep(R(QKV), 3 * Dm, 0, R(wf(p + "qn")), R(bP), R(bQ), R(Qh), NA, Na, HEADS, true, true, qscale, "prep q");
        prep(R(QKV), 3 * Dm, Dm, R(wf(p + "kn")), R(bP), R(bQ), R(Kh), NA, Na, HEADS, true, true, 1.0f, "prep k");
        prep(R(QKV), 3 * Dm, 2 * Dm, none, R(bP), R(bQ), R(Vh), NA, Na, HEADS, false, false, 1.0f, "prep v");
        attend(Qh, Kh, Vh, S, R(O), Na, b < 2 ? 2 * Ni : NA, HEADS, hc, Dm);
        rotX(p + "o", R(O), Dm, Dm, NA);
        mulq(0, "o", p + "o", R(A), none, none, NA);
        rmsadd(R(A), Dm, R(res), mv(b, 1), Ni, Dm);
        rmsadd(R(A, Ni * rowD2), Dm, R(res, Ni * rowD4), mv(b, 5), NA - Ni, Dm);
        for (int i = 0; i < 5; i++) if (INJB[i] == b) {
            rotLn(p + "cq", R(res), R(zero), R(cn1[i]), Dm, NA, 1e-6f, true);
            mulq(0, "cq", p + "cq", R(QKV), none, none, NA);
            prep(R(QKV), Dm, 0, R(wf(p + "cqn")), none, none, R(Qh), NA, Na, HEADS, true, false, qscale, "prep q");
            attendX(Qh, Na, Kc[i], Vc[i], Lk, T, S, R(O), HEADS, hcx, Dm);
            rotX(p + "co", R(O), Dm, Dm, NA);
            mulq(0, "co", p + "co", R(A), none, none, NA);
            rmsadd(R(A), Dm, R(res), R(wf(p + "cn2")), NA, Dm);
        }
        rotLn(p + "mlp_in", R(res), R(zero), mv(b, 2), Dm, Ni, 1e-6f, true);
        mulq(4, "mlp_in", p + "mlp_in", R(Hh), none, none, Ni);
        rotLn(p + "mlp_in", R(res, Ni * rowD4), R(zero), mv(b, 6), Dm, NA - Ni, 1e-6f, true);
        mulq(4, "mlp_in", p + "mlp_in", R(Hh, (size_t)Ni * F * 2), none, none, NA - Ni);
        rotX(p + "down", R(Hh), F, F, NA);
        mulq(0, "down", p + "down", R(A), none, none, NA);
        rmsadd(R(A), Dm, R(res), mv(b, 3), Ni, Dm);
        rmsadd(R(A, Ni * rowD2), Dm, R(res, Ni * rowD4), mv(b, 7), NA - Ni, Dm);
        if (b == 1) gistIn();   // the refiner left the gist rows alone in meaning, not in memory
    }
    rotLn("out", R(res), R(zero), R(bOs), Dm, Ni, 1e-6f, false);   // the output layer's norm is a layer norm
    mulq(0, "out", "out", R(Yout), none, none, Ni);
    vk::savePipelines();
    printf("setup: %.1f s rearranging weights, %.1f s building shaders\n", tWeights, tKern);
    printf("loaded and recorded in %.1f s: %.0f MB of weights on the chip, %zu dispatches per step\n",
           secsSince(t0), chipBytes / 1e6, gstep.ops.size());
    if (stream) printf("streaming the blocks from storage: %.0f MB of slots\n", slotBytes / 1e6);

    // ---- once per edit: the Perceiver's gist rows = rms(x) * pnorm, and the injection keys ----
    {
        auto a = std::chrono::steady_clock::now();
        const float *qv = vec("queries");
        memcpy(resG.map, qv, (size_t)Gn * Dm * 4);
        gonce.run();
        float *x = (float *)resG.map; const float *pn = vec("pnorm");
        for (int g = 0; g < Gn; g++) {
            double ss = 0;
            for (int k = 0; k < Dm; k++) ss += (double)x[(size_t)g * Dm + k] * x[(size_t)g * Dm + k];
            const float r = 1.0f / sqrtf((float)(ss / Dm) + 1e-6f);
            for (int k = 0; k < Dm; k++) x[(size_t)g * Dm + k] *= r * pn[k];
        }
        printf("prompt, gist and injection keys %.0f ms\n", secsSince(a) * 1000);
        // the Perceiver, the prompt key tables and the prompt bundles are not needed again
        size_t freed = dropTables([](const std::string &n) {
            return !n.compare(0, 3, "p0.") || !n.compare(0, 3, "p1.") || (n.size() > 4 && !n.compare(n.size() - 4, 4, ".ckv"));
        });
        for (auto &b : Xb) { freed += b.size; vk::freeBuf(b); }
        printf("freed %.0f MB used only for the prompt\n", freed / 1e6);
    }

    // noise level s -> every block's eight vectors and the output layer's scale
    struct BN { const float *n1, *n2, *f1, *f2, *z1, *z2, *zf1, *zf2; };
    std::vector<BN> bn(NB);
    for (int b = 0; b < NB; b++) {
        const std::string p = "b" + std::to_string(b) + ".";
        bn[b] = {vec(p + "n1"), vec(p + "n2"), vec(p + "f1"), vec(p + "f2"), vec(p + "z1"), vec(p + "z2"), vec(p + "zf1"), vec(p + "zf2")};
    }
    auto modsFor = [&](float s) {
        float *te = (float *)bTs.map;
        for (int i = 0; i < 128; i++) {
            const float a = 1000.0f * s * expf(-logf(10000.0f) * i / 128.0f);
            te[i] = cosf(a); te[128 + i] = sinf(a);
        }
        gmod.run();
        const float *mr = (const float *)mraw.map;
        float *all = (float *)mAll.map;
        for (int b = 0; b < NB; b++) {
            const float *m = mr + (size_t)b * 4 * Dm;
            float *a = all + (size_t)b * 8 * Dm;
            const BN &n = bn[b];
            for (int k = 0; k < Dm; k++) {   // s1, g1, s2, g2 (gates through tanh), then the t = 0 ones
                a[k] = n.n1[k] * (1.0f + m[k]) - 1.0f;
                a[Dm + k] = n.n2[k] * tanhf(m[Dm + k]);
                a[2 * Dm + k] = n.f1[k] * (1.0f + m[2 * Dm + k]) - 1.0f;
                a[3 * Dm + k] = n.f2[k] * tanhf(m[3 * Dm + k]);
                a[4 * Dm + k] = n.z1[k] - 1.0f; a[5 * Dm + k] = n.z2[k];
                a[6 * Dm + k] = n.zf1[k] - 1.0f; a[7 * Dm + k] = n.zf2[k];
            }
        }
    };

    // ---- inputs: picture rows from x, edit-picture rows from ref.f32 (128 of the 256 inputs; the rest zero) ----
    auto refv = rdf(Rd + "/ref.f32");
    auto x = rdf(Rd + "/noise.f32"), sg = rdf(Rd + "/sigmas.f32");
    if (refv.size() != (size_t)Ni * 128 || x.size() != (size_t)Ni * 128) { fprintf(stderr, "ref.f32 / noise.f32 need %d x 128 numbers\n", Ni); return 1; }
    sg.push_back(0.0f);
    std::vector<float> xin((size_t)2 * Ni * VIN, 0.0f);
    for (int r = 0; r < Ni; r++) std::copy(refv.begin() + (size_t)r * 128, refv.begin() + (size_t)(r + 1) * 128, xin.begin() + (size_t)(Ni + r) * VIN);
    std::vector<std::pair<std::string, double>> times;
    auto step = [&](float s) {
        modsFor(s);
        for (int r = 0; r < Ni; r++) std::copy(x.begin() + (size_t)r * 128, x.begin() + (size_t)(r + 1) * 128, xin.begin() + (size_t)r * VIN);
        putH(Xin, xin.data(), xin.size());
        if (stream) gstep.runStreamed([](size_t i) { fillBlock((int)i); }, &times);
        else gstep.run(&times);
        return readH(Yout, (size_t)Ni * 128);
    };
    if (getenv("DRAW")) {
        double total = 0;
        for (size_t i = 0; i + 1 < sg.size(); i++) {
            auto a = std::chrono::steady_clock::now();
            auto v = step(sg[i]);
            const double ms = secsSince(a) * 1000;
            total += ms;
            for (size_t k = 0; k < x.size(); k++) x[k] += (sg[i + 1] - sg[i]) * v[k];
            printf("draw step %zu: sigma %.3f -> %.3f, %.1f ms\n", i, sg[i], sg[i + 1], ms);
        }
        wrf(Rd + "/draw.f32", x.data(), x.size());
        printf("drawing: %.1f s of steps, latent in %s/draw.f32\n", total / 1000, Rd.c_str());
        if (stream) printf("reading blocks from storage: %.1f s in all (alongside the chip); the chip waited for it %zu times\n", tFill, vk::Graph::stalls);
    } else {
        auto v = step(sg[0]);
        if (fexists(Rd + "/out.f32")) {
            auto ref = rdf(Rd + "/out.f32");
            printf("  first step     error %.5f\n", relerr(v.data(), ref.data(), v.size()));
        }
        if (getenv("VELOUT")) wrf(Rd + "/vel_vk.f32", v.data(), v.size());
        for (int l = 0; l < loops; l++) {
            auto a = std::chrono::steady_clock::now();
            step(sg[0]);
            printf("step %d: %.1f ms%s\n", l, secsSince(a) * 1000, stream ? (" (reading " + std::to_string((int)(tFill * 1000)) + " ms so far, " + std::to_string(vk::Graph::stalls) + " waits)").c_str() : "");
        }
    }
    printTimes(times);
    return 0;
}

}  // namespace fibo
