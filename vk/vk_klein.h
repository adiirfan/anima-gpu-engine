// FLUX.2 Klein 4B on Vulkan (width 3072, 24 heads, 5 double + 20 single blocks), the same test folders as the OpenCL
// engine (model_klein.h). Token rows: picture [0, Ni), prompt [Ni, Nt); the OpenCL engine's padding rows are not
// computed here. Attention runs over Na = Nt rounded up to 128 rows: the extra rows of Q / K / V stay zero and are
// masked as keys. Double blocks use separate weights per stream (rows addressed by byte offsets) and attend jointly.
//   env: DRAW=1 (whole picture from noise.f32 along sigmas.f32; ref.f32 = picture rows to edit, kept every step)
#pragma once
#include "vkops.h"

namespace klein {
static const int ND = 5, NS = 20, Dm = 3072, HEADS = 24, F = 9216, CTXD = 7680;

static int run(const std::string &Rd, int loops) {
    if (autoDeclines("d0.img_qkv")) return 3;
    // the blocks' tables are read from storage block by block: double blocks d0.. are blocks 0..ND-1, single blocks
    // s0.. follow them
    const bool stream = streamWanted();
    if (stream) streamOf = [](const std::string &n) {
        int b; char suf[32];
        if (sscanf(n.c_str(), "d%d.%31s", &b, suf) == 2) return b;
        if (sscanf(n.c_str(), "s%d.%31s", &b, suf) == 2) return ND + b;
        return -1;
    };
    auto t0 = std::chrono::steady_clock::now();
    int Ni = 0, Lt = 0, Lp = 0, PW = 0, PH = 0;
    if (FILE *f = fopen((Rd + "/info.txt").c_str(), "r")) {
        if (fscanf(f, "%d %d %d %d %d", &Ni, &Lt, &Lp, &PW, &PH) != 5) Ni = 0;
        fclose(f);
    }
    if (!Ni) { fprintf(stderr, "%s/info.txt missing or bad\n", Rd.c_str()); return 1; }
    const int Nt = Ni + Lt, Na = pad128(Nt), N = Ni + Lt + Lp;
    const int hc = headsPerPass(HEADS, Na);
    printf("%dx%d: %d picture + %d prompt rows, attention over %d rows, %d heads at a time\n", PW, PH, Ni, Lt, Na, hc);

    // ---- inputs ----
    auto imgv = rdf(Rd + "/img.f32"), ctxv = rdf(Rd + "/ctx.f32");
    auto Pt = rdf(Rd + "/rope_P.f32"), Qt = rdf(Rd + "/rope_Q.f32");
    if (imgv.size() != (size_t)Ni * 128 || ctxv.size() < (size_t)Lt * CTXD || Pt.size() < (size_t)Nt * 128) { fprintf(stderr, "bad input sizes\n"); return 1; }
    Buf bP = up(Pt.data(), (size_t)Nt * 128 * 4), bQ = up(Qt.data(), (size_t)Nt * 128 * 4);
    Buf Ximg = vk::buf((size_t)Ni * 128 * 2), Xctx = vk::buf((size_t)Lt * CTXD * 2);
    putH(Ximg, imgv.data(), imgv.size());
    putH(Xctx, ctxv.data(), (size_t)Lt * CTXD);
    float sigma = rdf(Rd + "/sigma.f32")[0];

    // ---- work buffers ----
    Buf res = vk::buf((size_t)Nt * Dm * 4), QKV = vk::buf((size_t)Nt * 3 * Dm * 2), O = vk::buf((size_t)Na * Dm * 2);
    Buf Hh = vk::buf((size_t)Nt * F * 2), Xa = vk::buf((size_t)Ni * Dm * 2), Yout = vk::buf((size_t)Ni * 128 * 2);
    Buf Qh = vk::buf((size_t)HEADS * Na * 128 * 2), Kh = vk::buf((size_t)HEADS * Na * 128 * 2), Vh = vk::buf((size_t)HEADS * Na * 128 * 2);
    memset(Qh.map, 0, Qh.size); memset(Kh.map, 0, Kh.size); memset(Vh.map, 0, Vh.size);   // rows Nt..Na stay zero
    Buf S = vk::buf((size_t)hc * Na * Na * 2);
    scratch(Nt, F);
    Buf bTemb = vk::buf(256 * 4), bT1 = vk::buf(Dm * 4), bVec = vk::buf(Dm * 4);
    Buf mImg = vk::buf(6 * Dm * 4), mTxt = vk::buf(6 * Dm * 4), mSgl = vk::buf(3 * Dm * 4), mFin = vk::buf(2 * Dm * 4);
    auto part = [&](Buf &m, int i) { return R(m, (size_t)i * Dm * 4, Dm * 4); };
    std::vector<Buf> snap(3);
    for (auto &s : snap) s = vk::buf((size_t)Nt * Dm * 4);
    const float qscale = 1.0f / sqrtf(128.0f);

    // ---- graph 1: modulations from the noise level ----
    vk::Graph gmod; G = &gmod;
    gemv(wh("time_in1"), R(bTemb), R(dummy), R(bT1), Dm, 256, 0);
    gemv(wh("time_in2"), R(bT1), R(dummy), R(bVec), Dm, Dm, 1);
    gemv(wh("mod_img"), R(bVec), R(dummy), R(mImg), 6 * Dm, Dm, 1);
    gemv(wh("mod_txt"), R(bVec), R(dummy), R(mTxt), 6 * Dm, Dm, 1);
    gemv(wh("mod_single"), R(bVec), R(dummy), R(mSgl), 3 * Dm, Dm, 1);
    gemv(wh("final_mod"), R(bVec), R(dummy), R(mFin), 2 * Dm, Dm, 1);

    // ---- graph 2: one step ----
    const bool check = !getenv("DRAW");
    vk::Graph gstep; G = &gstep;
    mulh(3, "img_in", 0, R(Ximg), R(wh("img_in")), R(dummy), R(res), Ni, Dm, 128, 128, 128, Dm);
    mulh(3, "txt_in", 0, R(Xctx), R(wh("txt_in")), R(dummy), R(res, (size_t)Ni * Dm * 4), Lt, Dm, CTXD, CTXD, CTXD, Dm);
    // q / k / v of rows [n0, n0 + n) into the head-major attention inputs
    auto qkvPrep = [&](const std::string &qn, const std::string &kn, int n0, int n) {
        Ref in = R(QKV, (size_t)n0 * 3 * Dm * 2), P = R(bP, (size_t)n0 * 512), Q = R(bQ, (size_t)n0 * 512);
        prep(in, 3 * Dm, 0, R(wf(qn)), P, Q, R(Qh, (size_t)n0 * 256), n, Na, HEADS, true, true, qscale, "prep q");
        prep(in, 3 * Dm, Dm, R(wf(kn)), P, Q, R(Kh, (size_t)n0 * 256), n, Na, HEADS, true, true, 1.0f, "prep k");
        prep(in, 3 * Dm, 2 * Dm, R(dummy), P, Q, R(Vh, (size_t)n0 * 256), n, Na, HEADS, false, false, 1.0f, "prep v");
    };
    struct Stream { const char *s; Buf *m; int n0, n; };
    const Stream st[2] = {{"img", &mImg, 0, Ni}, {"txt", &mTxt, Ni, Lt}};
    for (int i = 0; i < ND; i++) {
        const std::string b = "d" + std::to_string(i) + ".";
        if (i) G->mark();   // segment i: double block i
        for (auto &s : st) {   // ModulationOut order: shift, scale, gate, then the same for the MLP
            const std::string w = b + s.s;
            rotLn(w + "_qkv", R(res, (size_t)s.n0 * Dm * 4), part(*s.m, 0), part(*s.m, 1), Dm, s.n);
            mulq(0, "qkv", w + "_qkv", R(QKV, (size_t)s.n0 * 3 * Dm * 2), R(dummy), R(dummy), s.n);
            qkvPrep(w + "_qn", w + "_kn", s.n0, s.n);
        }
        attend(Qh, Kh, Vh, S, R(O), Na, Nt, HEADS, hc, Dm);
        for (auto &s : st) {
            const std::string w = b + s.s;
            rotX(w + "_proj", R(O, (size_t)s.n0 * Dm * 2), Dm, Dm, s.n);
            mulq(2, "proj", w + "_proj", R(dummy), R(res, (size_t)s.n0 * Dm * 4), part(*s.m, 2), s.n);
        }
        for (auto &s : st) {
            const std::string w = b + s.s;
            rotLn(w + "_mlp_in", R(res, (size_t)s.n0 * Dm * 4), part(*s.m, 3), part(*s.m, 4), Dm, s.n);
            mulq(4, "mlp_in", w + "_mlp_in", R(Hh, (size_t)s.n0 * F * 2), R(dummy), R(dummy), s.n);
            rotX(w + "_mlp_out", R(Hh, (size_t)s.n0 * F * 2), F, F, s.n);
            mulq(2, "mlp_out", w + "_mlp_out", R(dummy), R(res, (size_t)s.n0 * Dm * 4), part(*s.m, 5), s.n);
        }
        if (check && i == 0) copyf(R(res), R(snap[0]), (size_t)Nt * Dm);
        if (check && i == ND - 1) copyf(R(res), R(snap[1]), (size_t)Nt * Dm);
    }
    for (int j = 0; j < NS; j++) {
        const std::string b = "s" + std::to_string(j) + ".";
        G->mark();   // segment ND + j: single block j
        // one rounding of the normed stream feeds both 8-bit multiplies
        rotLn(b + "qkv", R(res), part(mSgl, 0), part(mSgl, 1), Dm, Nt);
        mulq(0, "qkv", b + "qkv", R(QKV), R(dummy), R(dummy), Nt);
        mulq(4, "mlp_in", b + "mlp_in", R(Hh), R(dummy), R(dummy), Nt);
        qkvPrep(b + "qn", b + "kn", 0, Nt);
        attend(Qh, Kh, Vh, S, R(O), Na, Nt, HEADS, hc, Dm);
        rotX(b + "out_a", R(O), Dm, Dm, Nt);
        mulq(2, "out_a", b + "out_a", R(dummy), R(res), part(mSgl, 2), Nt);
        rotX(b + "out_m", R(Hh), F, F, Nt);
        mulq(2, "out_m", b + "out_m", R(dummy), R(res), part(mSgl, 2), Nt);
        if (check && j == 0) copyf(R(res), R(snap[2]), (size_t)Nt * Dm);
    }
    lnmod(R(res), part(mFin, 0), part(mFin, 1), R(Xa), Ni, Dm);
    mulh(0, "final", 0, R(Xa), R(wh("final_linear")), R(Yout), R(dummy), Ni, 128, Dm, Dm, Dm, 128);
    vk::savePipelines();
    printf("setup: %.1f s rearranging weights, %.1f s building shaders\n", tWeights, tKern);
    printf("loaded and recorded in %.1f s: %.0f MB of weights on the chip, %zu dispatches per step\n",
           secsSince(t0), chipBytes / 1e6, gstep.ops.size());
    if (stream) printf("streaming the blocks from storage: %.0f MB of slots\n", slotBytes / 1e6);

    // time signal: [cos, sin](1000 sigma * 10000^(-i/128))
    auto setMod = [&]() {
        float *te = (float *)bTemb.map;
        for (int i = 0; i < 128; i++) {
            const float a = 1000.0f * sigma * expf(-logf(10000.0f) * i / 128.0f);
            te[i] = cosf(a); te[128 + i] = sinf(a);
        }
        gmod.run();
    };
    std::vector<std::pair<std::string, double>> times;
    if (getenv("DRAW")) {
        auto x = rdf(Rd + "/noise.f32"), sg = rdf(Rd + "/sigmas.f32");
        std::vector<float> ref;
        if (fexists(Rd + "/ref.f32")) ref = rdf(Rd + "/ref.f32");
        const size_t Lr = ref.size() / 128;
        if (Lr > (size_t)Ni || x.size() + ref.size() != (size_t)Ni * 128) {
            fprintf(stderr, "noise (%zu rows) + picture to edit (%zu rows) must fill the %d picture rows\n", x.size() / 128, Lr, Ni);
            return 1;
        }
        x.insert(x.end(), ref.begin(), ref.end());
        const size_t Ld = Ni - Lr;   // rows being drawn
        double total = 0;
        for (size_t si = 0; si + 1 < sg.size(); si++) {
            sigma = sg[si];
            auto a = std::chrono::steady_clock::now();
            putH(Ximg, x.data(), x.size());
            setMod();
            runStep(gstep, &times);
            auto y = readH(Yout, (size_t)Ni * 128);
            const double ms = secsSince(a) * 1000;
            total += ms;
            for (size_t k = 0; k < Ld * 128; k++) x[k] += (sg[si + 1] - sg[si]) * y[k];
            printf("draw step %zu: sigma %.3f -> %.3f, %.1f ms\n", si, sg[si], sg[si + 1], ms);
        }
        wrf(Rd + "/draw.f32", x.data(), Ld * 128);
        printf("drawing: %.1f s of steps, latent in %s/draw.f32\n", total / 1000, Rd.c_str());
        if (stream) printf("reading blocks from storage: %.1f s in all (alongside the chip); the chip waited for it %zu times\n", tFill, vk::Graph::stalls);
    } else {
        setMod();
        runStep(gstep);
        const char *files[3] = {"x_double0", "x_double_last", "x_single0"}, *what[3] = {"double 0", "double last", "single 0"};
        for (int s = 0; s < 3; s++) {
            if (!fexists(Rd + "/" + files[s] + ".f32")) continue;
            auto ref = rdf(Rd + "/" + files[s] + ".f32");
            printf("  after %-12s error %.5f\n", what[s], relerr((float *)snap[s].map, ref.data(), (size_t)Nt * Dm));
        }
        auto y = readH(Yout, (size_t)Ni * 128);
        auto ref = rdf(Rd + "/out.f32");
        printf("  final output: error %.5f\n", relerr(y.data(), ref.data(), y.size()));
        for (int l = 0; l < loops; l++) {
            auto a = std::chrono::steady_clock::now();
            runStep(gstep, &times);
            printf("step %d: %.1f ms\n", l, secsSince(a) * 1000);
        }
    }
    printTimes(times);
    (void)N;
    return 0;
}

}  // namespace klein
