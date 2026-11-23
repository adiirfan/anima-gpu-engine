// Krea 2 Turbo (12B single-stream DiT: width 6144, 48 query heads sharing 12 key / value heads, 28 blocks) wired from
// the shared building blocks (tools/pack_krea.py makes the file). Token rows: picture patches [0, Ni), prompt
// [Ni, Ni + Lt), padding up to a multiple of 8 (masked as keys). Every block: RMS norm, modulation, one input multiply
// [q; k; v; gate], attention with the output gated by sigmoid(gate), then a SwiGLU MLP; the modulation is shared
// (from the noise level) plus a per-block table. The block norms' weights (1 + w) are folded into the modulation's
// scale on the host once per step.
// Inputs in <ref>: engine.txt "Ni Lt Lp W H", txt.f32 (Lt x 6144: the prompt after Krea's text fusion and text MLP),
// sigmas.f32; the test step also reads img0.f32 (Ni x 64 patches), t / tvec / x_block0 / x_blocklast / out0p
// (tools/ref_krea.py). DRAW=1: the whole latent from noise.f32 (16 x h x w) into draw.f32 (h x w x 16, channels
// innermost, as the small Wan 2.1 decoder reads it).
#pragma once
#include "ops.h"

namespace krea {
using namespace ops;
static const int NB = 28, Dm = 6144, NHEADS = 48, KVD = 1536, F = 16384, QKVG = Dm + 2 * KVD + Dm;

static std::vector<float> host(const std::string &n) {   // an fp32 table back from the chip (small ones only)
    const Wt &t = W.at(n);
    std::vector<float> v((size_t)t.M * t.K);
    CK(clEnqueueReadBuffer(q, t.m, CL_TRUE, 0, v.size() * 4, v.data(), 0, 0, 0));
    return v;
}

// Krea's text fusion, once per prompt: the 12 tapped reader layers of every prompt token (ctx: Lt x 12 x 2560) ->
// the prompt rows the drawing blocks see (Lt x 6144). Two blocks across each token's 12 layers (attention inside the
// group of 12), the projector 12 -> 1 (on the host), two blocks across the prompt, then RMS norm, linear, GELU, linear.
// Blocks: RMS norm (1 + w), one input multiply [q; k; v; gate], gated attention, SwiGLU; no modulation, no rope.
static std::vector<float> fuse(const std::vector<float> &ctx, int Lt) {
    const int TD = 2560, TF = 6912, NL = 12, TQ = 4 * TD, OD = 6144;
    const int Nt = (Lt * NL + 7) / 8 * 8;
    RMSN = 1; KVREP = 1; LNEPS = 1e-5f;
    ops::init(20, TD, Nt, TF);
    std::vector<float> r((size_t)Nt * TD, 0.0f);   // row t * 12 + l = token t's reader layer l
    for (int n = 0; n < Lt * NL; n++) for (int d = 0; d < TD; d++) r[((size_t)(d / 4) * Nt + n) * 4 + d % 4] = ctx[(size_t)n * TD + d];
    cl_mem res = buf(r.size() * 4, r.data());
    std::vector<float> z(TD, 0.0f), o(TD, 1.0f);
    cl_mem zero = buf(TD * 4, z.data()), ones = buf(TD * 4, o.data());
    cl_mem QKV = img(Nt, TQ / 8), O = img(Nt, TD / 8), Og = img(Nt, TD / 8), Hh = img(Nt, TF / 8);
    auto block = [&](const std::string &b, int grp, int n1) {
        normLinear(0, "qkvg", res, zero, w(b + "n1"), b + "qkvg", Nt, TQ, TD, QKV, nullptr, 0, n1);
        grpAttn(QKV, n1, grp, 0, TD / 8, 2 * TD / 8, w(b + "qn"), w(b + "kn"), O);
        sgate(O, QKV, 3 * TD / 8, n1, TD, Og);
        linear(2, "wo", Og, b + "wo", Nt, TD, TD, res, ones, 0, n1);
        normLinear(4, "mlp_in", res, zero, w(b + "n2"), b + "mlp_in", Nt, 2 * TF, TD, Hh, nullptr, 0, n1);
        linear(2, "down", Hh, b + "down", Nt, TD, TF, res, ones, 0, n1);
    };
    block("tl0.", NL, Lt * NL);
    block("tl1.", NL, Lt * NL);
    {   // projector: each token's 12 layer rows -> one row (weights tf.proj), rows [0, Lt) of the stream
        CK(clEnqueueReadBuffer(q, res, CL_TRUE, 0, r.size() * 4, r.data(), 0, 0, 0));
        const auto pw = host("tf.proj");
        std::vector<float> y((size_t)Nt * TD, 0.0f);
        for (int g = 0; g < TD / 4; g++) for (int t = 0; t < Lt; t++) for (int j = 0; j < 4; j++) {
            float a = 0;
            for (int l = 0; l < NL; l++) a += pw[l] * r[((size_t)g * Nt + t * NL + l) * 4 + j];
            y[((size_t)g * Nt + t) * 4 + j] = a;
        }
        CK(clEnqueueWriteBuffer(q, res, CL_TRUE, 0, y.size() * 4, y.data(), 0, 0, 0));
    }
    block("tr0.", Lt, Lt);
    block("tr1.", Lt, Lt);
    cl_mem Y1 = img(Nt, OD / 8), Y2 = img(Nt, OD / 8), Y3 = img(Nt, OD / 8);
    normLinear(0, "txt.l1", res, zero, w("txt.n"), "txt.l1", Nt, OD, TD, Y1, nullptr, 0, Lt);
    biasGelu(Y1, w("txt.l1.b"), Lt, OD, Y2);
    linear(0, "txt.l2", Y2, "txt.l2", Nt, OD, OD, Y3, nullptr, 0, Lt);
    std::vector<h16> y((size_t)Nt * OD);
    size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)Nt, (size_t)OD / 8, 1};
    CK(clEnqueueReadImage(q, Y3, CL_TRUE, org, reg, 0, 0, y.data(), 0, 0, 0));
    const auto b2 = host("txt.l2.b");
    std::vector<float> out((size_t)Lt * OD);
    for (int c8 = 0; c8 < OD / 8; c8++) for (int t = 0; t < Lt; t++) for (int j = 0; j < 8; j++)
        out[(size_t)t * OD + c8 * 8 + j] = (float)y[((size_t)c8 * Nt + t) * 8 + j] + b2[c8 * 8 + j];
    for (cl_mem m : {res, zero, ones, Y1, Y2, Y3}) clReleaseMemObject(m);
    for (cl_mem m : {QKV, O, Og, Hh}) clReleaseMemObject(m);
    return out;
}

static int run(const std::string &R, int loops) {
    int Ni = 0, Lt = 0, Lp = 0, PW = 0, PH = 0;
    if (FILE *f = fopen((R + "/engine.txt").c_str(), "r")) {
        if (fscanf(f, "%d %d %d %d %d", &Ni, &Lt, &Lp, &PW, &PH) != 5) Ni = 0;
        fclose(f);
    }
    if (!Ni) { fprintf(stderr, "%s/engine.txt missing or bad\n", R.c_str()); return 1; }
    const int N = Ni + Lt + Lp, Lv = Ni + Lt, hp = PH / 16, wp = PW / 16;   // patch grid
    if (hp * wp != Ni) { fprintf(stderr, "engine.txt: %d patches but %dx%d\n", Ni, PW, PH); return 1; }
    // the prompt rows: made here from the reader's layers (ctx.f32) when there is no txt.f32 or KTEXT=1
    std::vector<float> txtv;
    const bool haveTxt = access((R + "/txt.f32").c_str(), R_OK) == 0;
    if (access((R + "/ctx.f32").c_str(), R_OK) == 0 && (!haveTxt || getenv("KTEXT"))) {
        auto ctx = rdf(R + "/ctx.f32");
        if (ctx.size() != (size_t)Lt * 12 * 2560) { fprintf(stderr, "ctx.f32: %zu values, not %d x 12 x 2560\n", ctx.size(), Lt); return 1; }
        auto a = std::chrono::steady_clock::now();
        txtv = fuse(ctx, Lt);
        printf("prompt: %d tokens through the text fusion, %.1f ms", Lt,
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
        if (haveTxt) printf(", error %.5f", relerr(txtv, rdf(R + "/txt.f32")));
        printf("\n");
    } else txtv = rdf(R + "/txt.f32");
    RMSN = 1; KVREP = NHEADS / (KVD / 128); LNEPS = 1e-5f;
    ops::init(NHEADS, Dm, N, F);
    printf("%dx%d: %d picture + %d prompt + %d padding tokens, attention %d heads at a time\n", PW, PH, Ni, Lt, Lp, HC);

    // ---- the stream's starting rows: picture rows get img_in's bias (img_in adds onto it), prompt rows the prompt ----
    if (txtv.size() != (size_t)Lt * Dm) { fprintf(stderr, "txt.f32: %zu values, not %d x %d\n", txtv.size(), Lt, Dm); return 1; }
    std::vector<float> r0((size_t)N * Dm, 0.0f);   // channel groups first: r0[(og * N + n) * 4 + j]
    {
        auto fb = host("first.b");
        for (int og = 0; og < Dm / 4; og++) for (int j = 0; j < 4; j++) {
            for (int n = 0; n < Ni; n++) r0[((size_t)og * N + n) * 4 + j] = fb[og * 4 + j];
            for (int n = 0; n < Lt; n++) r0[((size_t)og * N + Ni + n) * 4 + j] = txtv[(size_t)n * Dm + og * 4 + j];
        }
    }
    cl_mem res0 = buf(r0.size() * 4, r0.data()), res = buf(r0.size() * 4);
    std::vector<float> ones(Dm, 1.0f);
    cl_mem bOnes = buf(Dm * 4, ones.data());

    // ---- rope (3 axes 32 / 48 / 48 of each head's 64 pairs, theta 1000): picture (0, y, x), prompt and padding at 0,
    // in rotate-half form (pack_krea.py reorders q / k channels): P = cos, cos; Q = -sin, sin ----
    std::vector<float> Pt((size_t)N * 128), Qt((size_t)N * 128);
    for (int n = 0; n < N; n++) for (int i = 0; i < 64; i++) {
        double ang = 0;
        if (n < Ni) {
            const int py = n / wp, px = n % wp;
            if (i >= 16 && i < 40) ang = py * pow(1000.0, -2.0 * (i - 16) / 48);
            else if (i >= 40) ang = px * pow(1000.0, -2.0 * (i - 40) / 48);
        }
        Pt[(size_t)n * 128 + i] = Pt[(size_t)n * 128 + 64 + i] = (float)cos(ang);
        Qt[(size_t)n * 128 + i] = (float)-sin(ang); Qt[(size_t)n * 128 + 64 + i] = (float)sin(ang);
    }
    cl_mem bP = buf(Pt.size() * 4, Pt.data()), bQ = buf(Qt.size() * 4, Qt.data());

    // ---- activations ----
    cl_mem QKV = img(N, QKVG / 8), O = img(N, Dm / 8), Og = img(N, Dm / 8), Hh = img(N, F / 8), Yout = img(N, 128 / 8);
    cl_mem bTemb = buf(256 * 4), bT1 = buf(Dm * 4), bT = buf(Dm * 4), bVec = buf(6 * Dm * 4);
    cl_mem bMod = buf((size_t)NB * 6 * Dm * 4), bFin = buf(2 * Dm * 4);
    std::vector<cl_mem> mod(NB * 6);
    for (int i = 0; i < NB * 6; i++) mod[i] = sub(bMod, (size_t)i * Dm * 4, Dm * 4);
    cl_mem fSc = sub(bFin, 0, Dm * 4), fSh = sub(bFin, Dm * 4, Dm * 4);
    // the per-block tables the modulation is made from (fp32, a few MB)
    std::vector<std::vector<float>> lin(NB), n1(NB), n2(NB);
    for (int i = 0; i < NB; i++) {
        const std::string m = "m" + std::to_string(i) + ".";
        lin[i] = host(m + "lin"); n1[i] = host(m + "n1"); n2[i] = host(m + "n2");
    }
    const auto lastLin = host("last.lin"), lastN = host("last.n"), lastB = host("last.b");
    std::vector<float> hT(Dm), hVec(6 * Dm), hMod((size_t)NB * 6 * Dm), hFin(2 * Dm);
    float sigma = 0.0f;

    auto check = [&](const char *what, const char *file) {
        std::vector<float> raw((size_t)N * Dm), out((size_t)Lv * Dm);
        CK(clEnqueueReadBuffer(q, res, CL_TRUE, 0, raw.size() * 4, raw.data(), 0, 0, 0));
        for (int og = 0; og < Dm / 4; og++) for (int n = 0; n < Lv; n++) for (int j = 0; j < 4; j++)
            out[(size_t)n * Dm + og * 4 + j] = raw[((size_t)og * N + n) * 4 + j];
        auto ref = rdf(R + "/" + file + ".f32"); ref.resize((size_t)Lv * Dm);
        printf("  after %-12s error %.5f\n", what, relerr(out, ref));
    };
    cl_mem Ximg = nullptr;
    auto setPicture = [&](const std::vector<float> &patches) {   // Ni x 64 -> rows of 128 (zero-padded) in an X image
        std::vector<float> ia((size_t)N * 128, 0.0f);
        for (int n = 0; n < Ni; n++) std::copy(patches.begin() + (size_t)n * 64, patches.begin() + (size_t)(n + 1) * 64, ia.begin() + (size_t)n * 128);
        if (Ximg) clReleaseMemObject(Ximg);
        Ximg = toX(ia, N, 128);
    };
    auto step = [&](bool chk) {
        // noise level -> t (tmlp: linear, GELU, linear) -> tvec (tproj: GELU, linear); the modulations on the host
        std::vector<float> te(256);
        for (int i = 0; i < 128; i++) {
            const float a = 1000.0f * sigma * expf(-logf(10000.0f) * i / 128.0f);
            te[i] = cosf(a); te[128 + i] = sinf(a);
        }
        CK(clEnqueueWriteBuffer(q, bTemb, CL_FALSE, 0, 256 * 4, te.data(), 0, 0, 0));
        gemv("gemv", w("t1"), bTemb, w("t1.b"), bT1, Dm, 256, 0);
        gemv("gemv", w("t2"), bT1, w("t2.b"), bT, Dm, Dm, 2);
        gemv("gemv", w("tproj"), bT, w("tproj.b"), bVec, 6 * Dm, Dm, 2);
        CK(clEnqueueReadBuffer(q, bT, CL_TRUE, 0, Dm * 4, hT.data(), 0, 0, 0));
        CK(clEnqueueReadBuffer(q, bVec, CL_TRUE, 0, 6 * Dm * 4, hVec.data(), 0, 0, 0));
        if (chk) {
            printf("  noise level: t error %.5f, modulation error %.5f\n", relerr(hT, rdf(R + "/t.f32")), relerr(hVec, rdf(R + "/tvec.f32")));
        }
        for (int i = 0; i < NB; i++) {
            float *m = hMod.data() + (size_t)i * 6 * Dm;
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
        for (int c = 0; c < Dm; c++) {   // output layer: scale, shift from t (not tvec)
            hFin[c] = (1 + lastN[c]) * (1 + hT[c] + lastLin[c]) - 1;
            hFin[Dm + c] = hT[c] + lastLin[Dm + c];
        }
        CK(clEnqueueWriteBuffer(q, bMod, CL_FALSE, 0, hMod.size() * 4, hMod.data(), 0, 0, 0));
        CK(clEnqueueWriteBuffer(q, bFin, CL_FALSE, 0, hFin.size() * 4, hFin.data(), 0, 0, 0));

        CK(clEnqueueCopyBuffer(q, res0, res, 0, 0, (size_t)N * Dm * 4, 0, 0, 0));
        linear(2, "first", Ximg, "first", N, Dm, 128, res, bOnes, 0, Ni);
        for (int i = 0; i < NB; i++) {
            const std::string b = "b" + std::to_string(i) + ".";
            if (STREAM) streamer.use(i, NB);
            cl_mem *m = &mod[(size_t)i * 6];
            normLinear(0, "qkvg", res, m[1], m[0], b + "qkvg", N, QKVG, Dm, QKV);
            attnPrep(QKV, N, 0, N, 0, Dm / 8, (Dm + KVD) / 8, w(b + "qn"), w(b + "kn"), bP, bQ);
            attnCore(N, O, Lv);
            sgate(O, QKV, (Dm + 2 * KVD) / 8, N, Dm, Og);
            linear(2, "wo", Og, b + "wo", N, Dm, Dm, res, m[2]);
            normLinear(4, "mlp_in", res, m[4], m[3], b + "mlp_in", N, 2 * F, Dm, Hh);
            linear(2, "down", Hh, b + "down", N, Dm, F, res, m[5]);
            if (chk && i == 0) check("block 0", "x_block0");
            if (chk && i == NB - 1) check("last block", "x_blocklast");
        }
        normLinear(0, "last", res, fSh, fSc, "last", N, 128, Dm, Yout, nullptr, 0, Ni);
    };
    auto output = [&]() {   // the step's velocity, Ni x 64 patches (bias added here)
        std::vector<h16> y((size_t)N * 128);
        size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)N, 16, 1};
        CK(clEnqueueReadImage(q, Yout, CL_TRUE, org, reg, 0, 0, y.data(), 0, 0, 0));
        std::vector<float> v((size_t)Ni * 64);
        for (int k8 = 0; k8 < 8; k8++) for (int n = 0; n < Ni; n++) for (int j = 0; j < 8; j++)
            v[(size_t)n * 64 + k8 * 8 + j] = (float)y[((size_t)k8 * N + n) * 8 + j] + lastB[k8 * 8 + j];
        return v;
    };
    auto sg = rdf(R + "/sigmas.f32");

    // DRAW=1: Euler along sigmas.f32 from noise.f32 (16 x h x w, the model's latent space)
    if (getenv("DRAW")) {
        auto nz = rdf(R + "/noise.f32");
        const int h = 2 * hp, wd = 2 * wp;
        if (nz.size() != (size_t)16 * h * wd) { fprintf(stderr, "noise.f32: %zu values, not 16 x %d x %d\n", nz.size(), h, wd); return 1; }
        std::vector<float> x((size_t)Ni * 64);   // patches: feature c * 4 + dy * 2 + dx of patch (py, px)
        auto pidx = [&](int c, int y, int xx) { return ((size_t)(y / 2) * wp + xx / 2) * 64 + c * 4 + (y % 2) * 2 + xx % 2; };
        for (int c = 0; c < 16; c++) for (int y = 0; y < h; y++) for (int xx = 0; xx < wd; xx++)
            x[pidx(c, y, xx)] = nz[((size_t)c * h + y) * wd + xx];
        double total = 0;
        for (size_t si = 0; si + 1 < sg.size(); si++) {
            sigma = sg[si];
            setPicture(x);
            clFinish(q); auto a = std::chrono::steady_clock::now();
            step(false);
            auto v = output();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
            total += ms;
            for (size_t k = 0; k < x.size(); k++) x[k] += (sg[si + 1] - sg[si]) * v[k];
            printf("draw step %zu: sigma %.3f -> %.3f, %.1f ms\n", si, sg[si], sg[si + 1], ms); fflush(stdout);
        }
        std::vector<float> lat((size_t)h * wd * 16);
        for (int c = 0; c < 16; c++) for (int y = 0; y < h; y++) for (int xx = 0; xx < wd; xx++)
            lat[((size_t)y * wd + xx) * 16 + c] = x[pidx(c, y, xx)];
        FILE *f = fopen((R + "/draw.f32").c_str(), "wb");
        fwrite(lat.data(), 4, lat.size(), f); fclose(f);
        printf("drawing: %.1f s of steps, latent in %s/draw.f32\n", total / 1000, R.c_str());
        return 0;
    }
    // test: the first step against the reference
    sigma = sg[0];
    setPicture(rdf(R + "/img0.f32"));
    step(true);
    printf("  final output: error %.5f\n", relerr(output(), rdf(R + "/out0p.f32")));
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

}  // namespace krea
