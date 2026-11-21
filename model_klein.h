// FLUX.2 Klein 4B (width 3072, 24 heads, 5 double + 20 single blocks) wired from the shared building blocks.
// Token rows: picture patches [0, Ni), prompt [Ni, Ni + Lt), padding up to a multiple of 8 (masked as keys).
// Double blocks keep separate weights per stream but attend jointly; single blocks treat all rows alike. One
// modulation per stream, shared by all blocks, from the noise level. Test runner like Anima's: one reference step
// (tools/ref_klein.py) and its error, then timed loops.
#pragma once
#include "ops.h"

namespace klein {
using namespace ops;
static const int ND = 5, NS = 20, Dm = 3072, NHEADS = 24, F = 9216, CTXD = 7680;

static int run(const std::string &R, int loops) {
    int Ni = 0, Lt = 0, Lp = 0, PW = 0, PH = 0;
    if (FILE *f = fopen((R + "/info.txt").c_str(), "r")) {
        if (fscanf(f, "%d %d %d %d %d", &Ni, &Lt, &Lp, &PW, &PH) != 5) Ni = 0;
        fclose(f);
    }
    if (!Ni) { fprintf(stderr, "%s/info.txt missing or bad\n", R.c_str()); return 1; }
    const int N = Ni + Lt + Lp, Lv = Ni + Lt;
    ops::init(NHEADS, Dm, N, F);
    printf("%dx%d: %d picture + %d prompt + %d padding tokens, attention %d heads at a time\n", PW, PH, Ni, Lt, Lp, HC);

    // ---- inputs: picture patches in rows [0, Ni), prompt in rows [Ni, Ni + Lt) of their own images ----
    auto imgv = rdf(R + "/img.f32"), ctxv = rdf(R + "/ctx.f32");
    auto Pt = rdf(R + "/rope_P.f32"), Qt = rdf(R + "/rope_Q.f32");
    std::vector<float> ia((size_t)N * 128, 0.0f), ca((size_t)N * CTXD, 0.0f);
    std::copy(imgv.begin(), imgv.end(), ia.begin());
    std::copy(ctxv.begin(), ctxv.end(), ca.begin() + (size_t)Ni * CTXD);
    cl_mem Ximg = toX(ia, N, 128), Xctx = toX(ca, N, CTXD);
    cl_mem bP = buf(Pt.size() * 4, Pt.data()), bQ = buf(Qt.size() * 4, Qt.data());
    float sigma = 0.0f;
    { auto s = rdf(R + "/sigma.f32"); sigma = s[0]; }

    // ---- activations ----
    cl_mem res = buf((size_t)N * Dm * 4);
    cl_mem QKV = img(N, 3 * Dm / 8), O = img(N, Dm / 8), Hh = img(N, F / 8), Yout = img(N, 128 / 8);
    cl_mem bTemb = buf(256 * 4), bT1 = buf(Dm * 4), bVec = buf(Dm * 4);
    cl_mem mImg = buf(6 * Dm * 4), mTxt = buf(6 * Dm * 4), mSgl = buf(3 * Dm * 4), mFin = buf(2 * Dm * 4);
    auto part = [&](cl_mem m, int i) { return sub(m, (size_t)i * Dm * 4, Dm * 4); };
    // ModulationOut order: shift, scale, gate (x2 for double blocks); the final layer: shift, scale
    cl_mem iSh1 = part(mImg, 0), iSc1 = part(mImg, 1), iG1 = part(mImg, 2), iSh2 = part(mImg, 3), iSc2 = part(mImg, 4), iG2 = part(mImg, 5);
    cl_mem tSh1 = part(mTxt, 0), tSc1 = part(mTxt, 1), tG1 = part(mTxt, 2), tSh2 = part(mTxt, 3), tSc2 = part(mTxt, 4), tG2 = part(mTxt, 5);
    cl_mem sSh = part(mSgl, 0), sSc = part(mSgl, 1), sG = part(mSgl, 2), fSh = part(mFin, 0), fSc = part(mFin, 1);
    const float zero = 0.0f;

    auto check = [&](const char *what, const char *file) {
        std::vector<float> raw((size_t)N * Dm), out((size_t)Lv * Dm);
        CK(clEnqueueReadBuffer(q, res, CL_TRUE, 0, raw.size() * 4, raw.data(), 0, 0, 0));
        for (int og = 0; og < Dm / 4; og++) for (int n = 0; n < Lv; n++) for (int j = 0; j < 4; j++)
            out[(size_t)n * Dm + og * 4 + j] = raw[((size_t)og * N + n) * 4 + j];
        auto ref = rdf(R + "/" + file + ".f32"); ref.resize((size_t)Lv * Dm);
        printf("  after %-12s error %.5f\n", what, relerr(out, ref));
    };

    // KDBG=1: largest |value| and NaN count of the stream and the fp16 work images through the first block
    const bool dbg = getenv("KDBG") != nullptr;
    auto statR = [&](const char *what) {
        if (!dbg) return;
        std::vector<float> raw((size_t)N * Dm);
        CK(clEnqueueReadBuffer(q, res, CL_TRUE, 0, raw.size() * 4, raw.data(), 0, 0, 0));
        double mx = 0; size_t bad = 0;
        for (float v : raw) { if (!std::isfinite(v)) bad++; else mx = std::max(mx, (double)fabsf(v)); }
        printf("    %-16s stream max %.1f, not finite %zu\n", what, mx, bad);
    };
    auto statI = [&](const char *what, cl_mem im, int texels) {
        if (!dbg) return;
        std::vector<h16> y((size_t)N * texels * 8);
        size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)N, (size_t)texels, 1};
        CK(clEnqueueReadImage(q, im, CL_TRUE, org, reg, 0, 0, y.data(), 0, 0, 0));
        double mx = 0; size_t bad = 0;
        for (h16 v : y) { float f = (float)v; if (!std::isfinite(f)) bad++; else mx = std::max(mx, (double)fabsf(f)); }
        printf("    %-16s max %.1f, not finite %zu\n", what, mx, bad);
    };
    auto step = [&](bool chk) {
        // time signal: [cos, sin](1000 sigma * 10000^(-i/128)) -> time_in -> vec; the modulations from silu(vec)
        std::vector<float> te(256);
        for (int i = 0; i < 128; i++) {
            const float a = 1000.0f * sigma * expf(-logf(10000.0f) * i / 128.0f);
            te[i] = cosf(a); te[128 + i] = sinf(a);
        }
        CK(clEnqueueWriteBuffer(q, bTemb, CL_FALSE, 0, 256 * 4, te.data(), 0, 0, 0));
        gemv("gemv", w("time_in1"), bTemb, nullptr, bT1, Dm, 256, 0);
        gemv("gemv", w("time_in2"), bT1, nullptr, bVec, Dm, Dm, 1);
        gemv("gemv", w("mod_img"), bVec, nullptr, mImg, 6 * Dm, Dm, 1);
        gemv("gemv", w("mod_txt"), bVec, nullptr, mTxt, 6 * Dm, Dm, 1);
        gemv("gemv", w("mod_single"), bVec, nullptr, mSgl, 3 * Dm, Dm, 1);
        gemv("gemv", w("final_mod"), bVec, nullptr, mFin, 2 * Dm, Dm, 1);

        CK(clEnqueueFillBuffer(q, res, &zero, 4, 0, (size_t)N * Dm * 4, 0, 0, 0));
        linear(3, "img_in", Ximg, "img_in", N, Dm, 128, res, nullptr, 0, Ni);
        linear(3, "txt_in", Xctx, "txt_in", N, Dm, CTXD, res, nullptr, Ni, Lv);
        if (chk) statR("embedded");
        for (int i = 0; i < ND; i++) {
            const std::string b = "d" + std::to_string(i) + ".";
            normLinear(0, "qkv", res, iSh1, iSc1, b + "img_qkv", N, 3 * Dm, Dm, QKV, nullptr, 0, Ni);
            normLinear(0, "qkv", res, tSh1, tSc1, b + "txt_qkv", N, 3 * Dm, Dm, QKV, nullptr, Ni, N);
            attnPrep(QKV, N, 0, Ni, 0, Dm / 8, 2 * Dm / 8, w(b + "img_qn"), w(b + "img_kn"), bP, bQ);
            attnPrep(QKV, N, Ni, N, 0, Dm / 8, 2 * Dm / 8, w(b + "txt_qn"), w(b + "txt_kn"), bP, bQ);
            if (chk && i == 0) statI("qkv", QKV, 3 * Dm / 8);
            attnCore(N, O, Lv);
            if (chk && i == 0) statI("attention", O, Dm / 8);
            linear(2, "proj", O, b + "img_proj", N, Dm, Dm, res, iG1, 0, Ni);
            linear(2, "proj", O, b + "txt_proj", N, Dm, Dm, res, tG1, Ni, N);
            normLinear(4, "mlp_in", res, iSh2, iSc2, b + "img_mlp_in", N, 2 * F, Dm, Hh, nullptr, 0, Ni);
            if (chk && i == 0) statI("mlp hidden img", Hh, F / 8);
            linear(2, "mlp_out", Hh, b + "img_mlp_out", N, Dm, F, res, iG2, 0, Ni);
            normLinear(4, "mlp_in", res, tSh2, tSc2, b + "txt_mlp_in", N, 2 * F, Dm, Hh, nullptr, Ni, N);
            linear(2, "mlp_out", Hh, b + "txt_mlp_out", N, Dm, F, res, tG2, Ni, N);
            if (chk && i == 0) { statR("after double 0"); check("double 0", "x_double0"); }
            if (chk && i == ND - 1) check("double last", "x_double_last");
        }
        for (int j = 0; j < NS; j++) {
            const std::string b = "s" + std::to_string(j) + ".";
            // one rounding of the normed stream feeds both 8-bit multiplies
            lnq(b + "qkv", res, sSh, sSc, N, Dm);
            gemmq(0, "qkv", b + "qkv", N, 3 * Dm, Dm, QKV, nullptr);
            gemmq(4, "mlp_in", b + "mlp_in", N, 2 * F, Dm, Hh, nullptr);
            attnPrep(QKV, N, 0, N, 0, Dm / 8, 2 * Dm / 8, w(b + "qn"), w(b + "kn"), bP, bQ);
            attnCore(N, O, Lv);
            linear(2, "out_a", O, b + "out_a", N, Dm, Dm, res, sG);
            linear(2, "out_m", Hh, b + "out_m", N, Dm, F, res, sG);
            if (chk && j == 0) check("single 0", "x_single0");
        }
        lnmod(res, fSh, fSc, Xa, N, Dm, 0, Ni);
        gemm(0, "gemm final", Xa, w("final_linear"), Ni, 128, Dm, Yout, nullptr, 1, 0, 0, 0, 0, 0, 0, 0, 0, N);
    };

    // DRAW=1: the whole picture from <ref>/noise.f32 along <ref>/sigmas.f32 (Euler), latent to <ref>/draw.f32
    if (getenv("DRAW")) {
        auto x = rdf(R + "/noise.f32"), sg = rdf(R + "/sigmas.f32");
        // an edit: <ref>/ref.f32 is the picture to edit (Lr x 128), its rows are the last Lr picture rows of the
        // stream (positions with index 10, as ComfyUI's "index" reference method); they are put back after every step
        std::vector<float> ref;
        if (FILE *rf = fopen((R + "/ref.f32").c_str(), "rb")) { fclose(rf); ref = rdf(R + "/ref.f32"); }
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
            std::copy(x.begin(), x.end(), ia.begin());
            clReleaseMemObject(Ximg); Ximg = toX(ia, N, 128);
            clFinish(q); auto a = std::chrono::steady_clock::now();
            step(false);
            std::vector<h16> y((size_t)N * 128);
            size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)N, 16, 1};
            CK(clEnqueueReadImage(q, Yout, CL_TRUE, org, reg, 0, 0, y.data(), 0, 0, 0));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
            total += ms;
            for (int k8 = 0; k8 < 16; k8++) for (size_t n = 0; n < Ld; n++) for (int j = 0; j < 8; j++)
                x[n * 128 + k8 * 8 + j] += (sg[si + 1] - sg[si]) * (float)y[((size_t)k8 * N + n) * 8 + j];
            printf("draw step %zu: sigma %.3f -> %.3f, %.1f ms\n", si, sg[si], sg[si + 1], ms); fflush(stdout);
        }
        FILE *f = fopen((R + "/draw.f32").c_str(), "wb");
        fwrite(x.data(), 4, Ld * 128, f); fclose(f);
        printf("drawing: %.1f s of steps, latent in %s/draw.f32\n", total / 1000, R.c_str());
        return 0;
    }
    step(true);
    {
        std::vector<h16> y((size_t)N * 128);
        size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)N, 16, 1};
        CK(clEnqueueReadImage(q, Yout, CL_TRUE, org, reg, 0, 0, y.data(), 0, 0, 0));
        std::vector<float> out((size_t)Ni * 128);
        for (int k8 = 0; k8 < 16; k8++) for (int n = 0; n < Ni; n++) for (int j = 0; j < 8; j++)
            out[(size_t)n * 128 + k8 * 8 + j] = y[((size_t)k8 * N + n) * 8 + j];
        printf("  final output: error %.5f\n", relerr(out, rdf(R + "/out.f32")));
    }
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

}  // namespace klein
