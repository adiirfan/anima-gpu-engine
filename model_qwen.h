// Qwen-Image 2.1 (width 4096, 32 heads, 32 blocks, SwiGLU 12288) wired from the shared building blocks.
// The prompt's rows are modulated at t = 0 and see only earlier prompt rows, so they do not depend on the step: one
// prompt pass per picture runs them through the blocks (causal attention) and keeps each block's keys / values.
// Every step then runs only the picture rows, attending to [picture keys, kept prompt keys] (padding masked).
// Modulation per step: scale / gate pairs shared by all blocks (no shift), gates through tanh; the output layer's
// norm scale from its own projection. Test runner: tools/ref_qwen.py's folder (prompt features, noise, schedule,
// first-step velocity, final latent); DRAW=1 for the whole picture.
#pragma once
#include "ops.h"

namespace qwen {
using namespace ops;
static const int NB = 32, Dm = 4096, NHEADS = 32, F = 12288, T = 128;

// rotations as the engine applies them (rotate-half, pair i at channels i and 64 + i): P = [cos, cos], Q = [-sin, sin];
// axes 16 / 56 / 56, theta 10000
static void rope(const std::vector<float> &ids3, int n, std::vector<float> &P, std::vector<float> &Q) {
    const int dims[3] = {16, 56, 56};
    P.assign((size_t)n * 128, 0.0f); Q.assign((size_t)n * 128, 0.0f);
    for (int r = 0; r < n; r++) {
        int k = 0;
        for (int a = 0; a < 3; a++)
            for (int i = 0; i < dims[a] / 2; i++, k++) {
                const double ang = ids3[(size_t)r * 3 + a] * std::pow(10000.0, -2.0 * i / dims[a]);
                P[(size_t)r * 128 + k] = P[(size_t)r * 128 + 64 + k] = (float)std::cos(ang);
                Q[(size_t)r * 128 + k] = -(float)std::sin(ang); Q[(size_t)r * 128 + 64 + k] = (float)std::sin(ang);
            }
    }
}

static int run(const std::string &R, int loops) {
    int nc = 0, PW = 0, PH = 0;
    if (FILE *f = fopen((R + "/info.txt").c_str(), "r")) {
        if (fscanf(f, "%d %d %d", &nc, &PW, &PH) != 3) nc = 0;
        fclose(f);
    }
    if (!nc) { fprintf(stderr, "%s/info.txt missing or bad\n", R.c_str()); return 1; }
    const int gh = PH / 16, gw = PW / 16, Ni = gh * gw, LK = Ni + T;
    if (Ni % 8) { fprintf(stderr, "picture tokens must be a multiple of 8\n"); return 1; }
    if (!getenv("QGS")) QGS = 32;   // 4-bit weights carry one scale per 32 inputs; the activations follow
    ops::init(NHEADS, Dm, LK, F);
    printf("%dx%d: %d picture tokens, prompt %d of %d rows, attention %d heads at a time\n", PW, PH, Ni, nc, T, HC);

    // ---- rotations: prompt rows (p, p, p), picture rows (nc, y, x) centred ----
    std::vector<float> idt((size_t)T * 3), idi((size_t)Ni * 3), Pt, Qt, Pi, Qi;
    for (int p = 0; p < T; p++) idt[p * 3] = idt[p * 3 + 1] = idt[p * 3 + 2] = (float)p;
    for (int y = 0; y < gh; y++) for (int x = 0; x < gw; x++) {
        float *pp = &idi[((size_t)y * gw + x) * 3];
        pp[0] = (float)nc; pp[1] = (float)(y - (gh - gh / 2)); pp[2] = (float)(x - (gw - gw / 2));
    }
    rope(idt, T, Pt, Qt); rope(idi, Ni, Pi, Qi);
    cl_mem bPt = buf(Pt.size() * 4, Pt.data()), bQt = buf(Qt.size() * 4, Qt.data());
    cl_mem bPi = buf(Pi.size() * 4, Pi.data()), bQi = buf(Qi.size() * 4, Qi.data());

    // ---- buffers ----
    cl_mem res = buf((size_t)Ni * Dm * 4), resT = buf((size_t)T * Dm * 4);
    cl_mem QKV = img(Ni, 3 * Dm / 8), O = img(Ni, Dm / 8), Hh = img(Ni, F / 8), Yout = img(Ni, 64 / 8);
    cl_mem bTemb = buf(256 * 4), bT1 = buf(Dm * 4), bTe = buf(Dm * 4), bMod = buf(4 * Dm * 4), bSc = buf(Dm * 4);
    cl_mem mods = buf(4 * Dm * 4), mods0 = buf(4 * Dm * 4), fin = buf(Dm * 4);
    std::vector<float> zeros(Dm, 0.0f);
    cl_mem bZero = buf(Dm * 4, zeros.data());
    auto part = [&](cl_mem m, int i) { return sub(m, (size_t)i * Dm * 4, Dm * 4); };
    cl_mem s1 = part(mods, 0), g1 = part(mods, 1), s2 = part(mods, 2), g2 = part(mods, 3);
    cl_mem s10 = part(mods0, 0), g10 = part(mods0, 1), s20 = part(mods0, 2), g20 = part(mods0, 3);
    std::vector<cl_mem> cK(NB), cV(NB);
    for (int b = 0; b < NB; b++) { cK[b] = img(T / 4, NHEADS * 64); cV[b] = img(NHEADS * 32, T / 2); }

    // QBENCH=1: the same multiply (b0.qkv's shape) with 4-bit and with random 8-bit weights, alternating, so heat
    // and clock changes hit both alike. Prints the chip's clock with each round.
    if (getenv("QBENCH")) {
        const int M = 3 * Dm, K = Dm;
        std::vector<char> r((size_t)M * K);
        for (auto &c : r) c = (char)(rand() % 31 - 15);
        std::vector<float> sc((size_t)M * K / QGS, 0.001f);
        W["bench8"] = {"q8", M, K, q8img(M, K, r.data())};
        W["bench8.s"] = {"qs", M, K / QGS, buf(sc.size() * 4, sc.data())};
        for (int i = 0; i < 8; i++) {
            const char *nm = i % 2 ? "bench8" : "b0.qkv";
            clFinish(q); auto a = std::chrono::steady_clock::now();
            for (int rep = 0; rep < 4; rep++) gemmq(0, "bench", nm, Ni, M, K, QKV, nullptr);
            clFinish(q);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count() / 4;
            char clk[32] = "?";
            if (FILE *cf = fopen("/sys/class/kgsl/kgsl-3d0/gpuclk", "r")) { if (!fgets(clk, sizeof clk, cf)) clk[0] = 0; fclose(cf); }
            const double tops = 2.0 * Ni * M * K / ms / 1e9;
            printf("%s: %.1f ms, %.2f TOPS, clock %s", i % 2 ? "8-bit" : "4-bit", ms, tops, clk);
            if (!strchr(clk, '\n')) printf("\n");
        }
        return 0;
    }

    // the modulations for noise level s: [s1, tanh g1, s2, tanh g2] into m; the output norm's scale into fin
    auto modsFor = [&](float s, cl_mem m) {
        std::vector<float> te(256);
        for (int i = 0; i < 128; i++) {
            const float a = 1000.0f * s * expf(-logf(10000.0f) * i / 128.0f);
            te[i] = cosf(a); te[128 + i] = sinf(a);
        }
        CK(clEnqueueWriteBuffer(q, bTemb, CL_TRUE, 0, 256 * 4, te.data(), 0, 0, 0));
        gemv("gemv", w("t1"), bTemb, nullptr, bT1, Dm, 256, 0);
        gemv("gemv", w("t2"), bT1, nullptr, bTe, Dm, Dm, 1);
        gemv("gemv", w("mod"), bTe, nullptr, bMod, 4 * Dm, Dm, 1);
        gemv("gemv", w("out_scale"), bTe, nullptr, bSc, Dm, Dm, 1);
        std::vector<float> v(4 * Dm);
        CK(clEnqueueReadBuffer(q, bMod, CL_TRUE, 0, v.size() * 4, v.data(), 0, 0, 0));
        for (int k = 0; k < Dm; k++) { v[Dm + k] = tanhf(v[Dm + k]); v[3 * Dm + k] = tanhf(v[3 * Dm + k]); }
        CK(clEnqueueWriteBuffer(q, m, CL_TRUE, 0, v.size() * 4, v.data(), 0, 0, 0));
        CK(clEnqueueCopyBuffer(q, bSc, fin, 0, 0, Dm * 4, 0, 0, 0));
    };
    auto check = [&](const char *what, const std::vector<float> &got, const char *file) {
        printf("  %-14s error %.5f\n", what, relerr(got, rdf(R + "/" + file + ".f32")));
    };
    auto readRows = [&](cl_mem im, int n, int C) {   // X-layout image -> rows (n x C)
        std::vector<h16> y((size_t)n * C);
        size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)n, (size_t)C / 8, 1};
        CK(clEnqueueReadImage(q, im, CL_TRUE, org, reg, 0, 0, y.data(), 0, 0, 0));
        std::vector<float> out((size_t)n * C);
        for (int k8 = 0; k8 < C / 8; k8++) for (int r = 0; r < n; r++) for (int j = 0; j < 8; j++)
            out[(size_t)r * C + k8 * 8 + j] = y[((size_t)k8 * n + r) * 8 + j];
        return out;
    };

    // ---- the prompt pass, once per picture ----
    {
        auto a = std::chrono::steady_clock::now();
        auto ctx = rdf(R + "/ctx.f32");                       // T x 4096 reader features (rows from nc on unused)
        std::vector<float> tn(Dm);
        CK(clEnqueueReadBuffer(q, w("txt_norm"), CL_TRUE, 0, Dm * 4, tn.data(), 0, 0, 0));
        for (int r = 0; r < T; r++) {                          // rms norm with weight, on the host
            double ss = 0;
            for (int k = 0; k < Dm; k++) ss += (double)ctx[(size_t)r * Dm + k] * ctx[(size_t)r * Dm + k];
            const float rs = 1.0f / sqrtf((float)(ss / Dm) + 1e-6f);
            for (int k = 0; k < Dm; k++) ctx[(size_t)r * Dm + k] *= rs * tn[k];
        }
        cl_mem Xc = toX(ctx, T, Dm), C1 = img(T, Dm / 8);
        linear(1, "txt_in1", Xc, "txt_in1", T, Dm, Dm, C1);
        linear(3, "txt_in2", C1, "txt_in2", T, Dm, Dm, resT);
        modsFor(0.0f, mods0);
        for (int b = 0; b < NB; b++) {
            const std::string p = "b" + std::to_string(b) + ".";
            streamer.use(b, NB);
            normLinear(0, "qkv", resT, bZero, s10, p + "qkv", T, 3 * Dm, Dm, QKV);
            attnPrep(QKV, T, 0, T, 0, Dm / 8, 2 * Dm / 8, w(p + "qn"), w(p + "kn"), bPt, bQt);
            size_t z[3] = {0, 0, 0}, rk[3] = {T / 4, NHEADS * 64, 1}, rv[3] = {NHEADS * 32, T / 2, 1};
            CK(clEnqueueCopyImage(q, Ko, cK[b], z, z, rk, 0, 0, 0));
            CK(clEnqueueCopyImage(q, Vo, cV[b], z, z, rv, 0, 0, 0));
            attnCore(T, O, nc, false, 0, 1);
            linear(2, "o", O, p + "o", T, Dm, Dm, resT, g10);
            normLinear(4, "mlp_in", resT, bZero, s20, p + "mlp_in", T, 2 * F, Dm, Hh);
            linear(2, "down", Hh, p + "down", T, Dm, F, resT, g20);
        }
        clFinish(q);
        clReleaseMemObject(Xc); clReleaseMemObject(C1);
        printf("prompt pass %.0f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
    }

    // ---- one step: velocity for picture latent rows x (Ni x 64) at noise level s ----
    std::vector<float> xin((size_t)Ni * 64);
    cl_mem Ximg = nullptr;
    auto step = [&](const std::vector<float> &x, float s) {
        modsFor(s, mods);
        if (Ximg) clReleaseMemObject(Ximg);
        Ximg = toX(x, Ni, 64);
        linear(3, "img_in", Ximg, "img_in", Ni, Dm, 64, res);
        for (int b = 0; b < NB; b++) {
            const std::string p = "b" + std::to_string(b) + ".";
            streamer.use(b, NB);
            normLinear(0, "qkv", res, bZero, s1, p + "qkv", Ni, 3 * Dm, Dm, QKV);
            attnPrep(QKV, Ni, 0, Ni, 0, Dm / 8, 2 * Dm / 8, w(p + "qn"), w(p + "kn"), bPi, bQi);
            size_t z[3] = {0, 0, 0}, ok[3] = {(size_t)Ni / 4, 0, 0}, ov[3] = {0, (size_t)Ni / 2, 0};
            size_t rk[3] = {T / 4, NHEADS * 64, 1}, rv[3] = {NHEADS * 32, T / 2, 1};
            CK(clEnqueueCopyImage(q, cK[b], Ko, z, ok, rk, 0, 0, 0));
            CK(clEnqueueCopyImage(q, cV[b], Vo, z, ov, rv, 0, 0, 0));
            attnCore(Ni, O, Ni + nc, false, LK, 0);
            linear(2, "o", O, p + "o", Ni, Dm, Dm, res, g1);
            normLinear(4, "mlp_in", res, bZero, s2, p + "mlp_in", Ni, 2 * F, Dm, Hh);
            linear(2, "down", Hh, p + "down", Ni, Dm, F, res, g2);
        }
        lnmod(res, bZero, fin, Xa, Ni, Dm);
        gemm(0, "gemm out", Xa, w("proj_out"), Ni, 64, Dm, Yout);
        return readRows(Yout, Ni, 64);
    };

    auto x = rdf(R + "/noise.f32"), sg = rdf(R + "/sigmas.f32");
    sg.push_back(0.0f);
    if (getenv("DRAW")) {
        double total = 0;
        for (size_t i = 0; i + 1 < sg.size(); i++) {
            clFinish(q); auto a = std::chrono::steady_clock::now();
            auto v = step(x, sg[i]);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
            total += ms;
            for (size_t k = 0; k < x.size(); k++) x[k] += (sg[i + 1] - sg[i]) * v[k];
            printf("draw step %zu: sigma %.3f -> %.3f, %.1f ms\n", i, sg[i], sg[i + 1], ms); fflush(stdout);
        }
        FILE *f = fopen((R + "/draw.f32").c_str(), "wb");
        fwrite(x.data(), 4, x.size(), f); fclose(f);
        printf("drawing: %.1f s of steps, latent in %s/draw.f32\n", total / 1000, R.c_str());
        return 0;
    }
    if (FILE *vf = fopen((R + "/vel.f32").c_str(), "rb")) { fclose(vf); check("first step", step(x, sg[0]), "vel"); }
    else step(x, sg[0]);   // warm-up only
    tsum.clear();
    for (int l = 0; l < loops; l++) {
        clFinish(q); auto a = std::chrono::steady_clock::now();
        step(x, sg[0]);
        clFinish(q);
        printf("step %d: %.1f ms\n", l, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
    }
    if (prof) {
        double tot = 0; for (auto &kv : tsum) tot += kv.second;
        for (auto &kv : tsum) printf("  %-12s %8.1f ms  %4.1f%%\n", kv.first.c_str(), kv.second / loops, 100 * kv.second / tot);
    }
    return 0;
}

}  // namespace qwen
