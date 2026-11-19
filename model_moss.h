// MOSS-Image (+ the 8-step add-on, merged by tools/pack_moss.py) wired from the shared building blocks.
// Width 3072, 24 heads, 42 blocks, MLP 8192 with MOSS's paired activation (SITU); RMS norms (RMSN).
// The prompt goes through its own 2 blocks once per picture (with biases, no modulation). Every step then runs ONE
// stream of [picture rows; prompt rows] through all 42 blocks - both groups share the weights and attend to each
// other; padding rows after the prompt are masked as keys. One modulation for all blocks per step; each block's own
// norm weights are folded into it on the host. Only the picture rows go through the output layer.
// Test runner: tools/ref_moss.py's folder (noise, prompt rows, levels, first direction, final latent); DRAW=1 for the
// whole picture (Euler on the 8 levels).
#pragma once
#include "ops.h"

namespace moss {
using namespace ops;
static const int NB = 42, Dm = 3072, NHEADS = 24, F = 8192, CTXD = 4096, VIN = 256;   // latent 128, padded to 256

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

static int run(const std::string &R, int loops) {
    int nc = 0, PW = 0, PH = 0;
    if (FILE *f = fopen((R + "/info.txt").c_str(), "r")) { if (fscanf(f, "%d %d %d", &nc, &PW, &PH) != 3) nc = 0; fclose(f); }
    if (!nc) { fprintf(stderr, "%s/info.txt missing or bad\n", R.c_str()); return 1; }
    const int gh = PH / 16, gw = PW / 16, Ni = gh * gw, Tp = (nc + 7) / 8 * 8, N = Ni + Tp;
    if (Ni % 8) { fprintf(stderr, "picture rows must be a multiple of 8\n"); return 1; }
    if (!getenv("QGS")) QGS = 32;
    BIASES = 1; RMSN = 1; SITU = 1;
    ops::init(NHEADS, Dm, N, F);
    biasInit();
    printf("%dx%d: %d picture rows + prompt %d (padded %d), attention %d heads at a time\n", PW, PH, Ni, nc, Tp, HC);

    // ---- rotations: picture rows (0, nc, y - (h-1)/2, x - (w-1)/2), prompt rows (0, i, 0, 0) ----
    std::vector<float> ids((size_t)N * 4, 0.0f), idt((size_t)Tp * 4, 0.0f), P, Q, Pt, Qt;
    for (int y = 0; y < gh; y++) for (int x = 0; x < gw; x++) {
        float *p = &ids[((size_t)y * gw + x) * 4];
        p[1] = (float)nc; p[2] = y - (gh - 1) / 2.0f; p[3] = x - (gw - 1) / 2.0f;
    }
    for (int i = 0; i < Tp; i++) { ids[((size_t)Ni + i) * 4 + 1] = (float)i; idt[(size_t)i * 4 + 1] = (float)i; }
    rope(ids, N, P, Q); rope(idt, Tp, Pt, Qt);
    cl_mem bP = buf(P.size() * 4, P.data()), bQ = buf(Q.size() * 4, Q.data());
    cl_mem bPt = buf(Pt.size() * 4, Pt.data()), bQt = buf(Qt.size() * 4, Qt.data());

    // ---- buffers ----
    cl_mem res = buf((size_t)N * Dm * 4), resT = buf((size_t)Tp * Dm * 4);
    cl_mem QKV = img(N, 3 * Dm / 8), O = img(N, Dm / 8), Hh = img(N, F / 8), Yout = img(N, 128 / 8);
    cl_mem bTemb = buf(256 * 4), bT1 = buf(Dm * 4), bE = buf(Dm * 4), bMod = buf(6 * Dm * 4), bHm = buf(2 * Dm * 4);
    cl_mem mAll = buf((size_t)NB * 6 * Dm * 4), mHead = buf(2 * Dm * 4);
    std::vector<float> zeros(Dm, 0.0f), ones(Dm, 1.0f);
    cl_mem bZero = buf(Dm * 4, zeros.data()), bOnes = buf(Dm * 4, ones.data());
    auto part = [&](cl_mem m, size_t i) { return sub(m, i * Dm * 4, Dm * 4); };
    std::vector<std::array<cl_mem, 6>> mods(NB);
    for (int i = 0; i < NB; i++) for (int j = 0; j < 6; j++) mods[i][j] = part(mAll, (size_t)i * 6 + j);
    cl_mem hSh = part(mHead, 0), hSc = part(mHead, 1);
    auto rdv = [&](const std::string &n) {
        std::vector<float> v(Dm);
        CK(clEnqueueReadBuffer(q, w(n), CL_TRUE, 0, Dm * 4, v.data(), 0, 0, 0));
        return v;
    };
    std::vector<std::vector<float>> n1(NB), n2(NB);
    for (int b = 0; b < NB; b++) { n1[b] = rdv("b" + std::to_string(b) + ".n1"); n2[b] = rdv("b" + std::to_string(b) + ".n2"); }
    const auto fnorm = rdv("fnorm");

    // noise level s -> every block's [shift1, scale1', gate1, shift2, scale2', gate2] (scale' = n * (1 + scale) - 1,
    // the engine's (1 + sc) form with the block's norm weight folded in) and the head's shift / scale'
    auto modsFor = [&](float s) {
        std::vector<float> te(256);
        for (int i = 0; i < 128; i++) {
            const float a = 1000.0f * s * expf(-logf(10000.0f) * i / 128.0f);
            te[i] = cosf(a); te[128 + i] = sinf(a);
        }
        CK(clEnqueueWriteBuffer(q, bTemb, CL_TRUE, 0, 256 * 4, te.data(), 0, 0, 0));
        gemv("gemv", w("t1"), bTemb, w("t1.b"), bT1, Dm, 256, 0);
        gemv("gemv", w("t2"), bT1, w("t2.b"), bE, Dm, Dm, 1);
        gemv("gemv", w("mod"), bE, w("mod.b"), bMod, 6 * Dm, Dm, 0);
        gemv("gemv", w("hmod"), bE, w("hmod.b"), bHm, 2 * Dm, Dm, 0);
        std::vector<float> m(6 * Dm), hm(2 * Dm), all((size_t)NB * 6 * Dm);
        CK(clEnqueueReadBuffer(q, bMod, CL_FALSE, 0, m.size() * 4, m.data(), 0, 0, 0));
        CK(clEnqueueReadBuffer(q, bHm, CL_TRUE, 0, hm.size() * 4, hm.data(), 0, 0, 0));
        for (int b = 0; b < NB; b++) {
            float *a = &all[(size_t)b * 6 * Dm];
            for (int k = 0; k < Dm; k++) {
                a[k] = m[k]; a[2 * Dm + k] = m[2 * Dm + k]; a[3 * Dm + k] = m[3 * Dm + k]; a[5 * Dm + k] = m[5 * Dm + k];
                a[Dm + k] = n1[b][k] * (1.0f + m[Dm + k]) - 1.0f;
                a[4 * Dm + k] = n2[b][k] * (1.0f + m[4 * Dm + k]) - 1.0f;
            }
        }
        for (int k = 0; k < Dm; k++) hm[Dm + k] = fnorm[k] * (1.0f + hm[Dm + k]) - 1.0f;
        CK(clEnqueueWriteBuffer(q, mAll, CL_FALSE, 0, all.size() * 4, all.data(), 0, 0, 0));
        CK(clEnqueueWriteBuffer(q, mHead, CL_TRUE, 0, hm.size() * 4, hm.data(), 0, 0, 0));
    };

    // ---- the prompt's own 2 blocks, once per picture (norm weight only: sc = n - 1, gates 1) ----
    {
        auto a = std::chrono::steady_clock::now();
        auto ctx = rdf(R + "/ctx.f32");
        ctx.resize((size_t)Tp * CTXD, 0.0f);
        cl_mem Xc = toX(ctx, Tp, CTXD);
        linear(3, "tproj", Xc, "tproj", Tp, Dm, CTXD, resT);
        for (int j = 0; j < 2; j++) {
            const std::string p = "tb" + std::to_string(j) + ".";
            auto a1 = rdv(p + "n1"), a2 = rdv(p + "n2");
            for (auto &v : a1) v -= 1.0f;
            for (auto &v : a2) v -= 1.0f;
            cl_mem s1 = buf(Dm * 4, a1.data()), s2 = buf(Dm * 4, a2.data());
            normLinear(0, "qkv", resT, bZero, s1, p + "qkv", Tp, 3 * Dm, Dm, QKV);
            attnPrep(QKV, Tp, 0, Tp, 0, Dm / 8, 2 * Dm / 8, w(p + "qn"), w(p + "kn"), bPt, bQt);
            attnCore(Tp, O, nc);
            linear(2, "o", O, p + "o", Tp, Dm, Dm, resT, bOnes);
            normLinear(4, "mlp_in", resT, bZero, s2, p + "mlp_in", Tp, 2 * F, Dm, Hh);
            linear(2, "down", Hh, p + "down", Tp, Dm, F, resT, bOnes);
            clFinish(q); clReleaseMemObject(s1); clReleaseMemObject(s2);
        }
        clReleaseMemObject(Xc);
        printf("prompt blocks %.0f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
    }

    auto readRows = [&](cl_mem im, int n, int C) {   // X-layout image -> rows (n x C)
        std::vector<h16> y((size_t)n * C);
        size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)n, (size_t)C / 8, 1};
        CK(clEnqueueReadImage(q, im, CL_TRUE, org, reg, 0, 0, y.data(), 0, 0, 0));
        std::vector<float> out((size_t)n * C);
        for (int k8 = 0; k8 < C / 8; k8++) for (int r = 0; r < n; r++) for (int j = 0; j < 8; j++)
            out[(size_t)r * C + k8 * 8 + j] = y[((size_t)k8 * n + r) * 8 + j];
        return out;
    };

    // ---- one step: direction for the picture latent rows x (Ni x 128) at noise level s ----
    std::vector<float> xin((size_t)N * VIN, 0.0f);
    cl_mem Xin = nullptr;
    auto step = [&](const std::vector<float> &x, float s) {
        modsFor(s);
        for (int r = 0; r < Ni; r++) std::copy(x.begin() + (size_t)r * 128, x.begin() + (size_t)(r + 1) * 128, xin.begin() + (size_t)r * VIN);
        if (Xin) clReleaseMemObject(Xin);
        Xin = toX(xin, N, VIN);
        linear(3, "vin", Xin, "vin", N, Dm, VIN, res, nullptr, 0, Ni);
        {   // the prompt rows, fresh from the prompt blocks every step: resT (c4 * Tp + t) -> res (c4 * N + Ni + t)
            size_t so[3] = {0, 0, 0}, dor[3] = {(size_t)Ni * 16, 0, 0}, reg[3] = {(size_t)Tp * 16, (size_t)Dm / 4, 1};
            CK(clEnqueueCopyBufferRect(q, resT, res, so, dor, reg, (size_t)Tp * 16, 0, (size_t)N * 16, 0, 0, 0, 0));
        }
        for (int b = 0; b < NB; b++) {
            const std::string p = "b" + std::to_string(b) + ".";
            streamer.use(b, NB);
            auto &m = mods[b];
            normLinear(0, "qkv", res, m[0], m[1], p + "qkv", N, 3 * Dm, Dm, QKV);
            attnPrep(QKV, N, 0, N, 0, Dm / 8, 2 * Dm / 8, w(p + "qn"), w(p + "kn"), bP, bQ);
            attnCore(N, O, Ni + nc);
            linear(2, "o", O, p + "o", N, Dm, Dm, res, m[2]);
            normLinear(4, "mlp_in", res, m[3], m[4], p + "mlp_in", N, 2 * F, Dm, Hh);
            linear(2, "down", Hh, p + "down", N, Dm, F, res, m[5]);
        }
        normLinear(0, "fout", res, hSh, hSc, "fout", N, 128, Dm, Yout, nullptr, 0, Ni);
        return readRows(Yout, Ni, 128);
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
    if (FILE *vf = fopen((R + "/out.f32").c_str(), "rb")) {
        fclose(vf);
        printf("  first step     error %.5f\n", relerr(step(x, sg[0]), rdf(R + "/out.f32")));
    } else step(x, sg[0]);
    tsum.clear();
    for (int l = 0; l < loops; l++) {
        clFinish(q); auto a = std::chrono::steady_clock::now(); streamer.waitMs = 0;
        step(x, sg[0]);
        clFinish(q);
        printf("step %d: %.1f ms (waiting for weights %.1f ms)\n", l, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count(), streamer.waitMs);
    }
    if (prof) {
        double tot = 0; for (auto &kv : tsum) tot += kv.second;
        for (auto &kv : tsum) printf("  %-12s %8.1f ms  %4.1f%%\n", kv.first.c_str(), kv.second / loops, 100 * kv.second / tot);
    }
    return 0;
}

}  // namespace moss
