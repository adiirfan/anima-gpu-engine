// Fibo 2.0 edit-turbo wired from the shared building blocks (2026-11-18). Width 3584, 28 heads, 2 refiner + 44 main
// blocks (b0..b45, see tools/pack_fibo.py), SwiGLU 9728, RMS norms before AND after each layer (rmsAdd), Perceiver.
// Rows every step: [picture Ni | edit picture Ni | gist G]. Only picture rows carry the step's modulation; edit-picture
// and gist rows use the t = 0 one, folded into their norm weights by the packer. The refiner sees picture + edit picture
// only (gist keys masked; gist rows are overwritten afterwards). Injection blocks (b8, b16, b24, b32, b40) also attend
// to their prompt bundle. Once per edit: the Perceiver (512 queries -> G gist rows from bundle 0) and the injection
// blocks' prompt keys / values. Meant to run with STREAM=1 (~9.4 GB at 8-bit).
// Test runner: tools/ref_fibo.py's folder; DRAW=1 for the whole edit (Euler on the edit-turbo's 4 levels).
#pragma once
#include "ops.h"

namespace fibo {
using namespace ops;
static const int NB = 46, Dm = 3584, NHEADS = 28, F = 9728, CAP = 7680, NBUN = 6, MAXG = 512, VIN = 256;
static const int INJB[5] = {8, 16, 24, 32, 40};

// rotations in the engine's form (P = [cos, cos], Q = [-sin, sin]); axes 32 / 48 / 48, theta 256
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

static int run(const std::string &R, int loops) {
    int T = 0, G = 0, h = 0, wd = 0;
    if (FILE *f = fopen((R + "/info.txt").c_str(), "r")) { if (fscanf(f, "%d %d %d %d", &T, &G, &h, &wd) != 4) T = 0; fclose(f); }
    if (!T) { fprintf(stderr, "%s/info.txt missing or bad\n", R.c_str()); return 1; }
    const int Ni = h * wd, NA = 2 * Ni + G, Tp = (T + 7) / 8 * 8;
    if (Ni % 8 || G % 8) { fprintf(stderr, "row counts must be multiples of 8\n"); return 1; }
    if (!getenv("QGS")) QGS = 32;
    BIASES = 1; RMSN = 1;
    ops::init(NHEADS, Dm, std::max(NA, Tp), F);
    biasInit();
    printf("%dx%d edit: %d picture + %d edit-picture + %d gist rows, prompt %d (padded %d), attention %d heads at a time\n",
           wd * 16, h * 16, Ni, Ni, G, T, Tp, HC);

    // ---- rotations: picture (MAXG + 1, y, x), edit picture (MAXG + 2, y, x), gist (1 .. G, 0, 0) ----
    std::vector<float> ids((size_t)NA * 3, 0.0f), P, Q;
    for (int y = 0; y < h; y++) for (int x = 0; x < wd; x++) {
        const size_t r = (size_t)y * wd + x;
        ids[r * 3] = MAXG + 1; ids[r * 3 + 1] = (float)y; ids[r * 3 + 2] = (float)x;
        ids[(Ni + r) * 3] = MAXG + 2; ids[(Ni + r) * 3 + 1] = (float)y; ids[(Ni + r) * 3 + 2] = (float)x;
    }
    for (int i = 0; i < G; i++) ids[((size_t)2 * Ni + i) * 3] = (float)(i + 1);
    rope(ids, NA, P, Q);
    cl_mem bP = buf(P.size() * 4, P.data()), bQ = buf(Q.size() * 4, Q.data());

    // ---- buffers ----
    cl_mem res = buf((size_t)NA * Dm * 4), resG = buf((size_t)G * Dm * 4);
    cl_mem QKV = img(NA, 3 * Dm / 8), O = img(NA, Dm / 8), A = img(NA, Dm / 8), Hh = img(NA, F / 8), Yout = img(NA, 128 / 8);
    cl_mem KVb = img(std::max(Tp, G), 2 * Dm / 8);
    cl_mem bTs = buf(256 * 4), bT1 = buf(1024 * 4), bTe = buf(256 * 4), bM = buf(4 * Dm * 4), bOs = buf(Dm * 4);
    std::vector<float> zeros(Dm, 0.0f), ones(Dm, 1.0f);
    cl_mem bZero = buf(Dm * 4, zeros.data()), bOnes = buf(Dm * 4, ones.data());
    // per block: [sc1 picture, w_o picture, sc2 picture, w_f picture, sc1 zero, w_o zero, sc2 zero, w_f zero]
    cl_mem mAll = buf((size_t)NB * 8 * Dm * 4);
    std::vector<std::array<cl_mem, 8>> mv(NB);
    for (int b = 0; b < NB; b++) for (int j = 0; j < 8; j++) mv[b][j] = sub(mAll, ((size_t)b * 8 + j) * Dm * 4, Dm * 4);
    auto rdv = [&](const std::string &n, int cnt = Dm) {
        std::vector<float> v(cnt);
        CK(clEnqueueReadBuffer(q, w(n), CL_TRUE, 0, (size_t)cnt * 4, v.data(), 0, 0, 0));
        return v;
    };
    struct BN { std::vector<float> n1, n2, f1, f2, z1, z2, zf1, zf2; };
    std::vector<BN> bn(NB);
    for (int b = 0; b < NB; b++) {
        const std::string p = "b" + std::to_string(b) + ".";
        bn[b] = {rdv(p + "n1"), rdv(p + "n2"), rdv(p + "f1"), rdv(p + "f2"), rdv(p + "z1"), rdv(p + "z2"), rdv(p + "zf1"), rdv(p + "zf2")};
    }
    auto minus1 = [&](const std::string &n) { auto v = rdv(n); for (auto &x : v) x -= 1.0f; return buf(Dm * 4, v.data()); };

    // noise level s -> every block's eight vectors (above) and the output layer's scale (bOs)
    auto modsFor = [&](float s) {
        std::vector<float> te(256);
        for (int i = 0; i < 128; i++) {
            const float a = 1000.0f * s * expf(-logf(10000.0f) * i / 128.0f);
            te[i] = cosf(a); te[128 + i] = sinf(a);
        }
        CK(clEnqueueWriteBuffer(q, bTs, CL_TRUE, 0, 256 * 4, te.data(), 0, 0, 0));
        gemv("gemv", w("t1"), bTs, w("t1.b"), bT1, 1024, 256, 0);
        gemv("gemv", w("t2"), bT1, w("t2.b"), bTe, 256, 1024, 1);
        gemv("gemv", w("out_scale"), bTe, w("out_scale.b"), bOs, Dm, 256, 1);
        std::vector<float> all((size_t)NB * 8 * Dm), m(4 * Dm);
        for (int b = 0; b < NB; b++) {
            const std::string p = "b" + std::to_string(b) + ".";
            gemv("gemv", w(p + "mod"), bTe, w(p + "mod.b"), bM, 4 * Dm, 256, 0);
            CK(clEnqueueReadBuffer(q, bM, CL_TRUE, 0, m.size() * 4, m.data(), 0, 0, 0));
            float *a = &all[(size_t)b * 8 * Dm];
            const BN &n = bn[b];
            for (int k = 0; k < Dm; k++) {   // mods: s1, g1, s2, g2 (gates through tanh)
                a[k] = n.n1[k] * (1.0f + m[k]) - 1.0f;
                a[Dm + k] = n.n2[k] * tanhf(m[Dm + k]);
                a[2 * Dm + k] = n.f1[k] * (1.0f + m[2 * Dm + k]) - 1.0f;
                a[3 * Dm + k] = n.f2[k] * tanhf(m[3 * Dm + k]);
                a[4 * Dm + k] = n.z1[k] - 1.0f; a[5 * Dm + k] = n.z2[k];
                a[6 * Dm + k] = n.zf1[k] - 1.0f; a[7 * Dm + k] = n.zf2[k];
            }
        }
        CK(clEnqueueWriteBuffer(q, mAll, CL_TRUE, 0, all.size() * 4, all.data(), 0, 0, 0));
    };

    // ---- once per edit: the prompt bundles, the Perceiver's gist rows, the injection blocks' keys / values ----
    std::vector<CrossKV> ckv(5);
    {
        auto a0 = std::chrono::steady_clock::now();
        auto ctx = rdf(R + "/ctx.f32");                       // T x (6 x 7680)
        std::vector<cl_mem> Xb(NBUN);
        for (int k = 0; k < NBUN; k++) {
            std::vector<float> bv((size_t)Tp * CAP, 0.0f);
            for (int t = 0; t < T; t++) std::copy(ctx.begin() + (size_t)t * NBUN * CAP + (size_t)k * CAP,
                                                  ctx.begin() + (size_t)t * NBUN * CAP + (size_t)(k + 1) * CAP, bv.begin() + (size_t)t * CAP);
            Xb[k] = toX(bv, Tp, CAP);
        }
        // Perceiver: x = queries[:G]; 2 layers of (cross to bundle 0, self, SwiGLU), each pre-normed, no gates
        {
            auto qv = rdv("queries", MAXG * Dm);
            std::vector<float> lay((size_t)G * Dm);   // engine layout: channel group c4, then row
            for (int c4 = 0; c4 < Dm / 4; c4++) for (int g = 0; g < G; g++) for (int j = 0; j < 4; j++)
                lay[((size_t)c4 * G + g) * 4 + j] = qv[(size_t)g * Dm + c4 * 4 + j];
            CK(clEnqueueWriteBuffer(q, resG, CL_TRUE, 0, lay.size() * 4, lay.data(), 0, 0, 0));
        }
        for (int j = 0; j < 2; j++) {
            const std::string p = "p" + std::to_string(j) + ".";
            cl_mem cn = minus1(p + "cn"), sn = minus1(p + "sn"), fn = minus1(p + "fn");
            linear(0, "ckv", Xb[0], p + "ckv", Tp, 2 * Dm, CAP, KVb);
            CrossKV c = crossKV(KVb, Tp, 0, Dm / 8, w(p + "ckn")); c.valid = T;
            normLinear(0, "cq", resG, bZero, cn, p + "cq", G, Dm, Dm, QKV);
            crossAttn(QKV, G, 0, w(p + "cqn"), c, 0.0f, O);
            linear(2, "co", O, p + "co", G, Dm, Dm, resG, bOnes);
            normLinear(0, "skv", resG, bZero, sn, p + "skv", G, 2 * Dm, Dm, KVb);
            CrossKV s = crossKV(KVb, G, 0, Dm / 8, w(p + "skn"));
            normLinear(0, "sq", resG, bZero, sn, p + "sq", G, Dm, Dm, QKV);
            crossAttn(QKV, G, 0, w(p + "sqn"), s, 0.0f, O);
            linear(2, "so", O, p + "so", G, Dm, Dm, resG, bOnes);
            normLinear(4, "mlp_in", resG, bZero, fn, p + "mlp_in", G, 2 * F, Dm, Hh);
            linear(2, "down", Hh, p + "down", G, Dm, F, resG, bOnes);
            clFinish(q);
            for (cl_mem m : {cn, sn, fn, c.K, c.V, s.K, s.V}) clReleaseMemObject(m);
        }
        {   // gist = rms(x) * pnorm, on the host (once)
            std::vector<float> lay((size_t)G * Dm);
            CK(clEnqueueReadBuffer(q, resG, CL_TRUE, 0, lay.size() * 4, lay.data(), 0, 0, 0));
            auto pn = rdv("pnorm");
            for (int g = 0; g < G; g++) {
                double ss = 0;
                for (int c4 = 0; c4 < Dm / 4; c4++) for (int j = 0; j < 4; j++) { const double v = lay[((size_t)c4 * G + g) * 4 + j]; ss += v * v; }
                const float r = 1.0f / sqrtf((float)(ss / Dm) + 1e-6f);
                for (int c4 = 0; c4 < Dm / 4; c4++) for (int j = 0; j < 4; j++) lay[((size_t)c4 * G + g) * 4 + j] *= r * pn[c4 * 4 + j];
            }
            CK(clEnqueueWriteBuffer(q, resG, CL_TRUE, 0, lay.size() * 4, lay.data(), 0, 0, 0));
        }
        for (int i = 0; i < 5; i++) {
            const int b = INJB[i];
            streamer.use(b, NB);
            linear(0, "ckv", Xb[1 + i], "b" + std::to_string(b) + ".ckv", Tp, 2 * Dm, CAP, KVb);
            ckv[i] = crossKV(KVb, Tp, 0, Dm / 8, w("b" + std::to_string(b) + ".ckn"));
            ckv[i].valid = T;
        }
        clFinish(q);
        for (cl_mem m : Xb) clReleaseMemObject(m);
        printf("prompt, gist and injection keys %.0f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a0).count());
    }
    std::vector<cl_mem> cn1(5);
    for (int i = 0; i < 5; i++) cn1[i] = minus1("b" + std::to_string(INJB[i]) + ".cn1");
    auto gistIn = [&]() {   // resG (c4 * G + g) -> res rows 2 Ni + g
        size_t so[3] = {0, 0, 0}, dor[3] = {(size_t)2 * Ni * 16, 0, 0}, reg[3] = {(size_t)G * 16, (size_t)Dm / 4, 1};
        CK(clEnqueueCopyBufferRect(q, resG, res, so, dor, reg, (size_t)G * 16, 0, (size_t)NA * 16, 0, 0, 0, 0));
    };
    auto readRows = [&](cl_mem im, int n, int C) {
        std::vector<h16> y((size_t)n * C);
        size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)n, (size_t)C / 8, 1};
        CK(clEnqueueReadImage(q, im, CL_TRUE, org, reg, 0, 0, y.data(), 0, 0, 0));
        std::vector<float> out((size_t)n * C);
        for (int k8 = 0; k8 < C / 8; k8++) for (int r = 0; r < n; r++) for (int j = 0; j < 8; j++)
            out[(size_t)r * C + k8 * 8 + j] = y[((size_t)k8 * n + r) * 8 + j];
        return out;
    };

    // ---- one step ----
    auto refv = rdf(R + "/ref.f32");
    std::vector<float> xin((size_t)NA * VIN, 0.0f);
    for (int r = 0; r < Ni; r++) std::copy(refv.begin() + (size_t)r * 128, refv.begin() + (size_t)(r + 1) * 128, xin.begin() + (size_t)(Ni + r) * VIN);
    cl_mem Xin = nullptr;
    auto step = [&](const std::vector<float> &x, float s) {
        modsFor(s);
        for (int r = 0; r < Ni; r++) std::copy(x.begin() + (size_t)r * 128, x.begin() + (size_t)(r + 1) * 128, xin.begin() + (size_t)r * VIN);
        if (Xin) clReleaseMemObject(Xin);
        Xin = toX(xin, NA, VIN);
        linear(3, "x_in", Xin, "x_in", NA, Dm, VIN, res, nullptr, 0, 2 * Ni);
        gistIn();
        for (int b = 0; b < NB; b++) {
            const std::string p = "b" + std::to_string(b) + ".";
            streamer.use(b, NB);
            auto &m = mv[b];
            normLinear(0, "qkv", res, bZero, m[0], p + "qkv", NA, 3 * Dm, Dm, QKV, nullptr, 0, Ni);
            normLinear(0, "qkv", res, bZero, m[4], p + "qkv", NA, 3 * Dm, Dm, QKV, nullptr, Ni, NA);
            attnPrep(QKV, NA, 0, NA, 0, Dm / 8, 2 * Dm / 8, w(p + "qn"), w(p + "kn"), bP, bQ);
            attnCore(NA, O, b < 2 ? 2 * Ni : -1);
            linear(0, "o", O, p + "o", NA, Dm, Dm, A);
            rmsAdd(A, res, m[1], NA, Dm, 0, Ni);
            rmsAdd(A, res, m[5], NA, Dm, Ni, NA);
            for (int i = 0; i < 5; i++) if (INJB[i] == b) {
                normLinear(0, "cq", res, bZero, cn1[i], p + "cq", NA, Dm, Dm, QKV);
                crossAttn(QKV, NA, 0, w(p + "cqn"), ckv[i], 0.0f, O);
                linear(0, "co", O, p + "co", NA, Dm, Dm, A);
                rmsAdd(A, res, w(p + "cn2"), NA, Dm, 0, NA);
            }
            normLinear(4, "mlp_in", res, bZero, m[2], p + "mlp_in", NA, 2 * F, Dm, Hh, nullptr, 0, Ni);
            normLinear(4, "mlp_in", res, bZero, m[6], p + "mlp_in", NA, 2 * F, Dm, Hh, nullptr, Ni, NA);
            linear(0, "down", Hh, p + "down", NA, Dm, F, A);
            rmsAdd(A, res, m[3], NA, Dm, 0, Ni);
            rmsAdd(A, res, m[7], NA, Dm, Ni, NA);
            if (b == 1) gistIn();   // the refiner left the gist rows alone in meaning, not in memory
        }
        LNNEXT = 1;
        normLinear(0, "out", res, bZero, bOs, "out", NA, 128, Dm, Yout, nullptr, 0, Ni);
        LNNEXT = 0;
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

}  // namespace fibo
