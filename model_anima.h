// Anima (Cosmos-style DiT, 28 blocks, width 2048, 16 heads) wired from the shared building blocks. Each block:
// self-attention, cross-attention to the prompt, MLP; each preceded by layernorm + per-block modulation (adaLN with a
// shared low-rank part, "lora"), each added back through a gate. Test runner: one step on a reference folder and its
// error, timed loops, or DRAW=<steps> for a whole picture.
#pragma once
#include "ops.h"

namespace anima {
using namespace ops;
static const int NB = 28, CTXD = 1024, WIDTH = 2048, NHEADS = 16;

// Cosmos VideoRopePosition3DEmb for one picture (T = 1), Anima's extrapolation ratios 4 (height) and 4 (width):
// per token 22 time pairs (all at position 0), 21 row pairs, 21 column pairs; P = [cos | cos], Q = [-sin | sin]
// (checked against ComfyUI's tables for all sizes the app draws, to 2e-6).
static void ropeTables(int hp, int wp, std::vector<float> &P, std::vector<float> &Q) {
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

// a block's table names: "<i>." in the file, "b<i>." when streamed (STREAM=1: the loader renames them, rt.h)
static std::string blk(int i) { return (STREAM ? "b" : "") + std::to_string(i) + "."; }
static int run(const std::string &weights, const std::string &R, int loops, int PW, int PH) {
    const int N = (PW / 16) * (PH / 16), LK = 256, Dm = WIDTH;
    ops::init(NHEADS, WIDTH, N, 4 * WIDTH);
    if (getenv("CEN")) {
        std::vector<CenLayer> L;
        const char *kinds[6] = {"sa_qkv", "sa_o", "ca_q", "ca_o", "l1", "l2"};
        const int kK[6] = {1, 1, 1, 1, 1, 4}, kM[6] = {3, 1, 1, 1, 4, 1};
        for (int b = 0; b < NB; b++)
            for (int l = 0; l < 6; l++) L.push_back({blk(b) + kinds[l], kK[l] * Dm, kM[l] * Dm});
        cenLoad(getenv("CEN"), L, NB);
    }

    // ---- inputs ----
    auto exists = [&](const char *n) { FILE *f = fopen((R + "/" + n).c_str(), "rb"); if (f) fclose(f); return f != nullptr; };
    auto patches = rdf(R + "/patches.f32"), ctxv = rdf(R + "/ctx.f32");
    // emb_raw: the test step's time signal (DRAW makes its own); rope tables: the test folder's, else made here
    std::vector<float> embraw = exists("emb_raw.f32") ? rdf(R + "/emb_raw.f32") : std::vector<float>(WIDTH, 0.0f);
    std::vector<float> Pt, Qt;
    if (exists("rope_P.f32")) { Pt = rdf(R + "/rope_P.f32"); Qt = rdf(R + "/rope_Q.f32"); }
    else ropeTables(PH / 16, PW / 16, Pt, Qt);
    if (patches.size() != (size_t)N * 68 || ctxv.size() != (size_t)LK * CTXD) {
        fprintf(stderr, "patches.f32 must hold %d x 68 and ctx.f32 %d x %d numbers\n", N, LK, CTXD); return 1;
    }
    cl_mem bP = buf(Pt.size() * 4, Pt.data()), bQ = buf(Qt.size() * 4, Qt.data());
    cl_mem Xp = toX(patches, N, 68), Xc = toX(ctxv, LK, CTXD);

    // ---- activations ----
    cl_mem res = buf((size_t)N * Dm * 4);
    cl_mem QKV = img(N, 3 * Dm / 8), O = img(N, Dm / 8), H = img(N, 4 * Dm / 8), KVc = img(LK, 2 * Dm / 8);
    // per-step vectors: emb (normed), lora, silu hidden, and per block 3 mods of [sh | sc | g]
    cl_mem bEmbRaw = buf(Dm * 4), bEmb = buf(Dm * 4), bT1 = buf(Dm * 4), bLora = buf(3 * Dm * 4), bHid = buf(256 * 4);
    cl_mem bLora2 = sub(bLora, 0, 2 * Dm * 4);
    std::vector<cl_mem> mod(NB * 3), msh(NB * 3), msc(NB * 3), mg(NB * 3);
    for (int i = 0; i < NB * 3; i++) {
        mod[i] = buf(3 * Dm * 4);
        msh[i] = sub(mod[i], 0, Dm * 4); msc[i] = sub(mod[i], Dm * 4, Dm * 4); mg[i] = sub(mod[i], 2 * Dm * 4, Dm * 4);
    }
    cl_mem fmod = buf(2 * Dm * 4), fsh = sub(fmod, 0, Dm * 4), fsc = sub(fmod, Dm * 4, Dm * 4);
    cl_mem Yout = img(N, 8);
    const float cbias = logf(512.0f - LK + 1.0f);
    printf("%dx%d: %d tokens, attention %d heads at a time\n", PW, PH, N, HC);

    // cross-attention keys / values: once per picture
    std::vector<CrossKV> ckv(NB);
    for (int i = 0; i < NB; i++) {
        std::string b = blk(i);
        gemm(0, "gemm ctx", Xc, w(b + "ca_kv"), LK, 2 * Dm, CTXD, KVc);
        ckv[i] = crossKV(KVc, LK, 0, Dm / 8, w(b + "ca_kn"));
    }

    std::vector<float> tn(Dm);
    CK(clEnqueueReadBuffer(q, w("t_norm"), CL_TRUE, 0, Dm * 4, tn.data(), 0, 0, 0));
    float modSigma = -1.0f;   // the modulation depends only on the noise level, so it is kept until that changes
    float sigma = 0.6f;
    if (FILE *sf = fopen((R + "/sigma.f32").c_str(), "rb")) { if (fread(&sigma, 4, 1, sf) != 1) sigma = 0.6f; fclose(sf); }
    printf("noise level %.3f\n", sigma);

    auto step = [&](bool check) {
        if (modSigma != sigma) {
            double ss = 0; for (int c = 0; c < Dm; c++) ss += embraw[c] * (double)embraw[c];
            float rs = 1.0f / sqrtf(ss / Dm + 1e-6f);
            std::vector<float> e(Dm); for (int c = 0; c < Dm; c++) e[c] = embraw[c] * rs * tn[c];
            CK(clEnqueueWriteBuffer(q, bEmbRaw, CL_FALSE, 0, Dm * 4, embraw.data(), 0, 0, 0));
            CK(clEnqueueWriteBuffer(q, bEmb, CL_FALSE, 0, Dm * 4, e.data(), 0, 0, 0));
            gemv("gemv", w("t1"), bEmbRaw, nullptr, bT1, Dm, Dm, 0);
            gemv("gemv", w("t2"), bT1, nullptr, bLora, 3 * Dm, Dm, 1);
            const char *mn[3] = {"msa", "mca", "mmlp"};
            for (int i = 0; i < NB; i++) for (int j = 0; j < 3; j++) {
                std::string b = blk(i) + mn[j];
                gemv("gemv", w(b + "1"), bEmb, nullptr, bHid, 256, Dm, 1);
                gemv("gemv", w(b + "2"), bHid, bLora, mod[i * 3 + j], 3 * Dm, 256, 0);
            }
            gemv("gemv", w("fmod1"), bEmb, nullptr, bHid, 256, Dm, 1);
            gemv("gemv", w("fmod2"), bHid, bLora2, fmod, 2 * Dm, 256, 0);
            cenSet(sigma);
            modSigma = getenv("NOCACHE") ? -1.0f : sigma;
        }
        gemm(3, "gemm embed", Xp, w("x_embedder"), N, Dm, 72, res);
        for (int i = 0; i < NB; i++) {
            const std::string b = blk(i);
            if (STREAM) streamer.use(i, NB);
            normLinear(0, "qkv", res, msh[i * 3], msc[i * 3], b + "sa_qkv", N, 3 * Dm, Dm, QKV);
            selfAttn(QKV, N, 0, Dm / 8, 2 * Dm / 8, w(b + "sa_qn"), w(b + "sa_kn"), bP, bQ, O, PW / 16, i);
            linear(2, "out", O, b + "sa_o", N, Dm, Dm, res, mg[i * 3]);
            normLinear(0, "q", res, msh[i * 3 + 1], msc[i * 3 + 1], b + "ca_q", N, Dm, Dm, QKV);
            crossAttn(QKV, N, 0, w(b + "ca_qn"), ckv[i], cbias, O);
            linear(2, "out", O, b + "ca_o", N, Dm, Dm, res, mg[i * 3 + 1]);
            normLinear(1, "mlp1", res, msh[i * 3 + 2], msc[i * 3 + 2], b + "l1", N, 4 * Dm, Dm, H);
            linear(2, "mlp2", H, b + "l2", N, Dm, 4 * Dm, res, mg[i * 3 + 2]);
            if (check && (i == 0 || i == 1 || i == NB - 1)) {
                std::vector<float> raw((size_t)N * Dm), out((size_t)N * Dm);
                CK(clEnqueueReadBuffer(q, res, CL_TRUE, 0, raw.size() * 4, raw.data(), 0, 0, 0));
                for (int og = 0; og < Dm / 4; og++) for (int n = 0; n < N; n++) for (int j = 0; j < 4; j++)
                    out[(size_t)n * Dm + og * 4 + j] = raw[((size_t)og * N + n) * 4 + j];
                printf("  after block %2d: error %.5f\n", i, relerr(out, rdf(R + "/x_after" + std::to_string(i) + ".f32")));
            }
        }
        lnmod(res, fsh, fsc, Xa, N, Dm);
        gemm(0, "gemm final", Xa, w("final_linear"), N, 64, Dm, Yout);
    };
    auto readOut = [&]() {
        std::vector<h16> y((size_t)N * 64);
        size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)N, 8, 1};
        CK(clEnqueueReadImage(q, Yout, CL_TRUE, org, reg, 0, 0, y.data(), 0, 0, 0));
        return y;
    };

    // DRAW=<steps>: a whole picture from the test folder's noise (Euler on the shift-3 flow schedule); the finished
    // latent patches go to <ref dir>/draw.f32. DRAWCOOL=<milli-degrees>: wait for thermal_zone31 before each step.
    if (getenv("DRAW")) {
        // <ref>/sigmas.f32 (steps + 1 levels, the app's schedule) when present; DRAW=<steps> otherwise
        std::vector<float> sg = exists("sigmas.f32") ? rdf(R + "/sigmas.f32") : std::vector<float>();
        const int steps = sg.size() > 1 ? (int)sg.size() - 1 : atoi(getenv("DRAW"));
        const int cool = getenv("DRAWCOOL") ? atoi(getenv("DRAWCOOL")) : 0;
        auto sh = [](double t) { return 3 * t / (1 + 2 * t); };
        double total = 0;
        for (int si = 0; si < steps; si++) {
            if (cool) for (int k = 0; k < 600; k++) {
                FILE *tf = fopen("/sys/class/thermal/thermal_zone31/temp", "r"); int t = 0;
                if (tf) { if (fscanf(tf, "%d", &t) != 1) t = 0; fclose(tf); }
                if (t <= cool) break;
                usleep(500000);
            }
            const float s0 = sg.empty() ? (float)sh(1.0 - (double)si / steps) : sg[si],
                        s1 = sg.empty() ? (si + 1 < steps ? (float)sh(1.0 - (double)(si + 1) / steps) : 0.0f) : sg[si + 1];
            sigma = s0;
            for (int k = 0; k < Dm / 2; k++) {
                const float e = expf(-logf(10000.0f) * k / (Dm / 2)), a = s0 * e;
                embraw[k] = cosf(a); embraw[Dm / 2 + k] = sinf(a);
            }
            clReleaseMemObject(Xp); Xp = toX(patches, N, 68);
            clFinish(q); auto a = std::chrono::steady_clock::now();
            step(false);
            auto y = readOut();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
            total += ms;
            for (int n = 0; n < N; n++) for (int C = 0; C < 16; C++) for (int pp = 0; pp < 4; pp++) {
                const int k8 = (pp * 16 + C) / 8, j = (pp * 16 + C) % 8;
                patches[(size_t)n * 68 + C * 4 + pp] += (s1 - s0) * (float)y[((size_t)k8 * N + n) * 8 + j];
            }
            printf("draw step %d: sigma %.3f -> %.3f, %.1f ms\n", si, s0, s1, ms); fflush(stdout);
        }
        FILE *f = fopen((R + "/draw.f32").c_str(), "wb");
        fwrite(patches.data(), 4, patches.size(), f); fclose(f);
        printf("drawing: %.1f s of steps, latent in %s/draw.f32\n", total / 1000, R.c_str());
        return 0;
    }
    step(true);
    {
        auto y = readOut();
        std::vector<float> out((size_t)N * 64);
        for (int k8 = 0; k8 < 8; k8++) for (int n = 0; n < N; n++) for (int j = 0; j < 8; j++)
            out[(size_t)n * 64 + k8 * 8 + j] = y[((size_t)k8 * N + n) * 8 + j];
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

}  // namespace anima
