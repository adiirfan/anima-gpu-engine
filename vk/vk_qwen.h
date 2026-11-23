// Qwen-Image 2.1 on Vulkan (width 4096, 32 heads, 32 blocks, SwiGLU 12288), the same test folders and file as the
// OpenCL engine (model_qwen.h), with the 4-bit two-level-scale file (tools/pack_qwen.py, Q4K=1) on the matrix
// hardware. The prompt's 128 rows go through the blocks once per picture (modulated at t = 0, one-way attention among
// the first nc of them) and each block keeps its prompt keys / values. Every step then runs only the picture rows:
// they attend to [picture keys 0..Ni, kept prompt keys Ni..Ni + nc] (rows up to Na = Ni + 128 rounded up to 128;
// the rest stay zero and are masked). Gates go through tanh on the host.
//   env: DRAW=1 (whole picture from noise.f32 along sigmas.f32 into draw.f32)
#pragma once
#include "vkops.h"

namespace qwen {
static const int NB = 32, Dm = 4096, HEADS = 32, F = 12288, T = 128;

// rotations (rotate-half, pair i at channels i and 64 + i): P = [cos, cos], Q = [-sin, sin]; axes 16 / 56 / 56
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

static int run(const std::string &Rd, int loops) {
    if (autoDeclines("b0.qkv")) return 3;
    // the blocks' tables are read from storage block by block, by the prompt pass and by every step
    const bool stream = streamWanted();
    if (stream) streamOf = [](const std::string &n) { int b; char suf[32]; return sscanf(n.c_str(), "b%d.%31s", &b, suf) == 2 ? b : -1; };
    auto runG = [&](vk::Graph &g, std::vector<std::pair<std::string, double>> *tm) {
        if (stream) g.runStreamed([](size_t i) { fillBlock((int)i); }, tm);
        else g.run(tm);
    };
    auto t0 = std::chrono::steady_clock::now();
    int nc = 0, PW = 0, PH = 0;
    if (FILE *f = fopen((Rd + "/info.txt").c_str(), "r")) {
        if (fscanf(f, "%d %d %d", &nc, &PW, &PH) != 3) nc = 0;
        fclose(f);
    }
    if (!nc || nc > T) { fprintf(stderr, "%s/info.txt missing or bad\n", Rd.c_str()); return 1; }
    const int gh = PH / 16, gw = PW / 16, Ni = gh * gw, Na = pad128(Ni + T), NR = std::max(Ni, T);
    const int hc = headsPerPass(HEADS, Na), hcT = headsPerPass(HEADS, T);
    printf("%dx%d: %d picture rows, prompt %d of %d rows, attention over %d rows, %d heads at a time\n", PW, PH, Ni, nc, T, Na, hc);

    // ---- rotations: prompt rows (p, p, p), picture rows (nc, y, x) centred ----
    std::vector<float> idt((size_t)T * 3), idi((size_t)Ni * 3), Pt, Qt, Pi, Qi;
    for (int p = 0; p < T; p++) idt[p * 3] = idt[p * 3 + 1] = idt[p * 3 + 2] = (float)p;
    for (int y = 0; y < gh; y++) for (int x = 0; x < gw; x++) {
        float *pp = &idi[((size_t)y * gw + x) * 3];
        pp[0] = (float)nc; pp[1] = (float)(y - (gh - gh / 2)); pp[2] = (float)(x - (gw - gw / 2));
    }
    rope(idt, T, Pt, Qt); rope(idi, Ni, Pi, Qi);
    Buf bPt = up(Pt.data(), Pt.size() * 4), bQt = up(Qt.data(), Qt.size() * 4);
    Buf bPi = up(Pi.data(), Pi.size() * 4), bQi = up(Qi.data(), Qi.size() * 4);

    // ---- the prompt rows: RMS norm with txt_norm (its +1 is in the file) on the host ----
    auto ctx = rdf(Rd + "/ctx.f32");
    if (ctx.size() < (size_t)T * Dm) { fprintf(stderr, "ctx.f32 needs %d x %d numbers\n", T, Dm); return 1; }
    {
        const float *tn = (const float *)ft("txt_norm").src;
        for (int r = 0; r < T; r++) {
            double ss = 0;
            for (int k = 0; k < Dm; k++) ss += (double)ctx[(size_t)r * Dm + k] * ctx[(size_t)r * Dm + k];
            const float rs = 1.0f / sqrtf((float)(ss / Dm) + 1e-6f);
            for (int k = 0; k < Dm; k++) ctx[(size_t)r * Dm + k] *= rs * tn[k];
        }
    }
    Buf Xc = vk::buf((size_t)T * Dm * 2);
    putH(Xc, ctx.data(), (size_t)T * Dm);

    // ---- work buffers ----
    auto zeroed = [](size_t bytes) { Buf b = vk::buf(bytes); memset(b.map, 0, bytes); return b; };
    Buf res = vk::buf((size_t)Ni * Dm * 4), resT = vk::buf((size_t)T * Dm * 4), C1 = vk::buf((size_t)T * Dm * 2);
    Buf QKV = vk::buf((size_t)NR * 3 * Dm * 2), O = vk::buf((size_t)Na * Dm * 2), Hh = vk::buf((size_t)NR * F * 2);
    Buf Ximg = vk::buf((size_t)Ni * 64 * 2), Xa = vk::buf((size_t)Ni * Dm * 2), Yout = vk::buf((size_t)Ni * 64 * 2);
    Buf Qh = zeroed((size_t)HEADS * Na * 128 * 2), Kh = zeroed((size_t)HEADS * Na * 128 * 2), Vh = zeroed((size_t)HEADS * Na * 128 * 2);
    Buf QhT = zeroed((size_t)HEADS * T * 128 * 2), KhT = zeroed((size_t)HEADS * T * 128 * 2), VhT = zeroed((size_t)HEADS * T * 128 * 2);
    Buf S = vk::buf(std::max((size_t)hc * Na * Na, (size_t)hcT * T * T) * 2);
    std::vector<Buf> cK(NB), cV(NB);
    for (int b = 0; b < NB; b++) { cK[b] = vk::buf((size_t)HEADS * T * 128 * 2); cV[b] = vk::buf((size_t)HEADS * T * 128 * 2); }
    scratch(NR, F);
    Buf bTemb = vk::buf(256 * 4), bT1 = vk::buf(Dm * 4), bTe = vk::buf(Dm * 4), bMod = vk::buf(4 * Dm * 4), bSc = vk::buf(Dm * 4);
    Buf mods = vk::buf(4 * Dm * 4), mods0 = vk::buf(4 * Dm * 4), fin = vk::buf(Dm * 4), zero = zeroed(Dm * 4);
    auto part = [&](Buf &m, int i) { return R(m, (size_t)i * Dm * 4, Dm * 4); };
    const float qscale = 1.0f / sqrtf(128.0f);
    const int HW = T * 128 / 2;   // one head's kept keys in 32-bit words

    // ---- graph 1: modulations from the noise level ----
    vk::Graph gmod; G = &gmod;
    gemv(wh("t1"), R(bTemb), R(dummy), R(bT1), Dm, 256, 0);
    gemv(wh("t2"), R(bT1), R(dummy), R(bTe), Dm, Dm, 1);
    gemv(wh("mod"), R(bTe), R(dummy), R(bMod), 4 * Dm, Dm, 1);
    gemv(wh("out_scale"), R(bTe), R(dummy), R(bSc), Dm, Dm, 1);

    // ---- graph 2: the prompt pass (once per picture) ----
    vk::Graph gtxt; G = &gtxt;
    rotX("txt_in1", R(Xc), Dm, Dm, T);
    mulq(1, "txt_in1", "txt_in1", R(C1), R(dummy), R(dummy), T);
    rotX("txt_in2", R(C1), Dm, Dm, T);
    mulq(3, "txt_in2", "txt_in2", R(dummy), R(resT), R(dummy), T);
    for (int b = 0; b < NB; b++) {
        const std::string p = "b" + std::to_string(b) + ".";
        if (b) G->mark();   // segment b: block b
        rotLn(p + "qkv", R(resT), R(zero), part(mods0, 0), Dm, T);
        mulq(0, "qkv", p + "qkv", R(QKV), R(dummy), R(dummy), T);
        prep(R(QKV), 3 * Dm, 0, R(wf(p + "qn")), R(bPt), R(bQt), R(QhT), T, T, HEADS, true, true, qscale, "prep q");
        prep(R(QKV), 3 * Dm, Dm, R(wf(p + "kn")), R(bPt), R(bQt), R(KhT), T, T, HEADS, true, true, 1.0f, "prep k");
        prep(R(QKV), 3 * Dm, 2 * Dm, R(dummy), R(bPt), R(bQt), R(VhT), T, T, HEADS, false, false, 1.0f, "prep v");
        copyf(R(KhT), R(cK[b]), (size_t)HEADS * HW);
        copyf(R(VhT), R(cV[b]), (size_t)HEADS * HW);
        attend(QhT, KhT, VhT, S, R(O), T, nc, HEADS, hcT, Dm, true);
        rotX(p + "o", R(O), Dm, Dm, T);
        mulq(2, "o", p + "o", R(dummy), R(resT), part(mods0, 1), T);
        rotLn(p + "mlp_in", R(resT), R(zero), part(mods0, 2), Dm, T);
        mulq(4, "mlp_in", p + "mlp_in", R(Hh), R(dummy), R(dummy), T);
        rotX(p + "down", R(Hh), F, F, T);
        mulq(2, "down", p + "down", R(dummy), R(resT), part(mods0, 3), T);
    }

    // ---- graph 3: one step over the picture rows ----
    vk::Graph gstep; G = &gstep;
    mulh(3, "img_in", 0, R(Ximg), R(wh("img_in")), R(dummy), R(res), Ni, Dm, 64, 64, 64, Dm);
    for (int b = 0; b < NB; b++) {
        const std::string p = "b" + std::to_string(b) + ".";
        if (b) G->mark();   // segment b: block b
        rotLn(p + "qkv", R(res), R(zero), part(mods, 0), Dm, Ni);
        mulq(0, "qkv", p + "qkv", R(QKV), R(dummy), R(dummy), Ni);
        prep(R(QKV), 3 * Dm, 0, R(wf(p + "qn")), R(bPi), R(bQi), R(Qh), Ni, Na, HEADS, true, true, qscale, "prep q");
        prep(R(QKV), 3 * Dm, Dm, R(wf(p + "kn")), R(bPi), R(bQi), R(Kh), Ni, Na, HEADS, true, true, 1.0f, "prep k");
        prep(R(QKV), 3 * Dm, 2 * Dm, R(dummy), R(bPi), R(bQi), R(Vh), Ni, Na, HEADS, false, false, 1.0f, "prep v");
        copyh(R(cK[b]), R(Kh, (size_t)Ni * 256), HW, HEADS, HW, Na * 64);
        copyh(R(cV[b]), R(Vh, (size_t)Ni * 256), HW, HEADS, HW, Na * 64);
        attend(Qh, Kh, Vh, S, R(O), Na, Ni + nc, HEADS, hc, Dm);
        rotX(p + "o", R(O), Dm, Dm, Ni);
        mulq(2, "o", p + "o", R(dummy), R(res), part(mods, 1), Ni);
        rotLn(p + "mlp_in", R(res), R(zero), part(mods, 2), Dm, Ni);
        mulq(4, "mlp_in", p + "mlp_in", R(Hh), R(dummy), R(dummy), Ni);
        rotX(p + "down", R(Hh), F, F, Ni);
        mulq(2, "down", p + "down", R(dummy), R(res), part(mods, 3), Ni);
    }
    lnmod(R(res), R(zero), R(fin), R(Xa), Ni, Dm);
    mulh(0, "final", 0, R(Xa), R(wh("proj_out")), R(Yout), R(dummy), Ni, 64, Dm, Dm, Dm, 64);
    vk::savePipelines();
    printf("setup: %.1f s rearranging weights, %.1f s building shaders\n", tWeights, tKern);
    printf("loaded and recorded in %.1f s: %.0f MB of weights on the chip, %zu dispatches per step\n",
           secsSince(t0), chipBytes / 1e6, gstep.ops.size());
    if (stream) printf("streaming the blocks from storage: %.0f MB of slots\n", slotBytes / 1e6);

    // the modulations for noise level s into m: [s1, tanh g1, s2, tanh g2]; the output norm's scale into fin
    auto modsFor = [&](float s, Buf &m) {
        float *te = (float *)bTemb.map;
        for (int i = 0; i < 128; i++) {
            const float a = 1000.0f * s * expf(-logf(10000.0f) * i / 128.0f);
            te[i] = cosf(a); te[128 + i] = sinf(a);
        }
        gmod.run();
        const float *v = (const float *)bMod.map;
        float *d = (float *)m.map;
        for (int k = 0; k < 4 * Dm; k++) d[k] = (k / Dm) % 2 ? tanhf(v[k]) : v[k];
        memcpy(fin.map, bSc.map, Dm * 4);
    };
    {
        auto a = std::chrono::steady_clock::now();
        modsFor(0.0f, mods0);
        runG(gtxt, nullptr);
        printf("prompt pass %.0f ms\n", secsSince(a) * 1000);
    }

    std::vector<std::pair<std::string, double>> times;
    auto x = rdf(Rd + "/noise.f32"), sg = rdf(Rd + "/sigmas.f32");
    if (x.size() != (size_t)Ni * 64) { fprintf(stderr, "noise.f32 needs %d x 64 numbers\n", Ni); return 1; }
    sg.push_back(0.0f);
    auto step = [&](float s) {
        putH(Ximg, x.data(), x.size());
        modsFor(s, mods);
        runG(gstep, &times);
        return readH(Yout, (size_t)Ni * 64);
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
        if (fexists(Rd + "/vel.f32")) {
            auto ref = rdf(Rd + "/vel.f32");
            printf("  first step   error %.5f\n", relerr(v.data(), ref.data(), v.size()));
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

}  // namespace qwen
