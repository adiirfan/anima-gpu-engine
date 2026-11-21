// The full Wan 2.2 VAE encoder for one picture (a clip's start frame), from the layer list tools/pack_wanenc.py writes
// (see there for the ops). Activations live in picture memory (RGBA half images, texel (x * C4 + c4, y)), filters too;
// the filters and norms are tae.cl's.
// Input: <dir>/lat.f32 = the picture * 2 - 1, pixel-unshuffled 2 x 2: (H/2, W/2, 12), <dir>/info.txt "H/2 W/2".
// Output: <dir>/gpu_rgb.f32 = the raw latent mu (H/16, W/16, 48); <dir>/ref.f32 present -> its error is printed.
#pragma once
#include "ops.h"
#include <array>
#include <sstream>

namespace vaeenc {

static int run(const std::string &R, int loops) {
    int h = 0, wd = 0;
    if (FILE *f = fopen((R + "/info.txt").c_str(), "r")) { if (fscanf(f, "%d %d", &h, &wd) != 2) h = 0; fclose(f); }
    if (!h) { fprintf(stderr, "%s/info.txt missing or bad\n", R.c_str()); return 1; }
    char opt[128];
    snprintf(opt, sizeof opt, "-DPX=%s -DOC=%s -DACC=float", getenv("TPX") ? getenv("TPX") : "4", getenv("TOC") ? getenv("TOC") : "8");
    const int TPX = getenv("TPX") ? atoi(getenv("TPX")) : 4, TOC = getenv("TOC") ? atoi(getenv("TOC")) : 8;
    cl_program prog = build(slurp(getenv("TAEKERNELS") ? getenv("TAEKERNELS") : "tae.cl"), opt);
    cl_kernel kConvI = kern(prog, "convi"), kNorm = kern(prog, "rmsni"), kAdd = kern(prog, "addi");
    cl_kernel kS = kern(prog, "attns"), kSm = kern(prog, "attnsm"), kO = kern(prog, "attno");

    // ---- the layer list; the widest and tallest activation ----
    struct Op { std::string kind, name, slot; int ci = 0, co = 0, k = 0, mode = 0; };
    std::vector<Op> ops;
    std::istringstream ps(GPROG);
    std::string line;
    int outC = 48;
    size_t maxTW = 0, maxTH = 0, maxN = 0;
    { int hh = h, ww = wd, C = 12;
      std::map<std::string, std::array<int, 3>> sd;   // each slot's size, restored by "load"
      maxTW = (size_t)ww * C / 4; maxTH = hh;
      while (std::getline(ps, line)) {
        std::istringstream ls(line); Op o; ls >> o.kind;
        if (o.kind == "conv") { ls >> o.name >> o.ci >> o.co >> o.k >> o.mode; if (o.mode < 0) { hh /= 2; ww /= 2; } C = o.co; }
        else if (o.kind == "convr") { ls >> o.name >> o.ci >> o.co >> o.k >> o.slot; C = o.co; }
        else if (o.kind == "norm") { ls >> o.name >> o.co >> o.mode; }
        else if (o.kind == "attn") { ls >> o.name >> o.co; maxN = std::max(maxN, (size_t)hh * ww); }
        else if (o.kind == "save" || o.kind == "load" || o.kind == "add") {
            ls >> o.slot;
            if (o.kind == "save") sd[o.slot] = {hh, ww, C};
            else if (o.kind == "load") { auto d = sd.at(o.slot); hh = d[0]; ww = d[1]; C = d[2]; }
        }
        else if (o.kind == "out") { ls >> outC; continue; }
        else { fprintf(stderr, "unknown op %s\n", o.kind.c_str()); return 1; }
        ops.push_back(o);
        maxTW = std::max(maxTW, (size_t)ww * std::max(C, std::max(o.ci, o.co)) / 4);
        maxTH = std::max(maxTH, (size_t)hh);
      }
    }
    size_t maxIW = 0; CK(clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof maxIW, &maxIW, nullptr));
    if (maxTW > maxIW) { fprintf(stderr, "encoder needs pictures %zu texels wide, the chip allows %zu (picture too large)\n", maxTW, maxIW); return 1; }
    auto lat = rdf(R + "/lat.f32");
    if (lat.size() != (size_t)h * wd * 12) { fprintf(stderr, "lat.f32 has %zu numbers, not %d x %d x 12\n", lat.size(), h, wd); return 1; }
    printf("encoder: %zu layers, input %dx%dx12, activations %zux%zu texels\n", ops.size(), wd, h, maxTW, maxTH);

    auto himg = [&](size_t iw, size_t ih, const void *data) {
        cl_image_format f = {CL_RGBA, CL_HALF_FLOAT};
        cl_image_desc d; memset(&d, 0, sizeof d);
        d.image_type = CL_MEM_OBJECT_IMAGE2D; d.image_width = iw; d.image_height = ih;
        cl_int e;
        cl_mem m = clCreateImage(ctx, data ? (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR) : CL_MEM_READ_WRITE, &f, &d, (void *)data, &e);
        if (e) { fprintf(stderr, "half image %zux%zu failed %d\n", iw, ih, e); exit(1); }
        return m;
    };
    std::map<std::string, cl_mem> WI;   // filter images: row o = [tap][in] halves
    for (auto &kv : W) {
        if (kv.second.kind != "f16") continue;
        const Wt &t = kv.second;
        std::vector<h16> hv((size_t)t.M * t.K);
        CK(clEnqueueReadBuffer(q, t.m, CL_TRUE, 0, hv.size() * 2, hv.data(), 0, 0, 0));
        WI[kv.first] = himg(t.K / 4, t.M, hv.data());
    }
    cl_mem dummyB = buf(16), S = maxN ? buf(maxN * maxN * 4) : nullptr;

    // activation pool: an image is back in the pool once neither the current activation nor a slot holds it
    std::vector<cl_mem> pool, all;
    std::map<std::string, cl_mem> slot;
    std::map<std::string, std::array<int, 3>> slotDim;
    cl_mem cur = nullptr;
    auto get = [&]() { if (!pool.empty()) { cl_mem m = pool.back(); pool.pop_back(); return m; }
                       cl_mem m = himg(maxTW, maxTH, nullptr); all.push_back(m); return m; };
    auto held = [&](cl_mem m) { if (m == cur) return true; for (auto &kv : slot) if (kv.second == m) return true; return false; };
    auto drop = [&](cl_mem m) { if (m && !held(m) && std::find(pool.begin(), pool.end(), m) == pool.end()) pool.push_back(m); };
    auto setCur = [&](cl_mem n) { cl_mem old = cur; cur = n; drop(old); };

    auto convI = [&](cl_mem X, const std::string &wn, int H, int Wd, int ci, int co, int k, int up, cl_mem Rr, cl_mem Y) {
        const bool hb = W.count(wn + ".b") != 0;
        arg(kConvI, 0, X); arg(kConvI, 1, WI.at(wn));
        if (hb) arg(kConvI, 2, w(wn + ".b")); else arg(kConvI, 2, dummyB);
        arg(kConvI, 3, Rr ? Rr : X);
        int zero = 0;
        arg(kConvI, 4, Y); arg(kConvI, 5, H); arg(kConvI, 6, Wd); arg(kConvI, 7, ci); arg(kConvI, 8, co); arg(kConvI, 9, k);
        arg(kConvI, 10, up); arg(kConvI, 11, zero); arg(kConvI, 12, (int)hb); arg(kConvI, 13, (int)(Rr != nullptr));
        run3(kConvI, wn.c_str(), (co + TOC - 1) / TOC, (Wd + TPX - 1) / TPX, H, 1, 16, 4);
    };
    auto norm = [&](cl_mem X, const std::string &wn, int H, int Wd, int C, int silu, cl_mem Y) {
        const int C4 = C / 4;
        arg(kNorm, 0, X); arg(kNorm, 1, Y); arg(kNorm, 2, w(wn)); arg(kNorm, 3, Wd); arg(kNorm, 4, H); arg(kNorm, 5, C4); arg(kNorm, 6, silu);
        run(kNorm, "norm", Wd, H, 1, 16, 4);
    };

    std::vector<float> out;
    int oh = 0, ow = 0;
    auto encode = [&]() {
        slot.clear(); cur = nullptr; pool.clear();
        for (auto m : all) pool.push_back(m);
        std::vector<h16> x16(lat.size());
        for (size_t i = 0; i < lat.size(); i++) x16[i] = (h16)lat[i];
        cl_mem in = get();
        { size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)wd * 3, (size_t)h, 1};
          CK(clEnqueueWriteImage(q, in, CL_FALSE, org, reg, 0, 0, x16.data(), 0, 0, 0)); }
        cur = in;
        int H = h, Wd = wd, C = 12;
        for (auto &o : ops) {
            if (o.kind == "conv") {
                if (o.mode < 0) { H /= 2; Wd /= 2; }
                cl_mem n = get(); convI(cur, o.name, H, Wd, o.ci, o.co, o.k, o.mode, nullptr, n); setCur(n); C = o.co;
            } else if (o.kind == "convr") {
                cl_mem n = get(); convI(cur, o.name, H, Wd, o.ci, o.co, o.k, 0, slot.at(o.slot), n); setCur(n); C = o.co;
            } else if (o.kind == "norm") {
                cl_mem n = get(); norm(cur, o.name, H, Wd, o.co, o.mode, n); setCur(n);
            } else if (o.kind == "attn") {
                const int N = H * Wd, C4 = o.co / 4;
                cl_mem nx = get(); norm(cur, o.name + ".n", H, Wd, o.co, 0, nx);
                cl_mem Q = get(), K = get(), V = get();
                convI(nx, o.name + ".q", H, Wd, o.co, o.co, 1, 0, nullptr, Q);
                convI(nx, o.name + ".k", H, Wd, o.co, o.co, 1, 0, nullptr, K);
                convI(nx, o.name + ".v", H, Wd, o.co, o.co, 1, 0, nullptr, V);
                const float sc = 1.0f / sqrtf((float)o.co);
                arg(kS, 0, Q); arg(kS, 1, K); arg(kS, 2, S); arg(kS, 3, Wd); arg(kS, 4, N); arg(kS, 5, C4); arg(kS, 6, sc);
                run(kS, "attn qk", N, N, 1, 16, 16);
                arg(kSm, 0, S); arg(kSm, 1, N);
                run(kSm, "attn softmax", N, 1, 1, 64, 1);
                arg(kO, 0, V); arg(kO, 1, S); arg(kO, 2, nx); arg(kO, 3, Wd); arg(kO, 4, N); arg(kO, 5, C4);
                run(kO, "attn pv", C4, N, 1, 4, 64);
                cl_mem n = get(); convI(nx, o.name + ".proj", H, Wd, o.co, o.co, 1, 0, cur, n);
                drop(Q); drop(K); drop(V); drop(nx);   // (not held: back to the pool)
                setCur(n);
            } else if (o.kind == "save") {
                cl_mem old = slot.count(o.slot) ? slot[o.slot] : nullptr;
                slot[o.slot] = cur; slotDim[o.slot] = {H, Wd, C}; drop(old);
            } else if (o.kind == "load") {
                setCur(slot.at(o.slot));
                auto d = slotDim.at(o.slot); H = d[0]; Wd = d[1]; C = d[2];
            } else if (o.kind == "add") {
                const int tw = Wd * C / 4;
                cl_mem n = get();
                arg(kAdd, 0, cur); arg(kAdd, 1, slot.at(o.slot)); arg(kAdd, 2, n); arg(kAdd, 3, tw); arg(kAdd, 4, H);
                run(kAdd, "add", tw, H, 1, 16, 4);
                setCur(n);
            }
        }
        std::vector<h16> y((size_t)H * Wd * C);
        { size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)Wd * C / 4, (size_t)H, 1};
          CK(clEnqueueReadImage(q, cur, CL_TRUE, org, reg, 0, 0, y.data(), 0, 0, 0)); }
        out.assign((size_t)H * Wd * outC, 0.0f);
        for (size_t p = 0; p < (size_t)H * Wd; p++) for (int c = 0; c < outC; c++) out[p * outC + c] = (float)y[p * C + c];
        oh = H; ow = Wd;
    };
    encode();
    printf("latent %dx%dx%d\n", ow, oh, outC);
    if (FILE *f = fopen((R + "/ref.f32").c_str(), "rb")) { fclose(f); printf("  error vs reference %.5f\n", relerr(out, rdf(R + "/ref.f32"))); }
    FILE *f = fopen((R + "/gpu_rgb.f32").c_str(), "wb"); fwrite(out.data(), 4, out.size(), f); fclose(f);
    tsum.clear();
    for (int l = 0; l < loops; l++) {
        clFinish(q); auto a = std::chrono::steady_clock::now();
        encode();
        printf("encode %d: %.1f ms\n", l, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
    }
    if (prof) {
        double tot = 0; for (auto &kv : tsum) tot += kv.second;
        std::vector<std::pair<double, std::string>> v; for (auto &kv : tsum) v.push_back({kv.second, kv.first});
        std::sort(v.rbegin(), v.rend());
        for (size_t i = 0; i < v.size() && i < 12; i++) printf("  %-14s %8.1f ms  %4.1f%%\n", v[i].second.c_str(), v[i].first / loops, 100 * v[i].first / tot);
    }
    return 0;
}

}  // namespace vaeenc
