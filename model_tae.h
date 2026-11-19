// Tiny picture decoders and encoders (TAESD family) on the graphics chip, run from the layer list "ugpu.program" that
// tools/pack_tae.py writes (see there for the ops). Test: <ref>/lat.f32 (h, w, C latent, channels innermost),
// <ref>/info.txt "h w", <ref>/rgb.f32 (H, W, 3) from tools/ref_tae.py; the result goes to <ref>/gpu_rgb.f32.
// TAEKERNELS=<file> (default tae.cl next to kernels.cl).
#pragma once
#include "ops.h"
#include <sstream>

namespace tae {

static int run(const std::string &R, int loops) {
    int h = 0, wd0 = 0;
    if (FILE *f = fopen((R + "/info.txt").c_str(), "r")) { if (fscanf(f, "%d %d", &h, &wd0) != 2) h = 0; fclose(f); }
    if (!h) { fprintf(stderr, "%s/info.txt missing or bad\n", R.c_str()); return 1; }
    const bool IMG = !getenv("TAEBUF");   // picture memory (texture cache) unless TAEBUF=1
    char opt[128];
    snprintf(opt, sizeof opt, "-DPX=%s -DOC=%s -DACC=%s", getenv("TPX") ? getenv("TPX") : "4", getenv("TOC") ? getenv("TOC") : "8",
             getenv("TACC") ? getenv("TACC") : "float");
    cl_program prog = build(slurp(getenv("TAEKERNELS") ? getenv("TAEKERNELS") : "tae.cl"), opt);
    cl_kernel kConv = kern(prog, "conv"), kClamp = kern(prog, "clampt"), kConvI = kern(prog, "convi"), kClampI = kern(prog, "clampi");
    cl_kernel kGnS = kern(prog, "gnstats"), kGnA = kern(prog, "gnapply");
    cl_mem gnSt = buf(64 * 2 * 4);
    const int TPX = getenv("TPX") ? atoi(getenv("TPX")) : 4, TOC = getenv("TOC") ? atoi(getenv("TOC")) : 8;

    // ---- the layer list, and the largest activation it needs ----
    struct Op { std::string kind, name; int ci = 0, co = 0, k = 0, bias = 0, relu = 0, up = 0, skip = 0; };
    std::vector<Op> ops;
    std::istringstream ps(GPROG);
    std::string line;
    int C = 0, shuf = 1, keep = 3;
    size_t maxe = 0, maxTW = 0, maxTH = 0;
    { int hh = h, ww = wd0;
      while (std::getline(ps, line)) {
        std::istringstream ls(line); Op o; ls >> o.kind;
        if (o.kind == "conv") { ls >> o.name >> o.ci >> o.co >> o.k >> o.bias >> o.relu >> o.up; if (o.up > 0) { hh *= 2; ww *= 2; } else if (o.up < 0) { hh /= 2; ww /= 2; } C = o.co; }
        else if (o.kind == "block") { ls >> o.name >> o.ci >> o.co >> o.skip; C = o.co; }
        else if (o.kind == "pool") { ls >> o.name >> o.ci >> o.co >> o.k; }   // co = the norm's channels, k = groups
        else if (o.kind == "shuffle") { ls >> shuf; continue; }
        else if (o.kind == "out") { ls >> keep; continue; }
        else if (o.kind != "clamp") { fprintf(stderr, "unknown op %s\n", o.kind.c_str()); return 1; }
        ops.push_back(o);
        maxe = std::max(maxe, (size_t)hh * ww * std::max(std::max(o.ci, o.co), 4));
        maxTW = std::max(maxTW, (size_t)ww * std::max(o.ci, o.co) / 4); maxTH = std::max(maxTH, (size_t)hh);
      }
    }
    auto lat = rdf(R + "/lat.f32");
    const int C0 = (int)(lat.size() / ((size_t)h * wd0));
    maxe = std::max(maxe, lat.size());
    printf("decoder: %zu layers, latent %dx%dx%d, out channels %d, shuffle %d\n", ops.size(), h, wd0, C0, C, shuf);
    cl_mem A = buf(maxe * 2), Bf = buf(maxe * 2), T1 = buf(maxe * 2), T2 = buf(maxe * 2), Sk = buf(maxe * 2);

    int oh = h, ow = wd0;

    // ---- picture-memory path: activations and filters as RGBA half images ----
    auto himg = [&](size_t iw, size_t ih, const void *data) {
        cl_image_format f = {CL_RGBA, CL_HALF_FLOAT};
        cl_image_desc d; memset(&d, 0, sizeof d);
        d.image_type = CL_MEM_OBJECT_IMAGE2D; d.image_width = iw; d.image_height = ih;
        cl_int e;
        cl_mem m = clCreateImage(ctx, data ? (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR) : CL_MEM_READ_WRITE, &f, &d, (void *)data, &e);
        if (e) { fprintf(stderr, "half image %zux%zu failed %d\n", iw, ih, e); exit(1); }
        return m;
    };
    std::map<std::string, cl_mem> WI;   // filter images: row o = [tap][in] halves, the stored layout as is
    cl_mem IA = nullptr, IB = nullptr, IT1 = nullptr, IT2 = nullptr, ISk = nullptr, dummyB = buf(16);
    if (IMG) {
        for (auto &kv : W) {
            if (kv.second.kind != "f16") continue;
            const Wt &t = kv.second;
            std::vector<h16> hv((size_t)t.M * t.K);
            CK(clEnqueueReadBuffer(q, t.m, CL_TRUE, 0, hv.size() * 2, hv.data(), 0, 0, 0));
            WI[kv.first] = himg(t.K / 4, t.M, hv.data());
        }
        IA = himg(maxTW, maxTH, nullptr); IB = himg(maxTW, maxTH, nullptr); IT1 = himg(maxTW, maxTH, nullptr);
        IT2 = himg(maxTW, maxTH, nullptr); ISk = himg(maxTW, maxTH, nullptr);
        printf("picture memory: activations %zux%zu texels, OC %d, PX %d\n", maxTW, maxTH, TOC, TPX);
    }
    auto convI = [&](cl_mem X, const std::string &wn, int H, int Wd, int ci, int co, int k, int up, int relu, cl_mem Rr, cl_mem Y) {
        const bool hb = W.count(wn + ".b") != 0;
        arg(kConvI, 0, X); arg(kConvI, 1, WI.at(wn));
        if (hb) arg(kConvI, 2, w(wn + ".b")); else arg(kConvI, 2, dummyB);
        arg(kConvI, 3, Rr ? Rr : X);
        arg(kConvI, 4, Y); arg(kConvI, 5, H); arg(kConvI, 6, Wd); arg(kConvI, 7, ci); arg(kConvI, 8, co); arg(kConvI, 9, k);
        arg(kConvI, 10, up); arg(kConvI, 11, relu); arg(kConvI, 12, (int)hb); arg(kConvI, 13, (int)(Rr != nullptr));
        run3(kConvI, wn.c_str(), (co + TOC - 1) / TOC, (Wd + TPX - 1) / TPX, H, 1, 16, 4);
    };
    auto conv = [&](cl_mem X, const std::string &wn, int H, int Wd, int ci, int co, int k, int up, int relu, cl_mem Rr, cl_mem Y) {
        const bool hb = W.count(wn + ".b") != 0;
        arg(kConv, 0, X); arg(kConv, 1, w(wn));
        if (hb) arg(kConv, 2, w(wn + ".b")); else arg(kConv, 2, X);
        if (Rr) arg(kConv, 3, Rr); else arg(kConv, 3, X);
        arg(kConv, 4, Y); arg(kConv, 5, H); arg(kConv, 6, Wd); arg(kConv, 7, ci); arg(kConv, 8, co); arg(kConv, 9, k);
        arg(kConv, 10, up); arg(kConv, 11, relu); arg(kConv, 12, (int)hb); arg(kConv, 13, (int)(Rr != nullptr));
        run3(kConv, wn.c_str(), co / 4, (Wd + 3) / 4, H, std::min(co / 4, 16), 4, 1);
    };
    auto decode = [&]() {
        std::vector<h16> x16(lat.size());
        for (size_t i = 0; i < lat.size(); i++) x16[i] = (h16)lat[i];
        CK(clEnqueueWriteBuffer(q, A, CL_FALSE, 0, x16.size() * 2, x16.data(), 0, 0, 0));
        cl_mem cur = A, nxt = Bf;
        int H = h, Wd = wd0;
        if (IMG) {
            size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)wd0 * C0 / 4, (size_t)h, 1};
            CK(clEnqueueWriteImage(q, IA, CL_FALSE, org, reg, 0, 0, x16.data(), 0, 0, 0));
            cur = IA; nxt = IB;
            for (auto &o : ops) {
                if (o.kind == "clamp") {
                    const int tw = Wd * C0 / 4;
                    arg(kClampI, 0, cur); arg(kClampI, 1, nxt); arg(kClampI, 2, tw); arg(kClampI, 3, H);
                    run(kClampI, "clamp", tw, H, 1, 16, 4);
                    std::swap(cur, nxt);
                } else if (o.kind == "pool") {
                    const int C4 = o.co / 4, GC4 = C4 / o.k;
                    convI(cur, o.name + ".p0", H, Wd, o.ci, o.co, 1, 0, 0, nullptr, IT1);
                    arg(kGnS, 0, IT1); arg(kGnS, 1, gnSt); arg(kGnS, 2, Wd); arg(kGnS, 3, H); arg(kGnS, 4, C4); arg(kGnS, 5, GC4);
                    run(kGnS, "gnstats", (size_t)o.k * 256, 1, 1, 256, 1);
                    arg(kGnA, 0, IT1); arg(kGnA, 1, IT2); arg(kGnA, 2, gnSt); arg(kGnA, 3, w(o.name + ".gn.w")); arg(kGnA, 4, w(o.name + ".gn.b"));
                    arg(kGnA, 5, Wd); arg(kGnA, 6, H); arg(kGnA, 7, C4); arg(kGnA, 8, GC4);
                    run(kGnA, "gnapply", (size_t)Wd * C4, H, 1, 16, 4);
                    convI(IT2, o.name + ".p3", H, Wd, o.co, o.ci, 1, 0, 0, cur, nxt);
                    std::swap(cur, nxt);
                } else if (o.kind == "conv") {
                    if (o.up > 0) { H *= 2; Wd *= 2; } else if (o.up < 0) { H /= 2; Wd /= 2; }
                    convI(cur, o.name, H, Wd, o.ci, o.co, o.k, o.up, o.relu, nullptr, nxt);
                    std::swap(cur, nxt);
                } else {
                    convI(cur, o.name + ".c0", H, Wd, o.ci, o.co, 3, 0, 1, nullptr, IT1);
                    convI(IT1, o.name + ".c1", H, Wd, o.co, o.co, 3, 0, 1, nullptr, IT2);
                    cl_mem sk = cur;
                    if (o.skip) { convI(cur, o.name + ".skip", H, Wd, o.ci, o.co, 1, 0, 0, nullptr, ISk); sk = ISk; }
                    convI(IT2, o.name + ".c2", H, Wd, o.co, o.co, 3, 0, 1, sk, nxt);
                    std::swap(cur, nxt);
                }
            }
        } else
        for (auto &o : ops) {
            if (o.kind == "pool") { fprintf(stderr, "pool layers need the picture-memory path (unset TAEBUF)\n"); exit(1); }
            if (o.kind == "clamp") {
                const int n = H * Wd * C0;
                arg(kClamp, 0, cur); arg(kClamp, 1, n);
                run(kClamp, "clamp", n, 1, 1, 64, 1);
            } else if (o.kind == "conv") {
                if (o.up > 0) { H *= 2; Wd *= 2; } else if (o.up < 0) { H /= 2; Wd /= 2; }
                conv(cur, o.name, H, Wd, o.ci, o.co, o.k, o.up, o.relu, nullptr, nxt);
                std::swap(cur, nxt);
            } else {   // block
                conv(cur, o.name + ".c0", H, Wd, o.ci, o.co, 3, 0, 1, nullptr, T1);
                conv(T1, o.name + ".c1", H, Wd, o.co, o.co, 3, 0, 1, nullptr, T2);
                cl_mem s = cur;
                if (o.skip) { conv(cur, o.name + ".skip", H, Wd, o.ci, o.co, 1, 0, 0, nullptr, Sk); s = Sk; }
                conv(T2, o.name + ".c2", H, Wd, o.co, o.co, 3, 0, 1, s, nxt);
                std::swap(cur, nxt);
            }
        }
        oh = H; ow = Wd;
        std::vector<h16> y((size_t)H * Wd * C);
        if (IMG) {
            size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)Wd * C / 4, (size_t)H, 1};
            CK(clEnqueueReadImage(q, cur, CL_TRUE, org, reg, 0, 0, y.data(), 0, 0, 0));
        } else CK(clEnqueueReadBuffer(q, cur, CL_TRUE, 0, y.size() * 2, y.data(), 0, 0, 0));
        // channels -> shuf x shuf pixels (PixelShuffle: channel c * r * r + dy * r + dx), keep the first channels
        const int Cp = C / (shuf * shuf), OH = H * shuf, OW = Wd * shuf;
        std::vector<float> rgb((size_t)OH * OW * keep);
        for (int yy = 0; yy < H; yy++) for (int xx = 0; xx < Wd; xx++)
            for (int c = 0; c < keep && c < Cp; c++) for (int dy = 0; dy < shuf; dy++) for (int dx = 0; dx < shuf; dx++)
                rgb[(((size_t)(yy * shuf + dy)) * OW + xx * shuf + dx) * keep + c] =
                    (float)y[((size_t)yy * Wd + xx) * C + c * shuf * shuf + dy * shuf + dx];
        oh = OH; ow = OW;
        return rgb;
    };
    auto rgb = decode();
    printf("picture %dx%d\n", ow, oh);
    if (FILE *f = fopen((R + "/rgb.f32").c_str(), "rb")) { fclose(f); printf("  error vs reference %.5f\n", relerr(rgb, rdf(R + "/rgb.f32"))); }
    FILE *f = fopen((R + "/gpu_rgb.f32").c_str(), "wb"); fwrite(rgb.data(), 4, rgb.size(), f); fclose(f);
    tsum.clear();
    for (int l = 0; l < loops; l++) {
        clFinish(q); auto a = std::chrono::steady_clock::now();
        decode();
        printf("decode %d: %.1f ms\n", l, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
    }
    if (prof) {
        double tot = 0; for (auto &kv : tsum) tot += kv.second;
        std::vector<std::pair<double, std::string>> v; for (auto &kv : tsum) v.push_back({kv.second, kv.first});
        std::sort(v.rbegin(), v.rend());
        for (size_t i = 0; i < v.size() && i < 12; i++) printf("  %-14s %8.1f ms  %4.1f%%\n", v[i].second.c_str(), v[i].first / loops, 100 * v[i].first / tot);
    }
    return 0;
}

}  // namespace tae
