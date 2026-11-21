// Tiny picture decoders and encoders (TAESD family) on the graphics chip, run from the layer list "ugpu.program" that
// tools/pack_tae.py writes (see there for the ops). Test: <ref>/lat.f32 (h, w, C latent, channels innermost),
// <ref>/info.txt "h w", <ref>/rgb.f32 (H, W, 3) from tools/ref_tae.py; the result goes to <ref>/gpu_rgb.f32.
// TAEKERNELS=<file> (default tae.cl next to kernels.cl).
#pragma once
#include "ops.h"
#include <sstream>

namespace tae {

static int run(const std::string &R, int loops) {
    int h = 0, wd0 = 0, NT = 1;   // NT: latent frames in lat.f32 (a whole clip, for the "frames all" decoders)
    if (FILE *f = fopen((R + "/info.txt").c_str(), "r")) { if (fscanf(f, "%d %d", &h, &wd0) != 2) h = 0; if (fscanf(f, "%d", &NT) != 1) NT = 1; fclose(f); }
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
    bool HV = false;   // the Wan 2.1 family (mem / tgrow): frames, picture = the last one
    bool allFrames = false;   // "frames all": every frame of the clip comes out (Wan 2.2's decoder), not just the last
    size_t maxe = 0, maxTW = 0, maxTH = 0;
    { int hh = h, ww = wd0;
      while (std::getline(ps, line)) {
        std::istringstream ls(line); Op o; ls >> o.kind;
        if (o.kind == "conv") { ls >> o.name >> o.ci >> o.co >> o.k >> o.bias >> o.relu >> o.up; if (o.up > 0) { hh *= 2; ww *= 2; } else if (o.up < 0) { hh /= 2; ww /= 2; } C = o.co; }
        else if (o.kind == "block") { ls >> o.name >> o.ci >> o.co >> o.skip; C = o.co; }
        else if (o.kind == "pool") { ls >> o.name >> o.ci >> o.co >> o.k; }   // co = the norm's channels, k = groups
        else if (o.kind == "mem") { ls >> o.name >> o.ci >> o.co >> o.skip; C = o.co; HV = true; }
        else if (o.kind == "tgrow") { ls >> o.name >> o.ci >> o.k >> o.up; o.co = o.ci; if (o.up > 0) { hh *= 2; ww *= 2; } HV = true; }
        else if (o.kind == "shuffle") { ls >> shuf; continue; }
        else if (o.kind == "out") { ls >> keep; continue; }
        else if (o.kind == "frames") { std::string v; ls >> v; allFrames = v == "all"; continue; }
        else if (o.kind != "clamp") { fprintf(stderr, "unknown op %s\n", o.kind.c_str()); return 1; }
        ops.push_back(o);
        maxe = std::max(maxe, (size_t)hh * ww * std::max(std::max(o.ci, o.co), 4));
        maxTW = std::max(maxTW, (size_t)ww * std::max(o.ci, o.co) / 4); maxTH = std::max(maxTH, (size_t)hh);
      }
    }
    auto lat = rdf(R + "/lat.f32");
    const int C0 = (int)(lat.size() / ((size_t)h * wd0 * NT));
    maxe = std::max(maxe, lat.size());
    printf("decoder: %zu layers, latent %dx%dx%d, out channels %d, shuffle %d\n", ops.size(), h, wd0, C0, C, shuf);
    const size_t bsz = HV ? 16 : maxe * 2;   // the frames path uses only images
    cl_mem A = buf(bsz), Bf = buf(bsz), T1 = buf(bsz), T2 = buf(bsz), Sk = buf(bsz);

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
        if (!HV) {   // the frames path (decodeHV) makes its own, strip-sized images
            IA = himg(maxTW, maxTH, nullptr); IB = himg(maxTW, maxTH, nullptr); IT1 = himg(maxTW, maxTH, nullptr);
            IT2 = himg(maxTW, maxTH, nullptr); ISk = himg(maxTW, maxTH, nullptr);
            printf("picture memory: activations %zux%zu texels, OC %d, PX %d\n", maxTW, maxTH, TOC, TPX);
        }
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
    // ---- TAEHV (taew2_1): frames, and column strips when a layer would be wider than the chip's images ----
    auto decodeHV = [&]() {
        if (!IMG) { fprintf(stderr, "this decoder needs the picture-memory path (unset TAEBUF)\n"); exit(1); }
        size_t maxIW = 0; CK(clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof maxIW, &maxIW, nullptr));
        if (getenv("TAEMAXW")) maxIW = (size_t)atol(getenv("TAEMAXW"));   // test: force strips
        // widest texel row per latent column, over all layers
        double perCol = (double)C0 / 4; int lastMem = -1;
        { double sc = 1; for (size_t i = 0; i < ops.size(); i++) { auto &o = ops[i];
            if ((o.kind == "conv" || o.kind == "tgrow") && o.up > 0) sc *= 2;
            perCol = std::max(perCol, sc * std::max(o.ci, o.co) / 4.0);
            if (o.kind == "mem") lastMem = (int)i; } }
        const int OV = 16;                                     // latent columns of overlap on each inner side
        int tw = std::min(wd0, (int)(maxIW / perCol));
        if (tw < wd0) tw = std::max(8, tw - 2 * OV);           // room for the overlap
        int ups = 1, grow = 1;   // spatial growth before the fold, frames per latent frame
        for (auto &o : ops) { if ((o.kind == "conv" || o.kind == "tgrow") && o.up > 0) ups *= 2; if (o.kind == "tgrow") grow *= o.k; }
        const int scale = ups * shuf;   // taew2_1: 8 pixels per latent column; Wan 2.2's: 8 x the 2 x 2 fold
        const int OH = h * scale, OW = wd0 * scale;
        const int trim = grow - 1, NF = allFrames ? NT * grow - trim : 1;   // frames out (ComfyUI drops the first trim)
        std::vector<float> out((size_t)NF * OH * OW * keep);
        // image pool: frames come and go
        const size_t PW = (size_t)std::min<double>(maxIW, perCol * (tw + 2 * OV) + 4);
        size_t PH = 0; { double sc = 1; for (auto &o : ops) if ((o.kind == "conv" || o.kind == "tgrow") && o.up > 0) sc *= 2; PH = (size_t)(h * sc); }
        std::vector<cl_mem> pool;
        auto get = [&]() { if (pool.empty()) return himg(PW, PH, nullptr); cl_mem m = pool.back(); pool.pop_back(); return m; };
        auto put = [&](cl_mem m) { pool.push_back(m); };
        cl_mem T1 = get(), T2 = get(), T3 = get(), Skp = get();
        for (int x0 = 0; x0 < wd0; x0 += tw) {
            const int a = std::max(0, x0 - OV), b = std::min(wd0, x0 + tw + OV), Wt = b - a;
            std::vector<cl_mem> fr;
            for (int t = 0; t < NT; t++) {   // latent frame t of the clip (one for a picture)
                std::vector<h16> x16((size_t)h * Wt * C0);
                const float *lt = lat.data() + (size_t)t * h * wd0 * C0;
                for (int y = 0; y < h; y++) for (int xx = 0; xx < Wt; xx++) for (int c = 0; c < C0; c++)
                    x16[((size_t)y * Wt + xx) * C0 + c] = (h16)lt[((size_t)y * wd0 + a + xx) * C0 + c];
                fr.push_back(get());
                size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)Wt * C0 / 4, (size_t)h, 1};
                CK(clEnqueueWriteImage(q, fr.back(), CL_TRUE, org, reg, 0, 0, x16.data(), 0, 0, 0));
            }
            int H = h, Wd = Wt;
            for (size_t i = 0; i < ops.size(); i++) {
                auto &o = ops[i];
                std::vector<cl_mem> nf;
                if (o.kind == "clamp") {
                    for (auto f : fr) { cl_mem n = get(); const int tws = Wd * C0 / 4;
                        arg(kClampI, 0, f); arg(kClampI, 1, n); arg(kClampI, 2, tws); arg(kClampI, 3, H);
                        run(kClampI, "clamp", tws, H, 1, 16, 4); nf.push_back(n); }
                } else if (o.kind == "conv") {
                    if (o.up > 0) { H *= 2; Wd *= 2; }
                    for (auto f : fr) { cl_mem n = get(); convI(f, o.name, H, Wd, o.ci, o.co, o.k, o.up, o.relu, nullptr, n); nf.push_back(n); }
                } else if (o.kind == "tgrow") {
                    if (o.up > 0) { H *= 2; Wd *= 2; }
                    for (auto f : fr) for (int j = 0; j < o.k; j++) {
                        cl_mem n = get(); convI(f, o.name + "." + std::to_string(j), H, Wd, o.ci, o.ci, 1, o.up, 0, nullptr, n); nf.push_back(n); }
                } else if (o.kind == "mem") {
                    for (size_t f = 0; f < fr.size(); f++) {
                        if (f == 0) convI(fr[f], o.name + ".c0x", H, Wd, o.ci, o.co, 3, 0, 1, nullptr, T1);
                        else { convI(fr[f], o.name + ".c0x", H, Wd, o.ci, o.co, 3, 0, 0, nullptr, T3);
                               convI(fr[f - 1], o.name + ".c0p", H, Wd, o.ci, o.co, 3, 0, 1, T3, T1); }
                        convI(T1, o.name + ".c1", H, Wd, o.co, o.co, 3, 0, 1, nullptr, T2);
                        cl_mem sk = fr[f];
                        if (o.skip) { convI(fr[f], o.name + ".skip", H, Wd, o.ci, o.co, 1, 0, 0, nullptr, Skp); sk = Skp; }
                        cl_mem n = get(); convI(T2, o.name + ".c2", H, Wd, o.co, o.co, 3, 0, 1, sk, n); nf.push_back(n);
                    }
                } else { fprintf(stderr, "op %s not supported in this decoder\n", o.kind.c_str()); exit(1); }
                for (auto f : fr) put(f);
                fr = nf;
                // past the last memory block only the picture (the last frame) is still needed
                if (!allFrames && (int)i >= lastMem && fr.size() > 1) { for (size_t f = 0; f + 1 < fr.size(); f++) put(fr[f]); fr = {fr.back()}; }
            }
            // this strip's frames (the last one, or all but the first trim) -> out, unfolding shuf x shuf
            // (PixelShuffle: channel c * r * r + dy * r + dx)
            const int s8 = H / h, keepFrom = (x0 - a) * s8, keepTo = keepFrom + std::min(tw, wd0 - x0) * s8;
            std::vector<h16> y((size_t)H * Wd * C);
            for (int k = 0; k < NF; k++) {
                cl_mem src = allFrames ? fr[trim + k] : fr.back();
                size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)Wd * C / 4, (size_t)H, 1};
                CK(clEnqueueReadImage(q, src, CL_TRUE, org, reg, 0, 0, y.data(), 0, 0, 0));
                float *of = out.data() + (size_t)k * OH * OW * keep;
                for (int yy = 0; yy < H; yy++) for (int xx = keepFrom; xx < keepTo; xx++)
                    for (int c = 0; c < keep; c++) for (int dy = 0; dy < shuf; dy++) for (int dx = 0; dx < shuf; dx++)
                        of[((size_t)(yy * shuf + dy) * OW + (a * s8 + xx) * shuf + dx) * keep + c] =
                            (float)y[((size_t)yy * Wd + xx) * C + c * shuf * shuf + dy * shuf + dx];
            }
            for (auto f : fr) put(f);
        }
        for (auto m : pool) clReleaseMemObject(m);
        clReleaseMemObject(T1); clReleaseMemObject(T2); clReleaseMemObject(T3); clReleaseMemObject(Skp);
        (void)scale;
        oh = OH; ow = OW;
        return out;
    };
    auto rgb = HV ? decodeHV() : decode();
    printf("picture %dx%d%s\n", ow, oh, allFrames ? (", " + std::to_string(rgb.size() / ((size_t)ow * oh * keep)) + " frames").c_str() : "");
    if (FILE *f = fopen((R + "/rgb.f32").c_str(), "rb")) { fclose(f); printf("  error vs reference %.5f\n", relerr(rgb, rdf(R + "/rgb.f32"))); }
    FILE *f = fopen((R + "/gpu_rgb.f32").c_str(), "wb"); fwrite(rgb.data(), 4, rgb.size(), f); fclose(f);
    tsum.clear();
    for (int l = 0; l < loops; l++) {
        clFinish(q); auto a = std::chrono::steady_clock::now();
        if (HV) decodeHV(); else decode();
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
