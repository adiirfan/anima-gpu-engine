// Universal graphics-chip engine: the building blocks every model is wired from (2026-11-18).
// Activations are images in "X layout" (texel (n, k8) = 8 halves of token n); the residual stream is a float4 buffer
// [D/4][N]. Token counts and widths are arguments, so one set of blocks serves any of the diffusion transformers on
// the list (Anima, Klein, Wan, Qwen-Image, MOSS, Fibo): they differ in sizes and wiring, and all use 128-wide heads.
#pragma once
#include "rt.h"
#include <set>

namespace ops {

static const int HD = 128;                  // channels per attention head (every supported model)
static int HEADS = 16, D = 2048;            // the model's heads and width (set by the model)
static int MAXN = 1024, MAXK = 8192;        // largest token count / multiply input the work buffers are sized for

static cl_kernel kG[4], kLn, kGemv, kPrepSelf, kPrepKV, kPq, kPqc, kKVdown, kPVsm, kFlash, kSgate, kGrp, kBiasGelu;
static int KVREP = 1;        // query heads per key / value head (set before init; Krea 4)
static float LNEPS = 1e-6f;  // the norm before 8-bit layers (Krea 1e-5)
static std::map<std::string, cl_kernel> kSoftBy;   // softmax kernels per (lanes, rows, texels per lane)

// options read once from the environment
static int PVSM = 1, PVT = 2, PVG = 4, PVWG = 128;  // softmax inside the blending pass, and its tile
static int FLASH = 0;                                // fused attention (slower so far; for experiments)
static int TODO = 0, TODO0 = 0, TODO1 = 1 << 30;     // half-resolution keys in blocks TODO0..TODO1
// set by the model before init: layers carry biases (8-bit layers get one through the centring path), GELU in its
// tanh form, q / k rms-normed across the whole width instead of per head
static int BIASES = 0, GELUT = 0, QKF = 0, RMSN = 0, SITU = 0;   // RMSN: RMS norms before 8-bit layers; SITU: MOSS's paired activation
static cl_kernel kRowRms;
static cl_mem rqB, rkB;

// ---- 8-bit multiplies (ConvRot W8A8): activations rotated + rounded per token, int8 weights with block scales ----
static int Q8 = 0, QGS = 64, QTv = 4, QTNv = 2, QOBv = 32, QWGv = 128;   // 2 x 32 beat 2 x 16 on a hot phone (MOSS)
static cl_kernel kQ[5], kRotX, kRotLn, kRotLnL, kRmsAdd;
static int LNNEXT = 0;   // 1: the next lnq / normLinear uses a layer norm even in an RMSN model (Fibo's output layer)
static cl_mem Xq, Sx;
// 8-bit path: weights int8 ("q8") or int4 ("q4", two per byte, unpacked to int8 as they are staged)
// (a streamed table is in W only while its block is in use, so the streamed list is asked too)
static bool isq8(const std::string &n) {
    auto it = W.find(n);
    if (it != W.end()) return it->second.kind == "q8" || it->second.kind == "q4";
    auto st = SW.find(n);
    return st != SW.end() && (st->second.kind == "q8" || st->second.kind == "q4");
}
static bool isq4(const std::string &n) {
    auto it = W.find(n);
    if (it != W.end()) return it->second.kind == "q4";
    auto st = SW.find(n);
    return st != SW.end() && st->second.kind == "q4";
}
static cl_kernel kQ4[5];
// A16 (chips without the byte dot product, auto on old phone drivers; QA16=0/1 overrides): each 8-bit layer is turned
// into fp16 weights just before it runs (deq8, one scratch image per shape) and the tuned fp16 multiply does the work;
// inputs are rotated but not rounded or centred (rotq HOUT). Models with real biases or other epilogues keep 8-bit maths.
static int A16 = 0;
static cl_kernel kDeq, kDeq4, kSwiglu16;
static std::map<std::pair<int, int>, cl_mem> deqImg;

// ---- mean centring for 8-bit layers (CEN=<file>): round (x - mu), add b = W.mu back; mu and b per noise level,
// blended between the two nearest calibrated levels. The model names the layers and their sizes in file order.
struct CenLayer { std::string name; int K, M; };
static std::vector<float> cenSig, cenTab;
static size_t cenStep = 0;
static cl_mem cenBuf;
static std::map<std::string, std::pair<cl_mem, cl_mem>> cen;  // layer -> (mu, b)
static void cenLoad(const char *path, const std::vector<CenLayer> &layers, int blocks) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("centring table"); exit(1); }
    int hdr[2];
    if (fread(hdr, 4, 2, f) != 2 || hdr[1] != blocks) { fprintf(stderr, "bad centring table\n"); exit(1); }
    cenStep = 0;
    for (auto &l : layers) cenStep += (size_t)l.K + l.M;
    cenSig.resize(hdr[0]); cenTab.resize(cenStep * hdr[0]);
    if (fread(cenSig.data(), 4, hdr[0], f) != (size_t)hdr[0] || fread(cenTab.data(), 4, cenTab.size(), f) != cenTab.size()) {
        fprintf(stderr, "short centring table\n"); exit(1);
    }
    fclose(f);
    cenBuf = buf(cenStep * 4);
    size_t o = 0;
    for (auto &l : layers) {
        cl_mem mu = sub(cenBuf, o * 4, (size_t)l.K * 4); o += l.K;
        cl_mem b = sub(cenBuf, o * 4, (size_t)l.M * 4); o += l.M;
        if (isq8(l.name)) cen[l.name] = {mu, b};   // layers kept in fp16 need none
    }
    printf("centring: %d noise levels (%.3f .. %.3f)\n", hdr[0], cenSig.front(), cenSig.back());
}
static void cenSet(float sigma) {
    if (cen.empty()) return;
    const int S = (int)cenSig.size();
    int a = 0; float t = 0;
    if (sigma <= cenSig[S - 1]) a = S - 1;
    else if (sigma < cenSig[0]) {
        while (a + 1 < S && cenSig[a + 1] > sigma) a++;
        if (a + 1 < S) t = (cenSig[a] - sigma) / (cenSig[a] - cenSig[a + 1]);
    }
    std::vector<float> v(cenTab.begin() + a * cenStep, cenTab.begin() + (a + 1) * cenStep);
    if (t > 0) for (size_t i = 0; i < cenStep; i++) v[i] += t * (cenTab[(a + 1) * cenStep + i] - v[i]);
    clEnqueueWriteBuffer(q, cenBuf, CL_TRUE, 0, cenStep * 4, v.data(), 0, nullptr, nullptr);
}

// ---- weights: fp16 file plus an optional 8-bit file that replaces the big layers ----
// Q8KEEP: layers that stay fp16, as "kind" (every block), "block" or "block.kind" ("l2", "0,27", "5.qkv"); a kind
// matches the part of the name after the block number. Kept layers simply do not get their 8-bit version loaded.
static std::set<std::string> KEEP;
static bool kept(const std::string &name) {
    if (KEEP.empty()) return false;
    const auto dot = name.find('.');
    const std::string b = name.substr(0, dot), k = dot == std::string::npos ? name : name.substr(dot + 1);
    std::string base = k; if (base.size() > 2 && base.substr(base.size() - 2) == ".s") base = base.substr(0, base.size() - 2);
    return KEEP.count(base) || KEEP.count(b) || KEEP.count(b + "." + base);
}
static std::string loadModel(const std::string &prefix) {
    if (getenv("Q8KEEP")) {
        std::string k = getenv("Q8KEEP"), t;
        for (char c : k + ",") { if (c == ',') { if (!t.empty()) KEEP.insert(t); t.clear(); } else t += c; }
    }
    if (prefix.size() > 5 && prefix.substr(prefix.size() - 5) == ".gguf") {   // one file: 8-bit and fp16 layers together
        const std::string m = loadGGUF(prefix);
        for (auto &kv : W) if (kv.second.kind == "q8" || kv.second.kind == "q4") { Q8 = 1; break; }
        for (auto &kv : SW) if (kv.second.kind == "q8" || kv.second.kind == "q4") { Q8 = 1; break; }
        return m;
    }
    if (getenv("Q8")) {   // 8-bit layers first, so the fp16 copies of those layers are never loaded
        Q8 = 1;
        loadWeights(getenv("Q8"), [](const std::string &n) { return kept(n); });
    }
    return loadWeights(prefix, [](const std::string &n) { return W.count(n) != 0; });
}

// ---- kernels and work buffers ----
static cl_kernel softmaxFor(int Lk) {
    int SL = 16; while (SL * 8 * 8 < Lk) SL *= 2;
    int SR = SL >= 256 ? 1 : 256 / SL; if (SR > 8) SR = 8;
    int MT = (Lk / 8 + SL - 1) / SL; if (MT < 1) MT = 1;
    char o[160]; snprintf(o, sizeof o, "-DDO_Q=0 -DDO_K=0 -DDO_V=0 -DROPE=1 -DSL=%d -DSR=%d -DMAXT=%d", SL, SR, MT);
    auto it = kSoftBy.find(o);
    if (it != kSoftBy.end()) return it->second;
    return kSoftBy[o] = kern(build(KSRC, o), "softmax2");
}
static void softmaxRun(cl_mem S, cl_mem P, int N, int Lk, int heads, float bias, int kvalid = -1) {
    if (kvalid < 0) kvalid = Lk;
    int SL = 16; while (SL * 8 * 8 < Lk) SL *= 2;
    int SR = SL >= 256 ? 1 : 256 / SL; if (SR > 8) SR = 8;
    cl_kernel k = softmaxFor(Lk);
    arg(k, 0, S); arg(k, 1, P); arg(k, 2, N); arg(k, 3, Lk); arg(k, 4, bias); arg(k, 5, kvalid);
    run3(k, "softmax", SL, N, heads, SL, SR, 1);
}

static cl_mem Xa, Qo, Ko, Vo, Ko2, Vo2, S, Pr;
static int HC = 16;   // self-attention heads per pass, so the score table stays under ~64 MB
static void init(int heads, int width, int maxTokens, int maxK) {
    HEADS = heads; D = width; MAXN = maxTokens; MAXK = maxK;
    char o[256];
    if (!getenv("LCFG") && !getenv("CFG") && getenv("TUNEFILE")) tune(KSRC, getenv("TUNEFILE"));
    for (int e = 0; e < 4; e++) {
        snprintf(o, sizeof o, "-DTN=%d -DTM=%d -DEPI=%d -DTNI=%d -DG=%d -DWG=%d -DKC=%d", TN, TM, e, TNI, GG, WGS, KCS);
        kG[e] = kern(build(KSRC, o), LOCALGEMM ? "gemml" : "gemm");
    }
    const std::string qkf = QKF ? " -DQKFULL=1" : "";
    cl_program pb = build(KSRC, "-DDO_Q=1 -DDO_K=1 -DDO_V=1 -DROPE=1" + qkf);
    kLn = kern(pb, "lnmod2"); kGemv = kern(pb, "gemv"); kRmsAdd = kern(pb, "rmsadd"); kPq = kern(pb, "prepq"); kRowRms = kern(pb, "rowrms");
    kPrepSelf = kern(build(KSRC, "-DDO_Q=0 -DDO_K=1 -DDO_V=1 -DROPE=1" + qkf + (KVREP > 1 ? " -DKVR=" + std::to_string(KVREP) : "")), "attnprep");
    kSgate = kern(pb, "sgate"); kGrp = kern(pb, "grpattn"); kBiasGelu = kern(pb, "biasgelu"); kSwiglu16 = kern(pb, "swiglu16");
    if (QKF) { rqB = buf((size_t)maxTokens * 4); rkB = buf((size_t)maxTokens * 4); }
    {
        cl_program pc = build(KSRC, "-DDO_Q=0 -DDO_K=1 -DDO_V=1 -DROPE=0 -DMAXT=2" + qkf);
        kPrepKV = kern(pc, "attnprep"); kPqc = kern(pc, "prepq");
    }
    PVSM = getenv("PVSM") ? atoi(getenv("PVSM")) : 1;   // default since 2026-11-18: the faster attention at 1024
    FLASH = getenv("FLASH") ? atoi(getenv("FLASH")) : 0;
    TODO = getenv("TODO") ? atoi(getenv("TODO")) : 0;
    if (getenv("TODOB")) sscanf(getenv("TODOB"), "%d,%d", &TODO0, &TODO1);
    if (TODO) kKVdown = kern(build(KSRC, std::string("-DDO_Q=0 -DDO_K=0 -DDO_V=0 -DROPE=1 -DKAVG=") + (getenv("KAVG") ? getenv("KAVG") : "0")), "kvdown");
    if (FLASH) kFlash = kern(build(kslice("// ======== fused attention", "// ======== softmax inside"),
                                   getenv("FLASHOPT") ? getenv("FLASHOPT") : ""), "flashattn");
    {
        int a = TNI, g2 = GG, wg = WGS, kc = KCS;
        if (getenv("PVCFG")) sscanf(getenv("PVCFG"), "%d,%d,%d,%d", &a, &g2, &wg, &kc);
        PVT = a; PVG = g2; PVWG = wg;
        snprintf(o, sizeof o, "-DEPI=0 -DTNI=%d -DG=%d -DWG=%d -DKC=%d", a, g2, wg, kc);
        kPVsm = kern(build(kslice("// ======== softmax inside"), o), "gemmlsm");
    }
    if (Q8) {
        // the scale block size is read from the file (an 8-bit table of K inputs with K / GS scales per output), so
        // 64- and 256-input files both just work; QGS overrides, the model's own default is the fallback
        {
            int fileGS = 0;
            auto look = [&](const std::string &n, const std::string &kind, int K) {
                if (fileGS || (kind != "q8" && kind != "q4")) return;
                auto s = W.find(n + ".s"); auto ss = SW.find(n + ".s");
                const int nb = s != W.end() ? s->second.K : ss != SW.end() ? ss->second.K : 0;
                if (nb > 0 && K % nb == 0) fileGS = K / nb;
            };
            for (auto &kv : W) look(kv.first, kv.second.kind, kv.second.K);
            for (auto &kv : SW) look(kv.first, kv.second.kind, kv.second.K);
            if (fileGS && fileGS != QGS && !getenv("QGS")) { printf("8-bit scale blocks of %d inputs (from the file)\n", fileGS); QGS = fileGS; }
        }
        if (getenv("QGS")) QGS = atoi(getenv("QGS"));
        if (getenv("QCFG")) sscanf(getenv("QCFG"), "%d,%d,%d", &QTNv, &QOBv, &QWGv);
        if (getenv("QT")) QTv = atoi(getenv("QT"));
        const int rot = getenv("QROT") ? atoi(getenv("QROT")) : 1;
        int cenv = getenv("CEN") || BIASES ? 1 : 0;
        A16 = (getenv("QA16") ? atoi(getenv("QA16")) : KEXTRA.find("ACC32") != std::string::npos) && !BIASES && !GELUT && !SITU;
        const char *a16o = A16 ? " -DHOUT=1" : "";
        if (A16) cenv = 0;   // fp16 inputs need no centring (W(x - mu) + W.mu is W x)
        const std::string qsrc = kslice("// ======== 8-bit path", "// ======== fused attention");
        if (A16) {
            snprintf(o, sizeof o, "-DGS=%d", QGS); kDeq = kern(build(qsrc, o), "deq8");
            snprintf(o, sizeof o, "-DGS=%d -DW4=1", QGS); kDeq4 = kern(build(qsrc, o), "deq8");
        }
        for (int e : {0, 1, 2, 3, 4}) {
            // the multiply stages QKC inputs at a time and needs whole scale blocks in each stage
            snprintf(o, sizeof o, "-DEPI=%d -DGS=%d -DQTN=%d -DQOB=%d -DQWG=%d -DROT=%d -DSRC=0 -DCEN=%d -DGELU_TANH=%d -DSITU=%d -DQKC=%d", e, QGS, QTNv, QOBv, QWGv, rot, cenv, GELUT, SITU, std::max(128, QGS));
            kQ[e] = kern(build(qsrc, o), "gemmq");
            bool any4 = false; for (auto &kv : W) if (kv.second.kind == "q4") { any4 = true; break; }
            for (auto &kv : SW) if (kv.second.kind == "q4") { any4 = true; break; }
            if (any4) kQ4[e] = kern(build(qsrc, std::string(o) + " -DW4=1"), "gemmq");
        }
        snprintf(o, sizeof o, "-DEPI=0 -DGS=%d -DROT=%d -DSRC=0 -DQT=%d -DCEN=%d%s", QGS, rot, QTv, cenv, a16o);
        kRotX = kern(build(qsrc, o), "rotq");
        snprintf(o, sizeof o, "-DEPI=0 -DGS=%d -DROT=%d -DSRC=1 -DQT=%d -DCEN=%d -DRMSN=%d%s", QGS, rot, QTv, cenv, RMSN, a16o);
        kRotLn = kern(build(qsrc, o), "rotq");
        snprintf(o, sizeof o, "-DEPI=0 -DGS=%d -DROT=%d -DSRC=1 -DQT=%d -DCEN=%d -DRMSN=0%s", QGS, rot, QTv, cenv, a16o);
        kRotLnL = RMSN ? kern(build(qsrc, o), "rotq") : kRotLn;
        Xq = img(MAXN, MAXK / (A16 ? 8 : 16)); Sx = buf((size_t)MAXN * (MAXK / QGS) * 4);
        printf("8-bit multiplies: GS %d, rotation %d, cfg %d,%d,%d%s\n", QGS, rot, QTNv, QOBv, QWGv,
               A16 ? ", turned into 16-bit per layer (no byte dot product)" : "");
    }
    // work buffers for the largest token count
    const int N = MAXN;
    Xa = img(N, MAXK / 8);
    Qo = img(N, D / 8);
    Ko = img(N / 4, HEADS * 64); Vo = img(HEADS * 32, N / 2);
    if (TODO) { Ko2 = img(N / 16, HEADS * 64); Vo2 = img(HEADS * 32, N / 8); }
    // heads per pass: the largest divisor of HEADS whose score table fits ATTNMB (default 64 MB)
    const size_t cap = (size_t)(getenv("ATTNMB") ? atoi(getenv("ATTNMB")) : 64) << 20;
    HC = 1;
    for (int h = HEADS; h >= 1; h--) if (HEADS % h == 0 && (size_t)h * N * N * 2 <= cap) { HC = h; break; }
    S = img(N, HC * (N / 8));
    if (!PVSM) Pr = img(N, HC * (N / 8));
}

// BIASES: every 8-bit layer gets a centring entry whose input average is zero and whose output offset is the layer's
// bias "<name>.b" (or zero). Call after init and loadModel.
static void biasInit() {
    if (!BIASES) return;
    size_t maxM = 0;
    for (auto &kv : W) if (kv.second.kind == "q8" || kv.second.kind == "q4") maxM = std::max(maxM, (size_t)kv.second.M);
    for (auto &kv : SW) if (kv.second.kind == "q8" || kv.second.kind == "q4") maxM = std::max(maxM, (size_t)kv.second.M);
    std::vector<float> z(std::max((size_t)MAXK, maxM), 0.0f);
    cl_mem zb = buf(z.size() * 4, z.data());
    int nb = 0;
    for (auto &kv : W) if (kv.second.kind == "q8" || kv.second.kind == "q4") {
        auto b = W.find(kv.first + ".b");
        cen[kv.first] = {zb, b != W.end() ? b->second.m : zb};
        nb += b != W.end();
    }
    for (auto &kv : SW) if (kv.second.kind == "q8" || kv.second.kind == "q4") {   // streamed layers (biases stay loaded)
        auto b = W.find(kv.first + ".b");
        cen[kv.first] = {zb, b != W.end() ? b->second.m : zb};
        nb += b != W.end();
    }
    printf("biases on %d of %zu 8-bit layers\n", nb, cen.size());
}
// whole-width 1/rms of tokens [0, n) over C8 texels from column c0 of an X-layout image (QKF models)
static void rowRms(cl_mem in, int n, int c0, int C8, cl_mem out) {
    arg(kRowRms, 0, in); arg(kRowRms, 1, n); arg(kRowRms, 2, c0); arg(kRowRms, 3, C8); arg(kRowRms, 4, out);
    run(kRowRms, "rms", 64, n, 1, 64, 1);
}

// ---- multiplies ----
// tokens [n0, n) of a stream whose buffers hold N rows (N only matters for the residual layout, given by the kernel)
static void gemm(int epi, const char *tag, cl_mem X, cl_mem Wm, int n, int M, int K, cl_mem out, cl_mem gate = nullptr,
                 int batch = 1, int xby = 0, int wbx = 0, int wby = 0, int yby = 0, int xb0 = 0, int wb0 = 0, int yb0 = 0,
                 int n0 = 0, int Nall = 0) {
    cl_kernel k = kG[epi];
    const int Nk = Nall ? Nall : n;   // the kernel's N: row stride of the residual stream
    arg(k, 0, X); arg(k, 1, Wm); arg(k, 2, Nk); arg(k, 3, M); arg(k, 4, K);
    arg(k, 5, xby); arg(k, 6, wbx); arg(k, 7, wby); arg(k, 8, yby); arg(k, 9, xb0); arg(k, 10, wb0); arg(k, 11, yb0);
    arg(k, 12, out);
    if (epi >= 2) arg(k, 13, gate ? gate : out);
    arg(k, epi >= 2 ? 14 : 13, n);
    // token batches whose input fits the cache: otherwise every block of outputs re-reads the whole input
    const int unit = LOCALGEMM ? TNI * WGS : TN * L1;
    int chunk = (int)(CACHE / ((size_t)K * 2)) / unit * unit;
    if (chunk < unit) chunk = unit;
    for (int c0 = n0; c0 < n; c0 += chunk) {
        const int cn = std::min(chunk, n - c0);
        if (LOCALGEMM) run(k, tag, (cn + TNI - 1) / TNI, M / 4 / GG, batch, WGS, 1, c0 / TNI, 0);
        else run(k, tag, M / 4 / TM, (cn + TN - 1) / TN, batch, L0, L1, 0, c0 / TN);
    }
}
static void gemmq(int epi, const char *tag, const std::string &wn, int N, int M, int K, cl_mem out, cl_mem gate,
                  int n0 = 0, int n1 = -1) {
    if (n1 < 0) n1 = N;
    cl_kernel k = isq4(wn) ? kQ4[epi] : kQ[epi];
    arg(k, 0, Xq); arg(k, 1, w(wn)); arg(k, 2, N); arg(k, 3, M); arg(k, 4, K); arg(k, 5, Sx); arg(k, 6, w(wn + ".s"));
    arg(k, 7, out);
    if (epi == 2 || epi == 3) arg(k, 8, gate ? gate : out);
    auto c = cen.find(wn);
    int ai = (epi == 2 || epi == 3) ? 9 : 8;
    if (c != cen.end()) arg(k, ai++, c->second.second);
    arg(k, ai, n1);
    const int unit = QTNv * QWGv;
    int chunk = (int)(CACHE / (size_t)K) / unit * unit;
    if (chunk < unit) chunk = unit;
    for (int c0 = n0; c0 < n1; c0 += chunk) {
        const int cn = std::min(chunk, n1 - c0);
        run(k, tag, (cn + QTNv - 1) / QTNv, M / QOBv, 1, QWGv, 1, c0 / QTNv, 0);
    }
}
static void rotq(const std::string &wn, cl_mem in, int N, int K, int n0 = 0, int n1 = -1) {
    if (n1 < 0) n1 = N;
    arg(kRotX, 0, in); arg(kRotX, 1, N); arg(kRotX, 2, K); arg(kRotX, 3, Xq); arg(kRotX, 4, Sx);
    auto c = cen.find(wn); if (c != cen.end() && !A16) arg(kRotX, 5, c->second.first);
    run(kRotX, "rotq", 64, n1 - n0, 1, 64, QTv, 0, n0);
}
static void lnmod(cl_mem res, cl_mem sh, cl_mem sc, cl_mem Y, int N, int Dm, int n0 = 0, int n1 = -1) {
    if (n1 < 0) n1 = N;
    float eps = 1e-6f;
    arg(kLn, 0, res); arg(kLn, 1, sh); arg(kLn, 2, sc); arg(kLn, 3, Y); arg(kLn, 4, N); arg(kLn, 5, Dm); arg(kLn, 6, eps);
    run(kLn, "lnmod", n1 - n0, 16, 1, 16, 16, n0, 0);
}
static void lnq(const std::string &wn, cl_mem res, cl_mem sh, cl_mem sc, int N, int Dm, int n0 = 0, int n1 = -1) {
    if (n1 < 0) n1 = N;
    float eps = LNEPS;
    cl_kernel kr = LNNEXT ? kRotLnL : kRotLn;
    arg(kr, 0, res); arg(kr, 1, sh); arg(kr, 2, sc); arg(kr, 3, eps); arg(kr, 4, N); arg(kr, 5, Dm);
    arg(kr, 6, Xq); arg(kr, 7, Sx);
    auto c = cen.find(wn); if (c != cen.end() && !A16) arg(kr, 8, c->second.first);
    run(kr, "lnq", 64, n1 - n0, 1, 64, QTv, 0, n0);
}
// A16: the fp16 W-layout copy of 8-bit layer wn (M x K), made now in the scratch image for its shape
static cl_mem deq(const std::string &wn, int M, int K) {
    cl_mem &d = deqImg[{M, K}];
    if (!d) d = img(M / 4, K / 2);
    cl_kernel k = isq4(wn) ? kDeq4 : kDeq;
    arg(k, 0, w(wn)); arg(k, 1, w(wn + ".s")); arg(k, 2, M); arg(k, 3, K); arg(k, 4, d);
    run(k, "deq8", M / 4, K / 16, 1, 16, 4);
    return d;
}
static bool useA16(const std::string &wn, int epi) { (void)wn; return A16 && epi <= 4; }
// A16 multiply of the rounded input Xq by layer wn; SwiGLU (epi 4) as a plain multiply into a scratch image + swiglu16
static void a16(int epi, const char *tag, const std::string &wn, int N, int M, int K, cl_mem out, cl_mem gate, int n0, int n1) {
    cl_mem Wd = deq(wn, M, K);
    if (epi != 4) { gemm(epi, tag, Xq, Wd, n1, M, K, out, gate, 1, 0, 0, 0, 0, 0, 0, 0, n0, N); return; }
    static cl_mem tmp = nullptr; static int tmpH = 0;
    if (tmpH < M / 8) { if (tmp) clReleaseMemObject(tmp); tmp = img(MAXN, M / 8); tmpH = M / 8; }
    gemm(0, tag, Xq, Wd, n1, M, K, tmp, nullptr, 1, 0, 0, 0, 0, 0, 0, 0, n0, N);
    arg(kSwiglu16, 0, tmp); arg(kSwiglu16, 1, n1); arg(kSwiglu16, 2, out);
    run(kSwiglu16, "swiglu16", n1 - n0, M / 16, 1, 16, 4, n0, 0);
}
// out = X W^T (epi 0 plain, 1 gelu, 2 res += r*gate, 3 res = r); 8-bit when the layer has 8-bit weights
// tokens [n0, n1) of N (default all); epi 4 = SwiGLU (8-bit layers only so far)
static void linear(int epi, const char *tag, cl_mem X, const std::string &wn, int N, int M, int K, cl_mem out, cl_mem gate = nullptr,
                   int n0 = 0, int n1 = -1) {
    if (n1 < 0) n1 = N;
    if (isq8(wn) && useA16(wn, epi)) {
        rotq(wn, X, N, K, n0, n1);
        a16(epi, tag, wn, N, M, K, out, gate, n0, n1);
    } else if (isq8(wn)) { rotq(wn, X, N, K, n0, n1); gemmq(epi, tag, wn, N, M, K, out, gate, n0, n1); }
    else {
        if (epi == 4) { fprintf(stderr, "SwiGLU needs 8-bit weights (%s)\n", wn.c_str()); exit(1); }
        gemm(epi, tag, X, w(wn), n1, M, K, out, gate, 1, 0, 0, 0, 0, 0, 0, 0, n0, N);
    }
}
// out = (layernorm(res) * (1 + sc) + sh) W^T: the norm is folded into the rounding on the 8-bit path
static void normLinear(int epi, const char *tag, cl_mem res, cl_mem sh, cl_mem sc, const std::string &wn, int N, int M, int K,
                       cl_mem out, cl_mem gate = nullptr, int n0 = 0, int n1 = -1) {
    if (n1 < 0) n1 = N;
    if (isq8(wn) && useA16(wn, epi)) {
        lnq(wn, res, sh, sc, N, K, n0, n1);
        a16(epi, tag, wn, N, M, K, out, gate, n0, n1);
    } else if (isq8(wn)) { lnq(wn, res, sh, sc, N, K, n0, n1); gemmq(epi, tag, wn, N, M, K, out, gate, n0, n1); }
    else {
        if (epi == 4) { fprintf(stderr, "SwiGLU needs 8-bit weights (%s)\n", wn.c_str()); exit(1); }
        lnmod(res, sh, sc, Xa, N, K, n0, n1); gemm(epi, tag, Xa, w(wn), n1, M, K, out, gate, 1, 0, 0, 0, 0, 0, 0, 0, n0, N);
    }
}
static void gemv(const char *tag, cl_mem Wm, cl_mem x, cl_mem add, cl_mem y, int M, int K, int silu) {
    arg(kGemv, 0, Wm); arg(kGemv, 1, x);
    if (add) arg(kGemv, 2, add); else argnull(kGemv, 2);
    arg(kGemv, 3, y); arg(kGemv, 4, M); arg(kGemv, 5, K); arg(kGemv, 6, silu);
    run(kGemv, tag, M, 1, 1, 64, 1);
}

// Y = O * sigmoid(G[gc..]) for tokens [0, N) (Krea's gated attention output; gc = gate's first texel column in G)
static void sgate(cl_mem O, cl_mem G, int gc, int N, int Dm, cl_mem Y) {
    arg(kSgate, 0, O); arg(kSgate, 1, G); arg(kSgate, 2, gc); arg(kSgate, 3, N); arg(kSgate, 4, Y);
    run(kSgate, "sgate", N, Dm / 8, 1, 16, 4);
}
// attention inside groups of G rows of QKV, rows [0, n1), all HEADS heads, no rope (Krea's text fusion) into O
static void grpAttn(cl_mem QKV, int n1, int G, int qr, int kr, int vr, cl_mem qn, cl_mem kn, cl_mem O) {
    cl_kernel k = kGrp;
    arg(k, 0, QKV); arg(k, 1, n1); arg(k, 2, G); arg(k, 3, qr); arg(k, 4, kr); arg(k, 5, vr); arg(k, 6, qn); arg(k, 7, kn); arg(k, 8, O);
    run(k, "grpattn", n1, HEADS, 1, 16, 1);
}
static void biasGelu(cl_mem Y, cl_mem b, int N, int C, cl_mem Z) {
    arg(kBiasGelu, 0, Y); arg(kBiasGelu, 1, b); arg(kBiasGelu, 2, N); arg(kBiasGelu, 3, Z);
    run(kBiasGelu, "biasgelu", N, C / 8, 1, 16, 4);
}
// res rows [n0, n1) += rms_norm(A) * w (A: X-layout output of a layer)
static void rmsAdd(cl_mem A, cl_mem res, cl_mem wv, int N, int Dm, int n0, int n1) {
    arg(kRmsAdd, 0, A); arg(kRmsAdd, 1, res); arg(kRmsAdd, 2, wv); arg(kRmsAdd, 3, N); arg(kRmsAdd, 4, Dm);
    run(kRmsAdd, "rmsadd", 64, n1 - n0, 1, 64, 1, 0, n0);
}
// ---- streaming (STREAM=1): blocks' tables loaded just in time, the next one on a helper thread ----
// Models bigger than memory: a block's big tables live in the file and come in one block ahead, on a helper thread.
// Two fixed slots of chip memory are made once and refilled in turn: making and freeing ~130 MB of chip memory per
// block stalled the chip itself (MOSS: steps were much slower with fresh memory per block than with none).
struct Streamer {
    struct Slot { std::map<std::string, Wt> t; int block = -1; };   // table name without "b<n>" -> chip object
    Slot slot[2];
    int cs = 0;                       // the slot the current block is in
    std::future<void> fut;            // filling slot[1 - cs] with block `next`
    int next = -1, cur = -1;
    double waitMs = 0;                // time the host spent waiting for the next block's weights
    cl_command_queue lq = nullptr;    // the helper thread's own queue
    static std::string tail(const std::string &n) { return n.substr(n.find('.')); }
    // fill slot sl with block b; `after` = the chip's work that still reads the slot's old contents
    void fill(int sl, int b, cl_event after) {
        if (after) { clWaitForEvents(1, &after); clReleaseEvent(after); }
        if (!prefetchBlock(b + 1)) prefetchBlock(0);
        const std::string pre = "b" + std::to_string(b) + ".";
        std::vector<std::future<void>> jobs;
        for (auto it = SW.lower_bound(pre); it != SW.end() && it->first.compare(0, pre.size(), pre) == 0; ++it) {
            Wt &w = slot[sl].t[tail(it->first)];
            const SInfo *s = &it->second;
            if (w.m && (w.kind != s->kind || w.M != s->M || w.K != s->K)) { clReleaseMemObject(w.m); w.m = nullptr; }
            jobs.push_back(std::async(std::launch::async, [this, &w, s] {
                if (w.m) { refill(lq, *s, w.m); return; }
                w = Wt{s->kind, s->M, s->K, nullptr};
                if (s->kind == "pk") w.m = img(s->M / 4, s->K / 2, s->src);   // STREAM_PK=1: fp16 blocks too
                else if (s->kind == "q8") w.m = q8img(s->M, s->K, s->src);
                else if (s->kind == "q4") w.m = q8img(s->M, s->K, s->src, 32);
                else if (s->kind == "qsk") { auto v = qskScales(s->src, s->sub, s->M, s->K); w.m = buf(v.size() * 4, v.data()); }
                else w.m = buf(s->bytes, s->src);
            }));
        }
        for (auto &j : jobs) j.get();
        slot[sl].block = b;
    }
    cl_event marker() { cl_event e; CK(clEnqueueMarkerWithWaitList(q, 0, 0, &e)); return e; }
    void use(int b, int nb) {
        if (SW.empty()) return;
        if (b == cur) return;
        QUIET_FOLD = true;
        if (!lq) { cl_int e; lq = clCreateCommandQueueWithProperties(ctx, dev, 0, &e); CK(e); }
        if (getenv("SAMEBLOCK")) {   // timing test only: every block reuses block 0's tables (wrong picture, no loading)
            if (slot[0].block < 0) fill(0, 0, nullptr);
            for (auto &kv : slot[0].t) W["b" + std::to_string(b) + kv.first] = kv.second;
            cur = b; return;
        }
        const auto tw0 = std::chrono::steady_clock::now();
        if (fut.valid()) fut.get();
        int sl = 1 - cs;
        if (slot[sl].block != b) {    // not the one prefetched (first call, or blocks used out of order)
            if (slot[cs].block == b) sl = cs;
            else fill(sl, b, cur >= 0 ? marker() : nullptr);
        }
        waitMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw0).count();
        if (cur >= 0) {               // the old block's names go; its chip objects stay in their slot
            const std::string pre = "b" + std::to_string(cur) + ".";
            for (auto it = W.lower_bound(pre); it != W.end() && it->first.compare(0, pre.size(), pre) == 0;)
                it = SW.count(it->first) ? W.erase(it) : std::next(it);
        }
        const std::string nm = "b" + std::to_string(b);
        for (auto &kv : slot[sl].t) if (SW.count(nm + kv.first)) W[nm + kv.first] = kv.second;
        cs = sl; cur = b; next = (b + 1) % nb;
        // the other slot is still read by the previous block's queued work: refill it once that is done
        cl_event after = marker();
        const int other = 1 - cs, nb2 = next;
        fut = std::async(std::launch::async, [this, other, nb2, after] { fill(other, nb2, after); });
    }
};
static Streamer streamer;

// ---- attention ----
// Self-attention, in two parts so two-stream models can prepare each stream with its own norms:
// attnPrep: tokens [n0, n1) of QKV (q, k, v at texel columns qr, kr, vr): rms-norm with qn / kn, rope from the per-token
// tables P, Q (indexed by absolute token), into Qo / Ko / Vo. Ranges must start at a multiple of 4.
static void attnPrep(cl_mem QKV, int N, int n0, int n1, int qr, int kr, int vr, cl_mem qn, cl_mem kn, cl_mem P, cl_mem Q) {
    const float qscale = 1.0f / sqrtf((float)HD);
    cl_kernel k = kPrepSelf;
    if (QKF) {   // one stream only (n0 = 0)
        rowRms(QKV, n1, qr, D / 8, rqB); rowRms(QKV, n1, kr, D / 8, rkB);
        arg(k, 13, rqB); arg(k, 14, rkB);
    }
    arg(k, 0, QKV); arg(k, 1, N); arg(k, 2, qr); arg(k, 3, kr); arg(k, 4, vr);
    arg(k, 5, qn); arg(k, 6, kn); arg(k, 7, qscale); arg(k, 8, P); arg(k, 9, Q);
    arg(k, 10, Qo); arg(k, 11, Ko); arg(k, 12, Vo);
    run(k, "prep kv", (n1 - n0 + 3) / 4, HEADS, 1, 16, 1, n0 / 4, 0);
    cl_kernel p = kPq;
    if (QKF) arg(p, 8, rqB);
    arg(p, 0, QKV); arg(p, 1, N); arg(p, 2, qr); arg(p, 3, qn); arg(p, 4, qscale);
    arg(p, 5, P); arg(p, 6, Q); arg(p, 7, Qo);
    run(p, "prep q", n1 - n0, HEADS, 1, 64, 1, n0, 0);
}
// attnCore: softmax(q k^T / sqrt(128)) v over all N tokens into O; keys from lvalid on are padding (no weight).
// ds: the keys / values are the half-resolution ones in Ko2 / Vo2 (N/4 of them).
// LK: number of keys when it differs from the queries (picture rows attending to [picture, kept prompt] keys);
// causal: query n ignores keys after n.
static void attnCore(int N, cl_mem O, int lvalid = -1, bool ds = false, int LK = 0, int causal = 0) {
    const float zero = 0.0f;
    const int LKs = ds ? N / 4 : LK > 0 ? LK : N;
    if (lvalid < 0) lvalid = LKs;
    cl_mem Kuse = ds ? Ko2 : Ko, Vuse = ds ? Vo2 : Vo;
    int hc = HEADS;
    while (hc > 1 && (size_t)hc * N * LKs * 2 > (64u << 20)) hc /= 2;
    while (hc > 1 && (size_t)hc * LKs > (size_t)HC * MAXN) hc /= 2;   // the score table's width (sized at init)
    if (FLASH && !ds && !causal && lvalid == LKs) {
        arg(kFlash, 0, Qo); arg(kFlash, 1, Ko); arg(kFlash, 2, Vo); arg(kFlash, 3, N); arg(kFlash, 4, O);
        run(kFlash, "flash", (size_t)(N + 63) / 64 * 128, HEADS, 1, 128, 1);
    } else if (PVSM || lvalid < LKs || causal) {   // padding needs the masked blend
        for (int h0 = 0; h0 < HEADS; h0 += hc) {
            gemm(0, "gemm qk", Qo, Kuse, N, LKs, HD, S, nullptr, hc, 16, 0, 64, LKs / 8, h0, h0, 0);
            cl_kernel k = kPVsm; int M = HD, xby = LKs / 8, wbx = 32, wby = 0, yby = 16, z = 0;
            arg(k, 0, S); arg(k, 1, Vuse); arg(k, 2, N); arg(k, 3, M); arg(k, 4, LKs);
            arg(k, 5, xby); arg(k, 6, wbx); arg(k, 7, wby); arg(k, 8, yby); arg(k, 9, z); arg(k, 10, h0); arg(k, 11, h0);
            arg(k, 12, O); arg(k, 13, lvalid); arg(k, 14, causal);
            const int unit = PVT * PVWG;
            int chunk = (int)(CACHE / ((size_t)LKs * 2)) / unit * unit;
            if (chunk < unit) chunk = unit;
            for (int c0 = 0; c0 < N; c0 += chunk)
                run(k, "pv+softmax", (std::min(chunk, N - c0) + PVT - 1) / PVT, M / 4 / PVG, hc, PVWG, 1, c0 / PVT, 0);
        }
    } else {
        for (int h0 = 0; h0 < HEADS; h0 += hc) {
            gemm(0, "gemm qk", Qo, Kuse, N, LKs, HD, S, nullptr, hc, 16, 0, 64, LKs / 8, h0, h0, 0);
            softmaxRun(S, Pr, N, LKs, hc, zero);
            gemm(0, "gemm pv", Pr, Vuse, N, HD, LKs, O, nullptr, hc, LKs / 8, 32, 0, 16, 0, h0, h0);
        }
    }
}
// One-stream self-attention (Anima): gridW > 0 allows the half-resolution keys of TODO=1 (tokens on a gridW-wide
// grid); block decides whether TODO applies here.
static void selfAttn(cl_mem QKV, int N, int qr, int kr, int vr, cl_mem qn, cl_mem kn, cl_mem P, cl_mem Q, cl_mem O,
                     int gridW = 0, int block = 0) {
    const bool ds = TODO && gridW > 0 && block >= TODO0 && block <= TODO1;
    if (ds) {
        cl_kernel k = kKVdown; int n2 = N / 4;
        arg(k, 0, QKV); arg(k, 1, n2); arg(k, 2, gridW); arg(k, 3, kr); arg(k, 4, vr); arg(k, 5, kn);
        arg(k, 6, P); arg(k, 7, Q); arg(k, 8, Ko2); arg(k, 9, Vo2);
        run(k, "prep kv down", n2 / 4, HEADS, 1, 16, 1);
        const float qscale = 1.0f / sqrtf((float)HD);
        cl_kernel p = kPq;
        arg(p, 0, QKV); arg(p, 1, N); arg(p, 2, qr); arg(p, 3, qn); arg(p, 4, qscale);
        arg(p, 5, P); arg(p, 6, Q); arg(p, 7, Qo);
        run(p, "prep q", N, HEADS, 1, 64, 1);
    } else attnPrep(QKV, N, 0, N, qr, kr, vr, qn, kn, P, Q);
    attnCore(N, O, -1, ds);
}

// Cross-attention to a fixed prompt: keys / values made once per picture (crossKV), queries every step (crossAttn).
struct CrossKV { cl_mem K, V; int LK; int valid = -1; };   // valid: keys after it are padding
static CrossKV crossKV(cl_mem KVin, int LK, int kr, int vr, cl_mem kn) {
    CrossKV c{img(LK / 4, HEADS * 64), img(HEADS * 32, LK / 2), LK};
    const float qscale = 1.0f / sqrtf((float)HD);
    cl_kernel k = kPrepKV; int qr = 0;
    if (QKF) { rowRms(KVin, LK, kr, D / 8, rkB); arg(k, 13, rkB); arg(k, 14, rkB); }
    arg(k, 0, KVin); arg(k, 1, LK); arg(k, 2, qr); arg(k, 3, kr); arg(k, 4, vr);
    arg(k, 5, kn); arg(k, 6, kn); arg(k, 7, qscale); arg(k, 8, kn); arg(k, 9, kn);  // q and rope unused here
    arg(k, 10, Qo); arg(k, 11, c.K); arg(k, 12, c.V);
    run(k, "prep", LK / 4, HEADS, 1, 16, 1);
    return c;
}
static cl_mem Sc, Pc;
static void crossAttn(cl_mem QKV, int N, int qr, cl_mem qn, const CrossKV &c, float bias, cl_mem O) {
    const float qscale = 1.0f / sqrtf((float)HD);
    static int scLK = 0;
    if (c.LK > scLK) {   // grows to the longest key set used (Fibo: the gist's self-attention and the prompt bundles)
        if (Sc) { clReleaseMemObject(Sc); clReleaseMemObject(Pc); }
        Sc = img(MAXN, HEADS * (c.LK / 8)); Pc = img(MAXN, HEADS * (c.LK / 8)); scLK = c.LK;
    }
    cl_kernel p = kPqc;
    if (QKF) { rowRms(QKV, N, qr, D / 8, rqB); arg(p, 8, rqB); }
    arg(p, 0, QKV); arg(p, 1, N); arg(p, 2, qr); arg(p, 3, qn); arg(p, 4, qscale);
    arg(p, 5, qn); arg(p, 6, qn); arg(p, 7, Qo);   // no rope on cross-attention queries
    run(p, "prep q", N, HEADS, 1, 64, 1);
    gemm(0, "gemm qk", Qo, c.K, N, c.LK, HD, Sc, nullptr, HEADS, 16, 0, 64, c.LK / 8);
    softmaxRun(Sc, Pc, N, c.LK, HEADS, bias, c.valid);
    gemm(0, "gemm pv", Pc, c.V, N, HD, c.LK, O, nullptr, HEADS, c.LK / 8, 32, 0, 16);
}

}  // namespace ops
