// Shared pieces of the Vulkan engine (one program, one model per run): the model file, weights in the shaders'
// layouts, the shaders, and the operations a model's step is recorded from (vkrt.h Graph). The models' wiring lives in
// vk_<model>.h; vkmain.cpp picks one by the file's "ugpu.model".
// Row ranges: every operation takes its row count n, and buffers are passed as Refs, so a stream that occupies rows
// [a, b) of the token rows is addressed with byte offsets (R(buf, a * rowBytes)).
#pragma once
#include "vkrt.h"
#include "gen/gemv.spv.h"
#include "gen/attnprep.spv.h"
#include "gen/softmax.spv.h"
#include "gen/lnmod.spv.h"
#include "gen/copyf.spv.h"
#include "gen/rotq.spv.h"
#include "gen/gemm_h.spv.h"
#include "gen/gemm_q8d.spv.h"
#include "gen/gemm_q8c.spv.h"
#include "gen/gemm_q4c.spv.h"
#include "gen/gemm_q4c1.spv.h"
#include "gen/copyh.spv.h"
#include "gen/rmsadd.spv.h"
#include "gen/gemm_hc.spv.h"
#include "gen/softmax2.spv.h"
#include "gen/rowstats.spv.h"
#include "gen/rotq_sg.spv.h"
#include "gen/rowstats_sg.spv.h"
#include "gen/rowrms16.spv.h"
#include <chrono>
#include <cmath>
#include <fcntl.h>
#include <functional>
#include <algorithm>
#include <map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

typedef _Float16 h16;
using vk::Buf; using vk::Ref; using vk::R;

// ---- the model file ----
struct T { std::string kind; int M = 0, K = 0; const char *src = nullptr; size_t bytes = 0; };
static std::map<std::string, T> F;
static std::string MODEL;   // the file's "ugpu.model"
// The file stays mapped (tables are read from it while loading, and a few small ones every step), but a table's pages
// are handed back as soon as it is on the chip: otherwise the whole model was held twice (Fibo: 5.2 GB of file pages in
// the process next to 4.8 GB of chip buffers, too much for an 11 GB phone with the app open). Pages read again later
// simply come back from storage.
static int FileFd = -1;
static const char *FileBase = nullptr;
static size_t FileSize = 0;
static void dropPages(const void *p, size_t n) {
    if (!FileBase || !n) return;
    const size_t pg = 4096, a = ((const char *)p - FileBase) / pg * pg, e = std::min(FileSize, (size_t)((const char *)p - FileBase) + n);
    madvise((void *)(FileBase + a), e - a, MADV_DONTNEED);
    posix_fadvise(FileFd, (off_t)a, (off_t)(e - a), POSIX_FADV_DONTNEED);
}
static void loadGGUF(const std::string &path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { perror(path.c_str()); exit(1); }
    struct stat st; fstat(fd, &st);
    const char *base = (const char *)mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    FileFd = fd; FileBase = base; FileSize = st.st_size;
    const char *p = base;
    auto u32 = [&]() { uint32_t v; memcpy(&v, p, 4); p += 4; return v; };
    auto u64 = [&]() { uint64_t v; memcpy(&v, p, 8); p += 8; return v; };
    auto str = [&]() { uint64_t n = u64(); std::string v(p, n); p += n; return v; };
    if (memcmp(p, "GGUF", 4)) { fprintf(stderr, "not a GGUF file\n"); exit(1); }
    p += 4; u32();
    const uint64_t nt = u64(), nkv = u64();
    std::string index; uint64_t align = 32;
    std::function<void(uint32_t)> skipv = [&](uint32_t t) {
        static const int sz[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        if (t == 8) { str(); return; }
        if (t == 9) { const uint32_t et = u32(); const uint64_t n = u64(); for (uint64_t i = 0; i < n; i++) skipv(et); return; }
        p += sz[t];
    };
    for (uint64_t i = 0; i < nkv; i++) {
        const std::string key = str(); const uint32_t t = u32();
        if (t == 8 && key == "ugpu.index") index = str();
        else if (t == 8 && key == "ugpu.model") MODEL = str();
        else if (t == 4 && key == "general.alignment") align = u32();
        else skipv(t);
    }
    struct TI { std::string name; uint64_t n; uint32_t type; uint64_t off; };
    std::vector<TI> tis;
    for (uint64_t i = 0; i < nt; i++) {
        TI t; t.name = str(); const uint32_t nd = u32(); t.n = 1;
        for (uint32_t d = 0; d < nd; d++) t.n *= u64();
        t.type = u32(); t.off = u64(); tis.push_back(t);
    }
    const uint64_t data = ((p - base) + align - 1) / align * align;
    std::map<std::string, T> kinds;
    for (size_t a = 0; a < index.size();) {
        size_t b = index.find('\n', a); if (b == std::string::npos) b = index.size();
        char nm[128], kd[8]; int M, K;
        if (sscanf(index.substr(a, b - a).c_str(), "%127s %7s %d %d", nm, kd, &M, &K) == 4) { T t; t.kind = kd; t.M = M; t.K = K; kinds[nm] = t; }
        a = b + 1;
    }
    for (auto &t : tis) {
        T k = kinds.at(t.name);
        const size_t es = t.type == 0 ? 4 : t.type == 1 ? 2 : 1;
        k.src = base + data + t.off; k.bytes = t.n * es;
        F[t.name] = k;
    }
}
static const T &ft(const std::string &n) {
    auto it = F.find(n);
    if (it == F.end()) { fprintf(stderr, "model file has no %s\n", n.c_str()); exit(1); }
    return it->second;
}
static bool has(const std::string &n) { return F.count(n) != 0; }

// ---- weights on the chip, in the shaders' layouts ----
static size_t chipBytes = 0;
static double tWeights = 0, tKern = 0;   // setup time spent rearranging weights / building shaders (seconds)
struct Clock { std::chrono::steady_clock::time_point a = std::chrono::steady_clock::now(); double &to;
    explicit Clock(double &t) : to(t) {} ~Clock() { to += std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count(); } };
static Buf up(const void *src, size_t bytes) { Buf b = vk::buf(bytes); memcpy(b.map, src, bytes); chipBytes += bytes; return b; }
struct Q8 { Buf w, s; int M, K, GS; Buf sc; bool q4 = false; };   // q4: 4-bit strips + sub-scales sc (gemm_q4c)
static std::map<std::string, Q8> Wq;
static std::map<std::string, Buf> Wh, Wf;   // fp16 [M][K] row-major; fp32 as stored
// ---- streaming: the blocks' 4-bit tables stay in storage ----
// The chip's buffers are the phone's own memory, so every table kept on the chip counts against it (Fibo: 4.8 GB). With
// streamOf set (the model's: table -> its block, or -1 to keep it on the chip), each 4-bit block table instead shares a
// slot buffer with the same table of every other block of its parity (block % 2), and the model cuts its step graph
// into one segment per block (Graph::mark). While the chip runs block b, fillBlock(b + 1) reads the next block's tables
// from the file into the other slot (Graph::runStreamed). VKSTREAM=0 keeps everything on the chip.
static std::function<int(const std::string &)> streamOf;
static std::vector<std::vector<std::string>> STab;   // per block: its streamed tables
static size_t slotBytes = 0;
static double tFill = 0;
static bool streamWanted() { const char *e = getenv("VKSTREAM"); return !e || strcmp(e, "0"); }
static std::string slotName(const std::string &n, int b) { return "slot" + std::to_string(b % 2) + n.substr(n.find('.')); }
static Q8 &streamQ(const std::string &n, int b) {
    const T &t = ft(n), &s = ft(n + ".s");
    if ((int)STab.size() <= b) STab.resize(b + 1);
    if (std::find(STab[b].begin(), STab[b].end(), n) == STab[b].end()) STab[b].push_back(n);
    const std::string sn = slotName(n, b);
    auto it = Wq.find(sn);
    if (it != Wq.end()) {
        if (it->second.M != t.M || it->second.K != t.K || it->second.K / it->second.GS != s.K) {
            fprintf(stderr, "%s: shape differs from its slot's\n", n.c_str()); exit(1);
        }
        return it->second;
    }
    const int M = t.M, K = t.K, NBk = s.K;
    Q8 q;
    if (t.kind == "q4") q = Q8{vk::buf((size_t)M * K / 2), vk::buf((size_t)NBk * M * 4), M, K, K / NBk, vk::buf((size_t)M * (K / 32)), true};
    else q = Q8{vk::buf((size_t)M * K), vk::buf((size_t)NBk * M * 4), M, K, K / NBk};
    slotBytes += q.w.size + q.s.size + (q.q4 ? q.sc.size : 0);
    return Wq[sn] = q;
}
static void preadAt(void *dst, const T &t, size_t from = 0, size_t len = 0) {
    size_t off = t.src - FileBase + from, n = len ? len : t.bytes; char *d = (char *)dst;
    while (n) {
        const ssize_t r = pread(FileFd, d, n, (off_t)off);
        if (r <= 0) { perror("reading the model file"); exit(1); }
        d += r; off += r; n -= r;
    }
}
// block b's tables into its slot: the strips as stored, sub-scales and scales turned block-major as q8() does
static void fillBlock(int b) {
    if (b >= (int)STab.size()) return;
    Clock ck(tFill);
    static std::vector<uint8_t> a, z;
    for (auto &n : STab[b]) {
        Q8 &q = Wq.at(slotName(n, b));
        const T &t = ft(n), &s = ft(n + ".s");
        const int M = q.M, K32 = q.K / 32, NBk = q.K / q.GS;
        if (q.q4) {
            const T &c = ft(n + ".sc");
            preadAt(q.w.map, t);
            a.resize(c.bytes); z.resize(c.bytes); preadAt(a.data(), c);
            for (int o = 0; o < M; o++) for (int k = 0; k < K32; k++) z[(size_t)k * M + o] = a[(size_t)o * K32 + k];
            memcpy(q.sc.map, z.data(), c.bytes);
        } else {
            // 8-bit: the file's rows of 16 inputs ([K/16][M][16]) interleaved in pairs into the chip's [K/32][M][32], a few
            // MB of the file at a time
            const size_t pair = (size_t)M * 32;
            const int per = std::max<size_t>(1, ((size_t)8 << 20) / pair);
            a.resize(pair * per);
            int8_t *w = (int8_t *)q.w.map;
            for (int j0 = 0; j0 < K32; j0 += per) {
                const int nj = std::min(per, K32 - j0);
                preadAt(a.data(), t, (size_t)j0 * pair, (size_t)nj * pair);
                for (int j = 0; j < nj; j++) {
                    const uint8_t *lo = a.data() + (size_t)j * pair, *hi = lo + (size_t)M * 16;
                    int8_t *d = w + (size_t)(j0 + j) * pair;
                    for (int o = 0; o < M; o++) { memcpy(d + (size_t)o * 32, lo + (size_t)o * 16, 16); memcpy(d + (size_t)o * 32 + 16, hi + (size_t)o * 16, 16); }
                }
            }
        }
        a.resize(s.bytes); z.resize(s.bytes); preadAt(a.data(), s);
        const float *sf = (const float *)a.data(); float *df = (float *)z.data();
        for (int o = 0; o < M; o++) for (int k = 0; k < NBk; k++) df[(size_t)k * M + o] = sf[(size_t)o * NBk + k];
        memcpy(q.s.map, z.data(), s.bytes);
    }
}
static Q8 &q8(const std::string &n) {
    auto it = Wq.find(n);
    if (it != Wq.end()) return it->second;
    if (streamOf && ((ft(n).kind == "q4" && has(n + ".sc")) || ft(n).kind == "q8")) { const int b = streamOf(n); if (b >= 0) return streamQ(n, b); }
    Clock ck(tWeights);
    const T &t = ft(n), &s = ft(n + ".s");
    const int M = t.M, K = t.K, NBk = s.K, GS = K / NBk;
    if (t.kind == "q4" && has(n + ".sc")) {
        // 4-bit with two-level scales: the file's strips ([K/32][M][16 bytes]) are the chip's; sub-scales [M][K/32] ->
        // [K/32][M]; the fp32 scales per GS (256) inputs block-major as for 8-bit
        const T &c = ft(n + ".sc");
        Buf wb = vk::buf((size_t)M * K / 2); chipBytes += wb.size;
        memcpy(wb.map, t.src, wb.size);
        const int K32 = K / 32;
        Buf cb = vk::buf((size_t)M * K32); chipBytes += cb.size;
        uint8_t *cd = (uint8_t *)cb.map; const uint8_t *cs = (const uint8_t *)c.src;
        for (int o = 0; o < M; o++) for (int b = 0; b < K32; b++) cd[(size_t)b * M + o] = cs[(size_t)o * K32 + b];
        std::vector<float> sc((size_t)NBk * M);
        const float *ss = (const float *)s.src;
        for (int o = 0; o < M; o++) for (int b = 0; b < NBk; b++) sc[(size_t)b * M + o] = ss[(size_t)o * NBk + b];
        Q8 q{wb, up(sc.data(), sc.size() * 4), M, K, GS, cb, true};
        dropPages(t.src, t.bytes); dropPages(s.src, s.bytes); dropPages(c.src, c.bytes);
        return Wq[n] = q;
    }
    if (t.kind != "q8") { fprintf(stderr, "%s: kind %s (only q8 and 4-bit with sub-scales)\n", n.c_str(), t.kind.c_str()); exit(1); }
    // file: texel (o, k16) = 16 inputs of output o, rows of texels -> [K/16][M][16]; chip: [K/32][M][32]
    // straight into the chip buffer, split across the processor's cores (2.9 s on one core for Anima)
    Buf wb = vk::buf((size_t)M * K); chipBytes += wb.size;
    int8_t *w = (int8_t *)wb.map;
    {
        const int nt = std::max(1u, std::min(8u, std::thread::hardware_concurrency())), K16 = K / 16;
        std::vector<std::thread> th;
        for (int ti = 0; ti < nt; ti++) th.emplace_back([&, ti] {
            for (int k16 = ti; k16 < K16; k16 += nt) for (int o = 0; o < M; o++)
                memcpy(&w[(((size_t)(k16 / 2) * M + o) * 32) + (k16 % 2) * 16], t.src + ((size_t)k16 * M + o) * 16, 16);
        });
        for (auto &x : th) x.join();
    }
    std::vector<float> sc((size_t)NBk * M);
    const float *ss = (const float *)s.src;
    for (int o = 0; o < M; o++) for (int b = 0; b < NBk; b++) sc[(size_t)b * M + o] = ss[(size_t)o * NBk + b];
    Q8 q{wb, up(sc.data(), sc.size() * 4), M, K, GS};
    dropPages(t.src, t.bytes); dropPages(s.src, s.bytes);
    return Wq[n] = q;
}
// hands back the chip memory of the 8-bit / 4-bit tables keep() rejects (tables a model used once and is done with)
static size_t dropTables(const std::function<bool(const std::string &)> &drop) {
    size_t freed = 0;
    for (auto it = Wq.begin(); it != Wq.end();) {
        if (!drop(it->first)) { ++it; continue; }
        Q8 &q = it->second;
        freed += q.w.size + q.s.size + (q.q4 ? q.sc.size : 0);
        vk::freeBuf(q.w); vk::freeBuf(q.s); if (q.q4) vk::freeBuf(q.sc);
        it = Wq.erase(it);
    }
    chipBytes -= freed;
    return freed;
}
// one run of a step graph: streamed block by block when the model set streamOf, else whole
static void runStep(vk::Graph &g, std::vector<std::pair<std::string, double>> *times = nullptr) {
    if (streamOf) g.runStreamed([](size_t i) { fillBlock((int)i); }, times);
    else g.run(times);
}
static int fileGS(const std::string &n) { return ft(n).K / ft(n + ".s").K; }
static Buf &wh(const std::string &n) {
    auto it = Wh.find(n);
    if (it != Wh.end()) return it->second;
    Clock ck(tWeights);
    const T &t = ft(n);
    const int M = t.M, K = t.K;
    std::vector<h16> w((size_t)M * K);
    const h16 *s = (const h16 *)t.src;
    if (t.kind == "pk")   // W layout: texel (o/4, k/2) = outputs 4og..+3 at input 2k2, then at 2k2+1
        for (int k = 0; k < K; k++) for (int o = 0; o < M; o++)
            w[(size_t)o * K + k] = s[(((size_t)(k / 2) * (M / 4) + o / 4) * 8) + (k % 2) * 4 + o % 4];
    else if (t.kind == "h" || t.kind == "f16") memcpy(w.data(), s, w.size() * 2);
    else { fprintf(stderr, "%s: kind %s is not fp16\n", n.c_str(), t.kind.c_str()); exit(1); }
    dropPages(t.src, t.bytes);
    return Wh[n] = up(w.data(), w.size() * 2);
}
static Buf &wf(const std::string &n) {
    auto it = Wf.find(n);
    if (it != Wf.end()) return it->second;
    const T &t = ft(n);
    Wf[n] = up(t.src, t.bytes);
    dropPages(t.src, t.bytes);
    return Wf[n];
}

// ---- files ----
static std::vector<float> rdf(const std::string &p) {
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) { perror(p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v(n / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fprintf(stderr, "short read %s\n", p.c_str()); exit(1); }
    fclose(f); return v;
}
static bool fexists(const std::string &p) { FILE *f = fopen(p.c_str(), "rb"); if (f) fclose(f); return f != nullptr; }
static void wrf(const std::string &p, const float *v, size_t n) { FILE *f = fopen(p.c_str(), "wb"); fwrite(v, 4, n, f); fclose(f); }
static double relerr(const float *a, const float *b, size_t n) {
    double e = 0, r = 0;
    for (size_t i = 0; i < n; i++) { e += (a[i] - b[i]) * (double)(a[i] - b[i]); r += b[i] * (double)b[i]; }
    return sqrt(e / r);
}
static double secsSince(std::chrono::steady_clock::time_point a) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
}

// ---- kernels (one pipeline per shader + specialization) ----
static std::map<std::string, vk::Kern> KS;
static vk::Kern &K(const char *name, const uint32_t *code, size_t bytes, uint32_t nbuf, uint32_t pc, const std::vector<uint32_t> &spec) {
    std::string key = name;
    for (auto v : spec) key += "," + std::to_string(v);
    auto it = KS.find(key);
    if (it != KS.end()) return it->second;
    Clock ck(tKern);
    vk::Kern &k = KS[key];
    if (!vk::kern(k, key.c_str(), code, bytes, nbuf, pc, spec)) { fprintf(stderr, "the chip refused shader %s\n", key.c_str()); exit(1); }
    return k;
}
#define KERN(n, nb, pc, ...) K(#n, n##_spv, sizeof n##_spv, nb, pc, __VA_ARGS__)

static Buf dummy;
static bool useCM = false, sg64 = false, attnCM = false;
static vk::Graph *G;
static Buf Xq, Sx, Stats;   // the rounded input of the next 8-bit multiply, its scales, per-row layer-norm stats
static std::map<std::string, std::pair<Ref, Ref>> CENR;   // layer -> (mu, W.mu): Anima's centring
static inline int pad64(int n) { return (n + 63) / 64 * 64; }
static inline int pad128(int n) { return (n + 127) / 128 * 128; }

// the chip and what it can do; false when it has neither matrix hardware nor the byte dot product
static bool vkStart() {
    vk::init();
    vk::describe();
    useCM = vk::C.f.coopmat && vk::C.f.int8 && !(getenv("VKQ") && !strcmp(getenv("VKQ"), "dot"));
    if (!useCM && !vk::C.f.dot8) { fprintf(stderr, "this chip has neither matrix hardware nor the byte dot product\n"); return false; }
    sg64 = vk::C.f.subgroup == 64 && !getenv("VKNOSG");
    attnCM = vk::C.f.coopmat && !(getenv("VKATT") && !strcmp(getenv("VKATT"), "plain"));
    printf("8-bit multiplies on the %s, attention on the %s\n", useCM ? "matrix hardware" : "byte dot product",
           attnCM ? "matrix hardware" : "16-bit shader");
    dummy = vk::buf(256);
    return true;
}
// VKAUTO=1 (the app's automatic choice): Vulkan is only worth it with scale blocks of 256+ on matrix hardware
static bool autoDeclines(const std::string &probe) {
    const int gs = fileGS(probe);
    if (getenv("VKAUTO") && (gs < 256 || !useCM)) {
        printf("Vulkan skipped: scale blocks of %d inputs%s, OpenCL is as fast here\n", gs, useCM ? "" : " and no matrix hardware");
        return true;
    }
    if (ft(probe).kind == "q4" && !useCM) { printf("Vulkan skipped: 4-bit weights need the matrix hardware\n"); return true; }
    return false;
}
// scratch for the 8-bit multiplies: up to maxRows rows of up to maxK inputs
static void scratch(int maxRows, int maxK) {
    const int np = pad64(maxRows);
    Xq = vk::buf((size_t)np * maxK); Sx = vk::buf((size_t)np * (maxK / 32) * 4);
    memset(Xq.map, 0, Xq.size);
    Stats = vk::buf((size_t)np * 8);
}

struct PCq { uint32_t N, Npad, M, K, ldy, cen, GS; };
struct PCr { uint32_t N, K, ld, col0, Npad, GS; float eps; };
struct PCh { uint32_t N, M, K, ldx, ldw, ldy, bsx, bsw, bsy, cen; };
struct PCa { uint32_t N, ld, col0, outN; float scale; };
struct PCs { uint32_t N, LK, kvalid; float bias; };
struct PCv { uint32_t M, K, act, hasAdd; };
struct PCl { uint32_t N, D; float eps; };
struct PCc { uint32_t N, M, K, ldx, ldw, ldy, bsx, bsw, bsy; };

// rounding for the next 8-bit multiply, n rows of Kd inputs into Xq / Sx:
//   rotLn: from the fp32 stream with a layer norm (rms: RMS norm) and the modulation x (1 + sc) + sh
//   rotX:  from an fp16 activation (row stride ld)
static void rotLn(const std::string &wn, Ref res, Ref sh, Ref sc, int Kd, int n, float eps = 1e-6f, bool rms = false) {
    auto c = CENR.find(wn);
    const bool cen = c != CENR.end();
    PCr pc{(uint32_t)n, (uint32_t)Kd, 0, 0, (uint32_t)pad64(n), (uint32_t)q8(wn).GS, eps};
    struct { uint32_t N, K; float eps; } ps{(uint32_t)n, (uint32_t)Kd, eps};
    if (sg64) {
        const uint32_t LB = q8(wn).GS / 4;
        G->add(KERN(rowstats_sg, 2, 12, {rms ? 1u : 0u}), {res, R(Stats)}, ps, (n + 3) / 4, 1, 1, "lnq");
        G->add(KERN(rotq_sg, 8, sizeof(PCr), {1, cen ? 1u : 0u, LB}), {res, R(dummy), sh, sc, cen ? c->second.first : R(dummy), R(Xq), R(Sx), R(Stats)}, pc, n, Kd / 256, 1, "lnq");
        return;
    }
    G->add(KERN(rowstats, 2, 12, {rms ? 1u : 0u}), {res, R(Stats)}, ps, n, 1, 1, "lnq");
    G->add(KERN(rotq, 8, sizeof(PCr), {1, cen ? 1u : 0u}), {res, R(dummy), sh, sc, cen ? c->second.first : R(dummy), R(Xq), R(Sx), R(Stats)}, pc, n, Kd / 256, 1, "lnq");
}
static void rotX(const std::string &wn, Ref in, int ld, int Kd, int n) {
    auto c = CENR.find(wn);
    const bool cen = c != CENR.end();
    PCr pc{(uint32_t)n, (uint32_t)Kd, (uint32_t)ld, 0, (uint32_t)pad64(n), (uint32_t)q8(wn).GS, 1e-6f};
    if (sg64) {
        const uint32_t LB = q8(wn).GS / 4;
        G->add(KERN(rotq_sg, 8, sizeof(PCr), {0, cen ? 1u : 0u, LB}), {R(dummy), in, R(dummy), R(dummy), cen ? c->second.first : R(dummy), R(Xq), R(Sx), R(dummy)}, pc, n, Kd / 256, 1, "rotq");
        return;
    }
    G->add(KERN(rotq, 8, sizeof(PCr), {0, cen ? 1u : 0u}), {R(dummy), in, R(dummy), R(dummy), cen ? c->second.first : R(dummy), R(Xq), R(Sx), R(dummy)}, pc, n, Kd / 256, 1, "rotq");
}
// the 8-bit multiply of the rounded rows (n of them) by layer wn (plus its bias "<wn>.b" when the file has one);
// epi as epi.glsl (4: SwiGLU, 5: MOSS's pair, M/2 fp16 outputs; 6: tanh GELU),
// into Y (fp16, row stride ldy) or res (fp32, row stride ldy); ldy 0: the layer's own output width
static void mulq(int epi, const char *tag, const std::string &wn, Ref Y, Ref res, Ref gate, int n, int ldy = 0) {
    Q8 &q = q8(wn);
    auto c = CENR.find(wn);
    // the ending's added vector: Anima's centring offset W.mu, or the layer's own bias "<wn>.b"
    const bool cen = c != CENR.end(), bias = !cen && has(wn + ".b");
    if (!ldy) ldy = epi == 4 || epi == 5 ? q.M / 2 : q.M;
    PCq pc{(uint32_t)n, (uint32_t)pad64(n), (uint32_t)q.M, (uint32_t)q.K, (uint32_t)ldy, cen || bias ? 1u : 0u, (uint32_t)q.GS};
    std::vector<Ref> b = {R(Xq), R(q.w), R(Sx), R(q.s), Y, res, gate, cen ? c->second.second : bias ? R(wf(wn + ".b")) : R(dummy)};
    if (q.q4) {
        if (!useCM || epi == 5) { fprintf(stderr, "%s: 4-bit weights need the matrix hardware\n", tag); exit(1); }
        // the expanded weights are shared by SM rows of subgroups (Q4SM / Q4SN to try others)
        const uint32_t SG = vk::C.f.subgroup, TM = 64;
        static const uint32_t SM = getenv("Q4SM") ? atoi(getenv("Q4SM")) : 2, SN = getenv("Q4SN") ? atoi(getenv("Q4SN")) : 1,
                              TN = getenv("Q4TN") ? atoi(getenv("Q4TN")) : 32;
        b.push_back(R(q.sc));
        if (getenv("Q4V1")) {
            G->add(KERN(gemm_q4c1, 9, sizeof(PCq), {TM, 32, 1, 2, SG, (uint32_t)epi, 2 * SG}), b, pc, q.M / 64, pad64(n) / TM, 1, tag);
            return;
        }
        G->add(KERN(gemm_q4c, 9, sizeof(PCq), {TM, TN, SM, SN, SG, (uint32_t)epi, SM * SN * SG}), b, pc, q.M / (TN * SN),
               (pad64(n) + TM * SM - 1) / (TM * SM), 1, tag);
    } else if (useCM) {
        const uint32_t SG = vk::C.f.subgroup, TM = 64, TN = 32, SM = 1, SN = 2;
        G->add(KERN(gemm_q8c, 8, sizeof(PCq), {TM, TN, SM, SN, SG, (uint32_t)epi, SM * SN * SG}), b, pc, q.M / (TN * SN), pad64(n) / (TM * SM), 1, tag);
    } else {
        if (epi == 4 || epi == 5) { fprintf(stderr, "%s: this ending needs the matrix hardware\n", tag); exit(1); }
        G->add(KERN(gemm_q8d, 8, sizeof(PCq), {(uint32_t)epi}), b, pc, q.M / 64, pad64(n) / 64, 1, tag);
    }
}
// fp16 multiply (batched over z): see gemm_h.comp. wt 0: W is [m][k], 1: W is [k][m]
static void mulh(int epi, const char *tag, int wt, Ref X, Ref W, Ref Y, Ref res, int n, int m, int k, int ldx, int ldw, int ldy,
                 int bsx = 0, int bsw = 0, int bsy = 0, int batch = 1) {
    const uint32_t BK = k % 16 ? 8 : 16;
    PCh pc{(uint32_t)n, (uint32_t)m, (uint32_t)k, (uint32_t)ldx, (uint32_t)ldw, (uint32_t)ldy, (uint32_t)bsx, (uint32_t)bsw, (uint32_t)bsy, 0};
    G->add(KERN(gemm_h, 6, sizeof(PCh), {64, 64, BK, 4, 4, (uint32_t)wt, (uint32_t)epi, 256}), {X, W, Y, res, R(dummy), R(dummy)}, pc,
           (m + 63) / 64, (n + 63) / 64, batch, tag);
}
// attention's fp16 multiplies (fp16 out): on the matrix hardware when there is one (sizes multiples of 64 / 128), else gemm_h
static void mula(const char *tag, int bt, Ref X, Ref W, Ref Y, int n, int m, int k, int ldx, int ldw, int ldy, int bsx, int bsw, int bsy, int batch) {
    if (attnCM && n % 64 == 0 && m % 128 == 0 && k % 64 == 0) {
        const uint32_t SG = vk::C.f.subgroup, SM = 1, SN = 2, KS = 4;
        PCc pc{(uint32_t)n, (uint32_t)m, (uint32_t)k, (uint32_t)ldx, (uint32_t)ldw, (uint32_t)ldy, (uint32_t)bsx, (uint32_t)bsw, (uint32_t)bsy};
        G->add(KERN(gemm_hc, 3, sizeof(PCc), {64, 64, SM, SN, KS, (uint32_t)bt, SM * SN * SG}), {X, W, Y}, pc, m / (64 * SN), n / (64 * SM), batch, tag);
    } else mulh(0, tag, bt, X, W, Y, R(dummy), n, m, k, ldx, ldw, ldy, bsx, bsw, bsy, batch);
}
// attention inputs for n rows: from fp16 [n][ld] at column col0 into [heads][outN][128] (out already offset to the
// first row); P / Q offset to the same first row
// rs: per-row 1 / rms over the whole width (rowrms below) for Wan's whole-width norms; R(dummy) for per-head norms
static void prep(Ref in, int ld, int col0, Ref w, Ref P, Ref Q, Ref out, int n, int outN, int heads, bool norm, bool rope, float scale,
                 const char *tag, Ref rs = Ref{nullptr}) {
    PCa pc{(uint32_t)n, (uint32_t)ld, (uint32_t)col0, (uint32_t)outN, scale};
    const bool full = rs.b != nullptr;
    G->add(KERN(attnprep, 6, sizeof(PCa), {norm ? 1u : 0u, rope ? 1u : 0u, full ? 1u : 0u}), {in, w, P, Q, out, full ? rs : R(dummy)}, pc, n, heads, 1, tag);
}
// RS[n] = 1 / rms of K fp16 channels of row n from column col0
static void rowrms(Ref in, int ld, int col0, int width, Ref RS, int n, float eps = 1e-6f) {
    struct { uint32_t N, ld, col0, K; float eps; } pc{(uint32_t)n, (uint32_t)ld, (uint32_t)col0, (uint32_t)width, eps};
    G->add(KERN(rowrms16, 2, sizeof(pc), {}), {in, RS}, pc, n, 1, 1, "rowrms");
}
static void softmax(Ref S, int n, int lk, int kvalid, float bias, int heads, bool causal = false) {
    PCs pc{(uint32_t)n, (uint32_t)lk, (uint32_t)kvalid, bias};
    if (lk % 4 == 0) {
        const uint32_t VPL = (lk + 255) / 256, RPW = 4;
        G->add(KERN(softmax2, 1, sizeof(PCs), {VPL, RPW, causal ? 1u : 0u}), {S}, pc, (n + RPW - 1) / RPW, heads, 1, "softmax");
    } else if (causal) { fprintf(stderr, "one-way attention needs a key count that is a multiple of 4\n"); exit(1); }
    else G->add(KERN(softmax, 1, sizeof(PCs), {}), {S}, pc, n, heads, 1, "softmax");
}
// self-attention over Qh / Kh / Vh ([heads][rows][128], rows a multiple of 128; keys from kvalid on masked), hc heads a
// pass through the score table S, into O (fp16 [rows][ldo], head h at column 128 h)
static void attend(Buf &Qh, Buf &Kh, Buf &Vh, Buf &S, Ref O, int rows, int kvalid, int heads, int hc, int ldo, bool causal = false) {
    for (int h0 = 0; h0 < heads; h0 += hc) {
        const int nh = std::min(hc, heads - h0);
        const size_t ho = (size_t)h0 * rows * 128 * 2;
        mula("qk", 0, R(Qh, ho), R(Kh, ho), R(S), rows, rows, 128, 128, 128, rows, rows * 128, rows * 128, rows * rows, nh);
        softmax(R(S), rows, rows, kvalid, 0.0f, nh, causal);
        mula("pv", 1, R(S), R(Vh, ho), Ref{O.b, O.off + (size_t)h0 * 128 * 2, 0}, rows, 128, rows, rows, 128, ldo, rows * rows, rows * 128, 128, nh);
    }
}
// heads per attention pass so the score table stays under limitMB
static int headsPerPass(int heads, int rows, int limitMB = 64) {
    int hc = heads;
    while (hc > 1 && (size_t)hc * rows * rows * 2 > ((size_t)limitMB << 20)) hc /= 2;
    return hc;
}
static void gemv(Buf &W, Ref x, Ref add, Ref y, int M, int Kd, int act) {
    PCv pc{(uint32_t)M, (uint32_t)Kd, (uint32_t)act, add.b != &dummy ? 1u : 0u};
    G->add(KERN(gemv, 4, sizeof(PCv), {}), {R(W), x, add, y}, pc, M, 1, 1, "gemv");
}
static void copyf(Ref src, Ref dst, size_t n) {
    struct { uint32_t n; } pc{(uint32_t)n};
    G->add(KERN(copyf, 2, 4, {}), {src, dst}, pc, (uint32_t)std::min<size_t>((n + 255) / 256, 65535), (uint32_t)((n + 255) / 256 + 65534) / 65535, 1, "copy");
}
// res (fp32 [n][D]) += A (fp16, row stride ld) / rms(A row) * w  (Fibo's post-norm residual)
static void rmsadd(Ref A, int ld, Ref res, Ref w, int n, int D) {
    struct { uint32_t N, D, ld; } pc{(uint32_t)n, (uint32_t)D, (uint32_t)ld};
    G->add(KERN(rmsadd, 3, sizeof(pc), {}), {A, res, w}, pc, (uint32_t)n, 1, 1, "rmsadd");
}
// cross-attention: nq query rows (Qh [heads][nq][128], nq a multiple of 64) over lk key rows (Kh / Vh [heads][lk][128],
// lk a multiple of 128; keys from kvalid on masked), hc heads a pass through S, into O (fp16 [nq][ldo])
static void attendX(Buf &Qh, int nq, Buf &Kh, Buf &Vh, int lk, int kvalid, Buf &S, Ref O, int heads, int hc, int ldo) {
    for (int h0 = 0; h0 < heads; h0 += hc) {
        const int nh = std::min(hc, heads - h0);
        mula("qk", 0, R(Qh, (size_t)h0 * nq * 256), R(Kh, (size_t)h0 * lk * 256), R(S), nq, lk, 128, 128, 128, lk, nq * 128, lk * 128, nq * lk, nh);
        softmax(R(S), nq, lk, kvalid, 0.0f, nh);
        mula("pv", 1, R(S), R(Vh, (size_t)h0 * lk * 256), Ref{O.b, O.off + (size_t)h0 * 128 * 2, 0}, nq, 128, lk, lk, 128, ldo, nq * lk, lk * 128, 128, nh);
    }
}
// heads blocks of n 32-bit words: dst[h * dstride + i] = src[h * sstride + i] (strides in words)
static void copyh(Ref src, Ref dst, int n, int heads, int sstride, int dstride) {
    struct { uint32_t n, s, d; } pc{(uint32_t)n, (uint32_t)sstride, (uint32_t)dstride};
    G->add(KERN(copyh, 2, sizeof(pc), {}), {src, dst}, pc, (uint32_t)(n + 255) / 256, (uint32_t)heads, 1, "copyh");
}
// Y (fp16 [n][D]) = layernorm(res) (1 + sc) + sh
static void lnmod(Ref res, Ref sh, Ref sc, Ref Y, int n, int D, float eps = 1e-6f) {
    PCl pc{(uint32_t)n, (uint32_t)D, eps};
    G->add(KERN(lnmod, 4, sizeof(PCl), {}), {res, sh, sc, Y}, pc, n, 1, 1, "lnmod");
}
// fp16 rows (n x C, row-major) -> floats
static std::vector<float> readH(Buf &b, size_t n) {
    std::vector<float> y(n);
    for (size_t i = 0; i < n; i++) y[i] = (float)((h16 *)b.map)[i];
    return y;
}
static void putH(Buf &b, const float *v, size_t n) { for (size_t i = 0; i < n; i++) ((h16 *)b.map)[i] = (h16)v[i]; }

static void printTimes(const std::vector<std::pair<std::string, double>> &times) {
    if (times.empty()) return;
    double tot = 0; for (auto &t : times) tot += t.second;
    for (auto &t : times) printf("  %-12s %8.1f ms  %4.1f%%\n", t.first.c_str(), t.second, 100 * t.second / tot);
}
