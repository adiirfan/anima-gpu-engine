// Universal graphics-chip engine: the runtime (2026-11-18). Talks to the chip through standard OpenCL 1.2/2.0 +
// cl_khr_fp16 only, tunes the main multiply per chip on first run, loads weight files. Nothing in here knows a model.
#pragma once
#define CL_TARGET_OPENCL_VERSION 200
#include <atomic>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <future>
#include <cctype>
#include <string>
#include <functional>
#include <tuple>
#include <vector>

#define CK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, e_); exit(1); } } while (0)
#ifdef __aarch64__
typedef __fp16 h16;
#else
typedef _Float16 h16;   // computer build (testing on the Radeon)
#endif

static size_t CACHE = 4u << 20;     // bytes of input a multiply works on at a time (the chip's cache)
static int TN = 6, TM = 2, L0 = 16, L1 = 8;                       // tile multiply
static int TNI = 2, GG = 4, WGS = 128, KCS = 64, LOCALGEMM = 1;  // shared-weight multiply (tuned per chip)

static cl_context ctx;
static cl_command_queue q;
static cl_device_id dev;
static bool prof = false;
static std::map<std::string, double> tsum;

static std::string slurp(const char *p) {
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::string s(n, 0); fread(&s[0], 1, n, f); fclose(f); return s;
}
static std::vector<float> rdf(const std::string &p) {
    std::string s = slurp(p.c_str());
    std::vector<float> v(s.size() / 4); memcpy(v.data(), s.data(), s.size()); return v;
}

static std::string KEXTRA;   // options added to every kernel build (e.g. -DNO_IDP=1 on chips without the byte dot product)
// KCACHE=<folder>: keep each built program there and load it next run instead of building it again (the chip's
// compiler takes seconds per run). The file name covers the source, the options, the chip and its driver, so a
// changed kernel or a driver update builds afresh; a file the driver refuses is simply built again and replaced.
static std::string kcachePath(const std::string &src, const std::string &opts) {
    const char *dir = getenv("KCACHE");
    if (!dir || !*dir) return "";
    char dn[256] = {0}, dv[256] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof dn - 1, dn, 0);
    clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof dv - 1, dv, 0);
    uint64_t h = 1469598103934665603ull;   // FNV-1a over everything that changes the built program
    auto mix = [&](const std::string &x) { for (unsigned char c : x) { h ^= c; h *= 1099511628211ull; } h ^= 0xff; h *= 1099511628211ull; };
    mix(src); mix(opts); mix(dn); mix(dv);
    char name[40]; snprintf(name, sizeof name, "/%016llx.bin", (unsigned long long)h);
    return std::string(dir) + name;
}
static cl_program build(const std::string &src, const std::string &opts0) {
    const std::string opts = opts0 + KEXTRA;
    const std::string cp = kcachePath(src, opts);
    cl_int err;
    if (!cp.empty()) {
        if (FILE *f = fopen(cp.c_str(), "rb")) {
            fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
            std::vector<unsigned char> bin(n > 0 ? n : 0);
            const bool got = n > 0 && fread(bin.data(), 1, n, f) == (size_t)n;
            fclose(f);
            if (got) {
                const unsigned char *b = bin.data(); size_t bn = bin.size(); cl_int st = 0;
                cl_program p = clCreateProgramWithBinary(ctx, 1, &dev, &bn, &b, &st, &err);
                if (!err && !st && !clBuildProgram(p, 1, &dev, opts.c_str(), 0, 0)) return p;
                if (p) clReleaseProgram(p);
            }
            remove(cp.c_str());   // refused (driver changed underneath): build from source below
        }
    }
    const char *s = src.c_str();
    cl_program p = clCreateProgramWithSource(ctx, 1, &s, 0, &err); CK(err);
    if (clBuildProgram(p, 1, &dev, opts.c_str(), 0, 0)) {
        size_t n; clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, 0, &n);
        std::string log(n, 0); clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, n, &log[0], 0);
        fprintf(stderr, "build failed [%s]:\n%s\n", opts.c_str(), log.c_str()); exit(1);
    }
    if (!cp.empty()) {   // keep it for next time; written aside and renamed so a killed run leaves no half file
        size_t bn = 0;
        if (!clGetProgramInfo(p, CL_PROGRAM_BINARY_SIZES, sizeof bn, &bn, 0) && bn) {
            std::vector<unsigned char> bin(bn); unsigned char *b = bin.data();
            if (!clGetProgramInfo(p, CL_PROGRAM_BINARIES, sizeof b, &b, 0)) {
                const std::string tmp = cp + ".part";
                if (FILE *f = fopen(tmp.c_str(), "wb")) {
                    const bool ok = fwrite(b, 1, bn, f) == bn;
                    fclose(f);
                    if (ok) rename(tmp.c_str(), cp.c_str()); else remove(tmp.c_str());
                }
            }
        }
    }
    return p;
}
static cl_kernel kern(cl_program p, const char *name) { cl_int e; cl_kernel k = clCreateKernel(p, name, &e); CK(e); return k; }

// MEMREPORT=1: the chip memory the program asked for in all (pictures 16 bytes a cell, buffers their size), printed
// at the end; streamed blocks reuse their two slots, so this is what a run holds
static std::atomic<size_t> MEMB{0};
static void memReport() { if (getenv("MEMREPORT")) printf("chip memory made: %.0f MB\n", MEMB.load() / 1e6); }
// MEMREPORT=2: also one line for every piece of 16 MB or more as it is made
static void memLine(const char *what, size_t bytes, int w, int h, bool filled) {
    static const bool on = getenv("MEMREPORT") && atoi(getenv("MEMREPORT")) >= 2;
    if (on && bytes >= (16u << 20)) fprintf(stderr, "  made %s %dx%d %s: %.0f MB (so far %.0f)\n", what, w, h, filled ? "weights" : "work", bytes / 1e6, MEMB.load() / 1e6);
}
static cl_mem img(int w, int h, const void *data = nullptr) {
    MEMB += (size_t)w * h * 16; memLine("picture", (size_t)w * h * 16, w, h, data != nullptr);
    cl_image_format f = {CL_RGBA, CL_UNSIGNED_INT32};
    cl_image_desc d; memset(&d, 0, sizeof d);
    d.image_type = CL_MEM_OBJECT_IMAGE2D; d.image_width = w; d.image_height = h;
    cl_int e;
    cl_mem m = clCreateImage(ctx, data ? (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR) : CL_MEM_READ_WRITE, &f, &d, (void *)data, &e);
    if (e) { fprintf(stderr, "image %dx%d failed %d\n", w, h, e); exit(1); }
    return m;
}
static cl_mem buf(size_t bytes, const void *data = nullptr) {
    MEMB += bytes; memLine("buffer", bytes, 0, 0, data != nullptr);
    cl_int e;
    cl_mem m = clCreateBuffer(ctx, data ? (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR) : CL_MEM_READ_WRITE, bytes, (void *)data, &e);
    CK(e); return m;
}
static cl_mem sub(cl_mem b, size_t off, size_t bytes) {
    cl_buffer_region r = {off, bytes}; cl_int e;
    cl_mem m = clCreateSubBuffer(b, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r, &e); CK(e); return m;
}

template <class T> static void arg(cl_kernel k, int i, const T &v) {
    const cl_int e = clSetKernelArg(k, i, sizeof(T), &v);
    if (e != CL_SUCCESS) {
        char nm[64] = {0}; clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, sizeof nm - 1, nm, 0);
        cl_uint na = 0; clGetKernelInfo(k, CL_KERNEL_NUM_ARGS, sizeof na, &na, 0);
        fprintf(stderr, "kernel %s argument %d (of %u): error %d\n", nm, i, na, e); exit(1);
    }
}
static void argnull(cl_kernel k, int i) { cl_mem z = nullptr; CK(clSetKernelArg(k, i, sizeof(cl_mem), &z)); }

static void run(cl_kernel k, const char *tag, size_t g0, size_t g1, size_t g2, size_t l0, size_t l1,
                size_t o0 = 0, size_t o1 = 0) {
    size_t lws[3] = {l0, l1, 1};
    size_t gws[3] = {(g0 + l0 - 1) / l0 * l0, (g1 + l1 - 1) / l1 * l1, g2};
    size_t off[3] = {o0, o1, 0};
    cl_event e;
    cl_int er = clEnqueueNDRangeKernel(q, k, 3, (o0 || o1) ? off : nullptr, gws, lws, 0, 0, prof ? &e : nullptr);
    if (er) { fprintf(stderr, "kernel %s failed: %d (global %zu %zu %zu)\n", tag, er, gws[0], gws[1], gws[2]); exit(1); }
    if (prof) {
        clWaitForEvents(1, &e);
        cl_ulong a, b;
        clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, 8, &a, 0);
        clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, 8, &b, 0);
        tsum[tag] += (b - a) * 1e-6; clReleaseEvent(e);
    }
}

static void run3(cl_kernel k, const char *tag, size_t g0, size_t g1, size_t g2, size_t l0, size_t l1, size_t l2) {
    size_t lws[3] = {l0, l1, l2};
    size_t gws[3] = {(g0 + l0 - 1) / l0 * l0, (g1 + l1 - 1) / l1 * l1, (g2 + l2 - 1) / l2 * l2};
    cl_event e;
    cl_int er = clEnqueueNDRangeKernel(q, k, 3, 0, gws, lws, 0, 0, prof ? &e : nullptr);
    if (er) { fprintf(stderr, "kernel %s failed: %d\n", tag, er); exit(1); }
    if (prof) {
        clWaitForEvents(1, &e);
        cl_ulong a, b;
        clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, 8, &a, 0);
        clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, 8, &b, 0);
        tsum[tag] += (b - a) * 1e-6; clReleaseEvent(e);
    }
}


// ---- kernel source ----
// The Adreno compiler builds a kernel differently depending on what else is in its program (gemmq beside softmax2
// ran 25x slower), so heavy kernels are built from their own slice: the shared header plus one marked section.
static std::string KSRC;
static std::string kslice(const char *from, const char *to = nullptr) {
    const size_t a = KSRC.find(from);
    if (a == std::string::npos) { fprintf(stderr, "kernel section %s missing\n", from); exit(1); }
    const size_t b = to ? KSRC.find(to) : std::string::npos;
    return KSRC.substr(0, KSRC.find("__kernel")) + KSRC.substr(a, b == std::string::npos ? std::string::npos : b - a);
}

// 8-bit weights as an image: texel (o, k16) = 16 int8 inputs of output o. Wider than the chip allows (Klein's
// 18432-row SwiGLU layers on Adreno's 16384), the outputs are folded into P parts stacked vertically: output o of
// part p = o / (M/P) sits at (o - p M/P, p K/16 + k16). The kernel unfolds with get_image_width.
static size_t IMGW = 0;
static bool QUIET_FOLD = false;   // set once blocks are streamed: no fold message per load
// kd: inputs per texel (16 for 8-bit weights, 32 for 4-bit ones)
// 8-bit / 4-bit weight table as an image: texel (o, k) = 16 bytes. Tables wider than the chip allows are folded:
// outputs past the width go to the next part, stacked below. Returns the image size and the bytes to put in it.
static const char *q8layout(int M, int K, const char *src, int kd, std::vector<char> &t, int &w, int &h) {
    if (!IMGW) clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof IMGW, &IMGW, 0);
    if ((size_t)M <= IMGW) { w = M; h = K / kd; return src; }
    int P = 2; while ((size_t)(M / P) > IMGW || M % P || (M / P) % 16) P++;
    const int MW = M / P, H = K / kd;
    t.resize((size_t)M * H * 16);
    for (int p = 0; p < P; p++) for (int k = 0; k < H; k++)
        memcpy(&t[((size_t)(p * H + k) * MW) * 16], src + ((size_t)k * M + (size_t)p * MW) * 16, (size_t)MW * 16);
    if (!QUIET_FOLD) printf("  8-bit weights %dx%d folded into %d parts\n", M, K, P);
    w = MW; h = P * H; return t.data();
}
static cl_mem q8img(int M, int K, const char *src, int kd = 16) {
    std::vector<char> t; int w, h;
    const char *d = q8layout(M, K, src, kd, t, w, h);
    return img(w, h, d);
}
// ---- weights ----
// <prefix>.bin + <prefix>.idx: one line per tensor "name kind M K offset bytes"; kinds f32/f16 (buffers), pk
// (fp16 packed weights, W layout image), q8 (int8, rotated), qs (8-bit block scales). An optional first line
// "#model <name>" names the wiring to use.
struct Wt { std::string kind; int M, K; cl_mem m; };
static std::map<std::string, Wt> W;
static std::string loadWeights(const std::string &prefix, const std::function<bool(const std::string &)> &skip = nullptr) {
    int fd = open((prefix + ".bin").c_str(), O_RDONLY);
    if (fd < 0) { perror((prefix + ".bin").c_str()); exit(1); }
    struct stat st; fstat(fd, &st);
    const char *base = (const char *)mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    FILE *f = fopen((prefix + ".idx").c_str(), "r");
    if (!f) { perror((prefix + ".idx").c_str()); exit(1); }
    std::string model;
    char line[512], name[128], kind[8]; int M, K; long off, bytes;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') { char m[64]; if (sscanf(line, "#model %63s", m) == 1) model = m; continue; }
        if (sscanf(line, "%127s %7s %d %d %ld %ld", name, kind, &M, &K, &off, &bytes) != 6) continue;
        if (skip && skip(name)) continue;
        Wt w{kind, M, K, nullptr};
        if (w.kind == "pk") w.m = img(M / 4, K / 2, base + off);
        else if (w.kind == "q8") w.m = q8img(M, K, base + off);
        else if (w.kind == "q4") w.m = q8img(M, K, base + off, 32);
        else w.m = buf(bytes, base + off);
        W[name] = w;
    }
    fclose(f); munmap((void *)base, st.st_size); close(fd);
    return model;
}
// GGUF container (2026-11-18): the same tensors stored as plain GGUF tensors (I8 / F16 / F32) with the engine's layouts
// inside; metadata "ugpu.index" holds "name kind M K" per tensor, "ugpu.model" the wiring. Standard tools can open the
// file; only this engine knows what the layouts mean. "ugpu.program" (optional) is a layer list for model_tae.h.
static std::string GPROG;
// STREAM=1: the drawing blocks' big tables (8/4-bit weights and their scales, names "b<k>.*") stay in the file and are
// turned into chip objects one block at a time just before use (ops.h Streamer), so a model bigger than the phone's
// memory still runs. The file stays mapped; everything else loads at start as usual.
struct SInfo { std::string kind; int M, K; const char *src; size_t bytes; };
static std::map<std::string, SInfo> SW;
static bool STREAM = false;
static bool streamedName(const std::string &n, const std::string &kind) {
    return STREAM && n.size() > 2 && n[0] == 'b' && isdigit((unsigned char)n[1]) && (kind == "q8" || kind == "q4" || kind == "qs");
}
static std::string loadGGUF(const std::string &path, const std::function<bool(const std::string &)> &skip = nullptr) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { perror(path.c_str()); exit(1); }
    struct stat st; fstat(fd, &st);
    const char *base = (const char *)mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    const char *p = base;
    auto u32 = [&]() { uint32_t v; memcpy(&v, p, 4); p += 4; return v; };
    auto u64 = [&]() { uint64_t v; memcpy(&v, p, 8); p += 8; return v; };
    auto str = [&]() { uint64_t n = u64(); std::string v(p, n); p += n; return v; };
    if (memcmp(p, "GGUF", 4)) { fprintf(stderr, "%s: not a GGUF file\n", path.c_str()); exit(1); }
    p += 4;
    const uint32_t ver = u32(); const uint64_t nt = u64(), nkv = u64();
    if (ver < 2) { fprintf(stderr, "GGUF version %u too old\n", ver); exit(1); }
    std::string model, index; uint64_t align = 32;
    std::function<void(uint32_t)> skipv = [&](uint32_t t) {
        static const int sz[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        if (t == 8) { str(); return; }
        if (t == 9) { const uint32_t et = u32(); const uint64_t n = u64(); for (uint64_t i = 0; i < n; i++) skipv(et); return; }
        p += sz[t];
    };
    for (uint64_t i = 0; i < nkv; i++) {
        const std::string key = str(); const uint32_t t = u32();
        if (t == 8 && key == "ugpu.model") model = str();
        else if (t == 8 && key == "ugpu.index") index = str();
        else if (t == 8 && key == "ugpu.program") GPROG = str();
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
    std::map<std::string, std::tuple<std::string, int, int>> kinds;
    {
        size_t a = 0;
        while (a < index.size()) {
            size_t b = index.find('\n', a); if (b == std::string::npos) b = index.size();
            char nm[128], kd[8]; int M, K;
            if (sscanf(index.substr(a, b - a).c_str(), "%127s %7s %d %d", nm, kd, &M, &K) == 4) kinds[nm] = {kd, M, K};
            a = b + 1;
        }
    }
    // Klein with STREAM=1: its blocks "d<i>." (double) and "s<j>." (single) take the streamer's names "b<i>." and
    // "b<ND + j>." here, so the published file streams as it is
    int kleinND = 0;
    if (STREAM && model == "klein")
        for (auto &t : tis) if (t.name[0] == 'd' && isdigit((unsigned char)t.name[1])) kleinND = std::max(kleinND, atoi(t.name.c_str() + 1) + 1);
    const bool animaStream = STREAM && model == "anima";   // Anima's blocks are "<i>.": they stream as "b<i>."
    auto alias = [&](const std::string &n) {
        const size_t dot = n.find('.');
        if (animaStream && dot != std::string::npos && dot > 0) {
            bool num = true;
            for (size_t i = 0; i < dot; i++) num = num && isdigit((unsigned char)n[i]);
            if (num) return "b" + n;
        }
        if (!kleinND || dot == std::string::npos || dot < 2 || (n[0] != 'd' && n[0] != 's')) return n;
        for (size_t i = 1; i < dot; i++) if (!isdigit((unsigned char)n[i])) return n;
        return "b" + std::to_string(atoi(n.c_str() + 1) + (n[0] == 's' ? kleinND : 0)) + n.substr(dot);
    };
    for (auto &t : tis) {
        if (skip && skip(t.name)) continue;
        auto k = kinds.find(t.name);
        if (k == kinds.end()) { fprintf(stderr, "GGUF tensor %s not in ugpu.index\n", t.name.c_str()); exit(1); }
        const size_t es = t.type == 0 ? 4 : t.type == 1 ? 2 : t.type == 24 ? 1 : 0;   // F32, F16, I8
        if (!es) { fprintf(stderr, "GGUF tensor %s: type %u not supported\n", t.name.c_str(), t.type); exit(1); }
        Wt w{std::get<0>(k->second), std::get<1>(k->second), std::get<2>(k->second), nullptr};
        const char *src = base + data + t.off;
        t.name = alias(t.name);
        if (streamedName(t.name, w.kind)) { SW[t.name] = {w.kind, w.M, w.K, src, t.n * es}; continue; }
        if (w.kind == "pk") w.m = img(w.M / 4, w.K / 2, src);
        else if (w.kind == "q8") w.m = q8img(w.M, w.K, src);
        else if (w.kind == "q4") w.m = q8img(w.M, w.K, src, 32);
        else w.m = buf(t.n * es, src);
        W[t.name] = w;
    }
    if (SW.empty()) { munmap((void *)base, st.st_size); close(fd); }
    else printf("streaming %zu block tables from the file\n", SW.size());
    return model;
}
// copy a streamed table into a chip object made earlier for a table of the same kind and shape (no new memory)
static void refill(cl_command_queue lq, const SInfo &s, cl_mem m) {
    if (s.kind == "q8" || s.kind == "q4") {
        std::vector<char> t; int w, h;
        const char *d = q8layout(s.M, s.K, s.src, s.kind == "q4" ? 32 : 16, t, w, h);
        const size_t o[3] = {0, 0, 0}, r[3] = {(size_t)w, (size_t)h, 1};
        CK(clEnqueueWriteImage(lq, m, CL_TRUE, o, r, (size_t)w * 16, 0, d, 0, 0, 0));
    } else CK(clEnqueueWriteBuffer(lq, m, CL_TRUE, 0, s.bytes, s.src, 0, 0, 0));
}
// ask the system to start reading block b's tables from storage now, in big pieces (page cache only: the phone
// can drop it again under memory pressure, so this never risks running out of memory)
static bool prefetchBlock(int b) {
    const std::string pre = "b" + std::to_string(b) + ".";
    const size_t pg = 4096; bool any = false;
    for (auto it = SW.lower_bound(pre); it != SW.end() && it->first.compare(0, pre.size(), pre) == 0; ++it) {
        const uintptr_t a = (uintptr_t)it->second.src & ~(pg - 1), e = (uintptr_t)it->second.src + it->second.bytes;
        madvise((void *)a, e - a, MADV_WILLNEED); any = true;
    }
    return any;
}
// one block's streamed tables as chip objects (callable from a helper thread; does not touch W). The tables are
// made on parallel threads, and the block after this one is already being read from storage.
static std::map<std::string, Wt> loadBlock(int b) {
    if (!prefetchBlock(b + 1)) prefetchBlock(0);
    const std::string pre = "b" + std::to_string(b) + ".";
    std::vector<std::pair<std::string, std::future<Wt>>> jobs;
    for (auto it = SW.lower_bound(pre); it != SW.end() && it->first.compare(0, pre.size(), pre) == 0; ++it) {
        const SInfo *s = &it->second;
        jobs.emplace_back(it->first, std::async(std::launch::async, [s] {
            Wt w{s->kind, s->M, s->K, nullptr};
            if (s->kind == "q8") w.m = q8img(s->M, s->K, s->src);
            else if (s->kind == "q4") w.m = q8img(s->M, s->K, s->src, 32);
            else w.m = buf(s->bytes, s->src);
            return w;
        }));
    }
    std::map<std::string, Wt> out;
    for (auto &j : jobs) out[j.first] = j.second.get();
    return out;
}
static bool has(const std::string &n) { return W.count(n) != 0; }
static cl_mem w(const std::string &n) {
    auto it = W.find(n);
    if (it == W.end()) { fprintf(stderr, "no weight %s\n", n.c_str()); exit(1); }
    return it->second.m;
}

// ---- start-up ----
static void clInit() {
    cl_platform_id pf; CK(clGetPlatformIDs(1, &pf, 0)); CK(clGetDeviceIDs(pf, CL_DEVICE_TYPE_GPU, 1, &dev, 0));
    char dn[256]; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof dn, dn, 0);

    // GPU_PRIO=low (default; "normal" to switch off): ask for low priority where the driver offers it
    // (cl_khr_priority_hints), so the screen's own drawing goes first; other chips simply ignore the request
    char ext[8192] = {0}; clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, sizeof ext - 1, ext, 0);
    // Qualcomm's own hints, set on the context: low priority (the screen's drawing goes first) and, optionally,
    // GPU_PERF=low|normal|high (the clock the driver aims for; low = cooler and slower). Other chips skip these.
    const char *prq = getenv("GPU_PRIO") ? getenv("GPU_PRIO") : "low";
    std::vector<cl_context_properties> cp;
    if (strstr(ext, "cl_qcom_priority_hint") && strcmp(prq, "normal") != 0) {
        cp.push_back(0x40C9);  // CL_CONTEXT_PRIORITY_HINT_QCOM
        cp.push_back(strcmp(prq, "high") == 0 ? 0x40CA : 0x40CC);  // HIGH / LOW
    }
    if (strstr(ext, "cl_qcom_perf_hint") && getenv("GPU_PERF")) {
        const char *pf = getenv("GPU_PERF");
        cp.push_back(0x40C2);  // CL_CONTEXT_PERF_HINT_QCOM
        cp.push_back(strcmp(pf, "low") == 0 ? 0x40C5 : strcmp(pf, "high") == 0 ? 0x40C3 : 0x40C4);
    }
    cp.push_back(0);
    cl_int err; ctx = clCreateContext(cp.size() > 1 ? cp.data() : nullptr, 1, &dev, 0, 0, &err); CK(err);
    const bool qlow = cp.size() > 1 && cp[0] == 0x40C9 && cp[1] == 0x40CC;
    const char *pr = getenv("GPU_PRIO") ? getenv("GPU_PRIO") : "low";
    if (getenv("CLEXT")) printf("extensions: %s\n", ext);
#ifndef __aarch64__
    if (!strstr(ext, "cl_khr_integer_dot_product")) KEXTRA += " -DNO_IDP=1";   // computer test builds only
#endif
    const bool low = strcmp(pr, "low") == 0 && strstr(ext, "cl_khr_priority_hints");
    cl_queue_properties qp[] = {CL_QUEUE_PROPERTIES, prof ? (cl_queue_properties)CL_QUEUE_PROFILING_ENABLE : 0,
                                low ? (cl_queue_properties)CL_QUEUE_PRIORITY_KHR : 0, low ? (cl_queue_properties)CL_QUEUE_PRIORITY_LOW_KHR : 0, 0};
    q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err);
    if (err && low) { qp[2] = 0; q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err); }
    CK(err);
    printf("%s, %s priority%s%s\n", dn, (low && qp[2]) || qlow ? "low" : "normal", getenv("GPU_PERF") ? ", clock hint " : "",
           getenv("GPU_PERF") ? getenv("GPU_PERF") : "");
}

// ---- first-run tuning: time each candidate multiply on this chip, check its answers on the processor, keep the
// fastest correct one in a small file next to the weights. Drivers differ: a kernel can be fast, slow, or (seen on
// Adreno with a fixed group-size attribute) silently skipped, so every candidate's output is checked.
struct GemmCfg { int local, a, b, c, d; };  // local: tokens/item, groups of 4 outputs, group size, chunk; tile: TN, TM, L0, L1
static void applyCfg(const GemmCfg &g) {
    LOCALGEMM = g.local;
    if (g.local) { TNI = g.a; GG = g.b; WGS = g.c; KCS = g.d; } else { TN = g.a; TM = g.b; L0 = g.c; L1 = g.d; }
}
static void tune(const std::string &src, const std::string &cacheFile) {
    char dn[256], dv[256];
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof dn, dn, 0);
    clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof dv, dv, 0);
    std::string key = std::string(dn) + " | " + dv;
    if (FILE *f = fopen(cacheFile.c_str(), "r")) {
        char line[512]; GemmCfg g;
        if (fgets(line, sizeof line, f) && std::string(line) == key + "\n" &&
            fscanf(f, "%d %d %d %d %d", &g.local, &g.a, &g.b, &g.c, &g.d) == 5) {
            fclose(f); applyCfg(g); printf("tuning: cached\n"); return;
        }
        fclose(f);
    }
    const int n = 1024, K = 2048, M = 2048;
    std::vector<h16> xh((size_t)n * K), wh((size_t)M * K);
    srand(7);
    for (auto &v : xh) v = (h16)((rand() % 2001 - 1000) / 1000.0f);
    for (auto &v : wh) v = (h16)((rand() % 2001 - 1000) / 1000.0f / 45.0f);
    // X layout: texel (n, k8); W layout: texel (og, k2) = w[4og..+3][2k2], w[4og..+3][2k2+1]
    std::vector<h16> xi((size_t)n * K), wi((size_t)M * K);
    for (int k8 = 0; k8 < K / 8; k8++) for (int i = 0; i < n; i++) for (int j = 0; j < 8; j++)
        xi[((size_t)k8 * n + i) * 8 + j] = xh[(size_t)i * K + k8 * 8 + j];
    for (int k2 = 0; k2 < K / 2; k2++) for (int og = 0; og < M / 4; og++) for (int j = 0; j < 4; j++) {
        wi[((size_t)k2 * (M / 4) + og) * 8 + j] = wh[(size_t)(og * 4 + j) * K + 2 * k2];
        wi[((size_t)k2 * (M / 4) + og) * 8 + 4 + j] = wh[(size_t)(og * 4 + j) * K + 2 * k2 + 1];
    }
    cl_mem X = img(n, K / 8, xi.data()), Wm = img(M / 4, K / 2, wi.data()), Y = img(n, M / 8);
    cl_int err;
    cl_queue_properties qp[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue tq = clCreateCommandQueueWithProperties(ctx, dev, qp, &err); CK(err);
    std::vector<GemmCfg> cands = {
        {1, 2, 4, 128, 64}, {1, 2, 4, 64, 64}, {1, 1, 4, 128, 64}, {1, 2, 2, 128, 64}, {1, 1, 8, 128, 64},
        {1, 2, 4, 256, 64}, {1, 2, 8, 128, 64}, {1, 4, 2, 64, 64},
        {0, 6, 2, 16, 8}, {0, 4, 2, 16, 8}, {0, 8, 2, 16, 8}, {0, 4, 4, 8, 8}};
    GemmCfg best{}; double bestMs = 1e9;
    for (auto &g : cands) {
        char o[256];
        snprintf(o, sizeof o, "-DEPI=0 -DTN=%d -DTM=%d -DTNI=%d -DG=%d -DWG=%d -DKC=%d", g.local ? 6 : g.a, g.local ? 2 : g.b,
                 g.local ? g.a : 2, g.local ? g.b : 4, g.local ? g.c : 128, g.local ? g.d : 64);
        const char *s = src.c_str();
        cl_program p = clCreateProgramWithSource(ctx, 1, &s, 0, &err); CK(err);
        if (clBuildProgram(p, 1, &dev, (std::string(o) + KEXTRA).c_str(), 0, 0)) {   // the options real runs use
            static bool told = false;
            if (!told) {   // say why once: a kernel source that no longer builds alone would fail every candidate
                size_t n2 = 0; clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, 0, &n2);
                std::string lg(n2, 0); clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, n2, &lg[0], 0);
                fprintf(stderr, "tuning: a candidate did not build: %.600s\n", lg.c_str()); told = true;
            }
            clReleaseProgram(p); continue;
        }
        cl_kernel k = clCreateKernel(p, g.local ? "gemml" : "gemm", &err);
        if (err) { clReleaseProgram(p); continue; }
        int z = 0;
        arg(k, 0, X); arg(k, 1, Wm); arg(k, 2, n); arg(k, 3, M); arg(k, 4, K);
        arg(k, 5, z); arg(k, 6, z); arg(k, 7, z); arg(k, 8, z); arg(k, 9, z); arg(k, 10, z); arg(k, 11, z); arg(k, 12, Y);
        arg(k, 13, n);
        size_t lws[3], gws[3];
        if (g.local) { lws[0] = g.c; lws[1] = 1; gws[0] = ((n + g.a - 1) / g.a + g.c - 1) / g.c * g.c; gws[1] = M / 4 / g.b; }
        else { lws[0] = g.c; lws[1] = g.d; gws[0] = (M / 4 / g.b + g.c - 1) / g.c * g.c; gws[1] = ((n + g.a - 1) / g.a + g.d - 1) / g.d * g.d; }
        lws[2] = gws[2] = 1;
        std::vector<double> t;
        bool ok = true;
        for (int r = 0; r < 4 && ok; r++) {
            cl_event ev;
            if (clEnqueueNDRangeKernel(tq, k, 3, 0, gws, lws, 0, 0, &ev) != CL_SUCCESS) { ok = false; break; }
            clWaitForEvents(1, &ev);
            cl_ulong a, b;
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, 8, &a, 0);
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, 8, &b, 0);
            clReleaseEvent(ev);
            if (r) t.push_back((b - a) * 1e-6);
            if (r == 0 || (b - a) * 1e-6 > 4 * bestMs) { if (r) break; }
        }
        double err2 = 1;
        if (ok && !t.empty()) {
            std::vector<h16> y((size_t)n * M);
            size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)n, (size_t)M / 8, 1};
            clEnqueueReadImage(tq, Y, CL_TRUE, org, reg, 0, 0, y.data(), 0, 0, 0);
            double se = 0, sr = 0;
            for (int s2 = 0; s2 < 400; s2++) {
                int i = (s2 * 7919) % n, m = (s2 * 104729) % M; double r = 0;
                for (int kk = 0; kk < K; kk++) r += (double)xh[(size_t)i * K + kk] * wh[(size_t)m * K + kk];
                double v = y[((size_t)(m / 8) * n + i) * 8 + m % 8];
                se += (v - r) * (v - r); sr += r * r;
            }
            err2 = sqrt(se / sr);
            // clear Y so the next candidate cannot pass on this one's answers
            std::vector<h16> zero((size_t)n * M, (h16)0.0f);
            clEnqueueWriteImage(tq, Y, CL_TRUE, org, reg, 0, 0, zero.data(), 0, 0, 0);
        }
        double ms = t.empty() ? 1e9 : *std::min_element(t.begin(), t.end());
        printf("  %s %d %d %d %d: %s %.2f ms  error %.4f\n", g.local ? "shared" : "tile  ", g.a, g.b, g.c, g.d,
               ok ? "" : "FAILED", ms, err2);
        if (ok && err2 < 0.005 && ms < bestMs) { bestMs = ms; best = g; }
        clReleaseKernel(k); clReleaseProgram(p);
    }
    clReleaseCommandQueue(tq); clReleaseMemObject(X); clReleaseMemObject(Wm); clReleaseMemObject(Y);
    if (bestMs >= 1e9) { fprintf(stderr, "tuning: no multiply produced correct answers on this chip\n"); exit(1); }
    applyCfg(best);
    printf("tuning: picked %s %d %d %d %d (%.0f GFLOPS)\n", best.local ? "shared" : "tile", best.a, best.b, best.c, best.d,
           2.0 * n * K * M / bestMs / 1e6);
    if (FILE *f = fopen(cacheFile.c_str(), "w")) {
        fprintf(f, "%s\n%d %d %d %d %d\n", key.c_str(), best.local, best.a, best.b, best.c, best.d); fclose(f);
    }
}

static double relerr(const std::vector<float> &a, const std::vector<float> &b) {
    double e = 0, r = 0;
    for (size_t i = 0; i < a.size(); i++) { e += (a[i] - b[i]) * (double)(a[i] - b[i]); r += b[i] * (double)b[i]; }
    return sqrt(e / r);
}
// host [n][C] floats -> X layout image (C padded to a multiple of 8)
static cl_mem toX(const std::vector<float> &a, int n, int C) {
    int C8 = (C + 7) / 8;
    std::vector<h16> t((size_t)n * C8 * 8, (h16)0.0f);
    for (int k8 = 0; k8 < C8; k8++) for (int i = 0; i < n; i++) for (int j = 0; j < 8; j++) {
        int c = k8 * 8 + j;
        t[((size_t)k8 * n + i) * 8 + j] = (h16)(c < C ? a[(size_t)i * C + c] : 0.0f);
    }
    return img(n, C8, t.data());
}

