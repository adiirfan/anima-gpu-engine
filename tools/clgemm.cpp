// clgemm: try multiply kernels for old Adreno chips (630) on the engine's own layouts, against the engine's gemml.
//   clgemm [N M K] [reps]   env CANDS="mode tni g kc wx localw [wy localx];..." (default: a built-in list)
// wx x wy threads per group: wx share the weights (row blocks), wy share the inputs (output groups). localx 1 stages
// the inputs of the whole group in local memory too, so each texel is fetched once per group instead of once per thread.
// Layouts as kernels.cl: X image RGBA32UI, texel (n, k8) = 8 halves; W image RGBA32UI, texel (og, k2) = outputs
// 4og..+3 at inputs 2k2, 2k2+1. Y (fp32 buffer, n*M + m) is checked against the processor on 256 sampled outputs.
// mode 0: the engine's DOT8 with ACC32=4 SCL=64 (half pairs, fp32 totals); mode 1: every product in fp32 (operands
// converted once per load, shared across the tile); mode 2: half products summed over 8 inputs per output in fp16
// with x scaled by 64, then fp32 (the fast path old drivers got wrong without the scale); mode 3: the same over 4 inputs.
// localw 1: the weights go through local memory as in gemml; 0: read straight from the image (texture cache).
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static cl_context ctx; static cl_command_queue q; static cl_device_id dev;
#define CK(x) do { cl_int e_ = (x); if (e_) { fprintf(stderr, "%s failed: %d (line %d)\n", #x, e_, __LINE__); exit(1); } } while (0)

static const char *SRC = R"(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#define LD8(img, x, y) as_half8(read_imageui(img, smp, (int2)(x, y)))
#define SCL 64
#if LOCALX
#define XLD(i) xl[(k8l * TNI + (i)) * WX + lx]
#else
#define XLD(i) LD8(X, n0 + (i), kc + k8l)
#endif
#if MODE == 0
#define XS(v) ((v) * (half)SCL)
#define UNS (1.0f / SCL)
#define DOT8(x, w0, w1, w2, w3) ( \
    convert_float4(mad((x).s1, (w0).hi, (x).s0 * (w0).lo)) + convert_float4(mad((x).s3, (w1).hi, (x).s2 * (w1).lo)) + \
    convert_float4(mad((x).s5, (w2).hi, (x).s4 * (w2).lo)) + convert_float4(mad((x).s7, (w3).hi, (x).s6 * (w3).lo)))
#elif MODE == 3
#define XS(v) ((v) * (half)SCL)
#define UNS (1.0f / SCL)
#define DOT8(x, w0, w1, w2, w3) ( \
    convert_float4(mad((x).s3, (w1).hi, mad((x).s2, (w1).lo, mad((x).s1, (w0).hi, (x).s0 * (w0).lo)))) + \
    convert_float4(mad((x).s7, (w3).hi, mad((x).s6, (w3).lo, mad((x).s5, (w2).hi, (x).s4 * (w2).lo)))))
#elif MODE == 2
#define XS(v) ((v) * (half)SCL)
#define UNS (1.0f / SCL)
inline float4 DOT8(half8 x, half8 w0, half8 w1, half8 w2, half8 w3) {
    half4 t = x.s0 * w0.lo;
    t = mad(x.s1, w0.hi, t); t = mad(x.s2, w1.lo, t); t = mad(x.s3, w1.hi, t);
    t = mad(x.s4, w2.lo, t); t = mad(x.s5, w2.hi, t); t = mad(x.s6, w3.lo, t);
    return convert_float4(mad(x.s7, w3.hi, t));
}
#endif

__kernel void gemm(__read_only image2d_t X, __read_only image2d_t W, int N, int M, int K, __global float4 *Y) {
    const int lx = get_local_id(0), ly = get_local_id(1), lid = ly * WX + lx;
#if LOCALX
    __local half8 xl[KC / 8 * WX * TNI];   // [k8l][i][lx]: neighbouring threads read neighbouring entries
    const int gn0 = get_group_id(0) * WX * TNI;
#endif
#if LOCALW
    __local half8 wl[KC / 8 * WY * G * 4];
    const int gw0 = get_group_id(1) * WY * G;
#endif
    const int nb = get_global_id(0), mb = get_global_id(1);
    const int n0 = nb * TNI, wx = mb * G;
    float4 r[TNI][G];
    #pragma unroll
    for (int i = 0; i < TNI; i++)
#pragma unroll
        for (int g = 0; g < G; g++) r[i][g] = 0;
    for (int kc = 0; kc < K / 8; kc += KC / 8) {
#if LOCALX || LOCALW
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if LOCALX
        for (int idx = lid; idx < KC / 8 * WX * TNI; idx += WX * WY) {
            const int k8l = idx / (WX * TNI), j = idx % (WX * TNI), i = j / WX, t = j % WX;
            xl[idx] = LD8(X, gn0 + t * TNI + i, kc + k8l);
        }
#endif
#if LOCALW
        for (int idx = lid; idx < KC / 8 * WY * G * 4; idx += WX * WY) {
            const int k8l = idx / (WY * G * 4), rr = idx % (WY * G * 4);
            wl[idx] = LD8(W, gw0 + rr / 4, (kc + k8l) * 4 + (rr & 3));
        }
#endif
#if LOCALX || LOCALW
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        for (int k8l = 0; k8l < KC / 8; k8l++) {
#if MODE == 1
            float8 x[TNI];
            #pragma unroll
            for (int i = 0; i < TNI; i++) x[i] = convert_float8(XLD(i));
#else
            half8 x[TNI];
            #pragma unroll
            for (int i = 0; i < TNI; i++) x[i] = XS(XLD(i));
#endif
            #pragma unroll
            for (int g = 0; g < G; g++) {
#if LOCALW
                const int bi = (k8l * WY * G + ly * G + g) * 4;
                const half8 w0 = wl[bi], w1 = wl[bi + 1], w2 = wl[bi + 2], w3 = wl[bi + 3];
#else
                const int y = (kc + k8l) * 4;
                const half8 w0 = LD8(W, wx + g, y), w1 = LD8(W, wx + g, y + 1), w2 = LD8(W, wx + g, y + 2), w3 = LD8(W, wx + g, y + 3);
#endif
#if MODE == 1
                const float8 f0 = convert_float8(w0), f1 = convert_float8(w1), f2 = convert_float8(w2), f3 = convert_float8(w3);
                #pragma unroll
                for (int i = 0; i < TNI; i++) {
                    float4 a = r[i][g];
                    a = mad((float4)x[i].s0, f0.lo, a); a = mad((float4)x[i].s1, f0.hi, a);
                    a = mad((float4)x[i].s2, f1.lo, a); a = mad((float4)x[i].s3, f1.hi, a);
                    a = mad((float4)x[i].s4, f2.lo, a); a = mad((float4)x[i].s5, f2.hi, a);
                    a = mad((float4)x[i].s6, f3.lo, a); a = mad((float4)x[i].s7, f3.hi, a);
                    r[i][g] = a;
                }
#else
                #pragma unroll
                for (int i = 0; i < TNI; i++) r[i][g] += DOT8(x[i], w0, w1, w2, w3);
#endif
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < TNI; i++)
#pragma unroll
    for (int g = 0; g < G; g++) {
#ifdef UNS
        r[i][g] *= UNS;
#endif
        const int n = n0 + i, og = mb * G + g;
        if (n < N) Y[(n * M) / 4 + og] = r[i][g];
    }
}
)";

static uint16_t toh(float f) {   // fp32 -> fp16 (round to nearest)
    uint32_t x; memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000; int e = ((x >> 23) & 255) - 127 + 15; uint32_t m = x & 0x7fffff;
    if (e <= 0) return s; if (e >= 31) return s | 0x7c00;
    uint32_t h = s | (e << 10) | (m >> 13); if (m & 0x1000) h++; return h;
}
static float fromh(uint16_t h) {
    const uint32_t s = (h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023;
    uint32_t x = e == 0 ? s : s | ((e - 15 + 127) << 23) | (m << 13); float f; memcpy(&f, &x, 4); return f;
}

int main(int argc, char **argv) {
    const int N = argc > 3 ? atoi(argv[1]) : 1024, M = argc > 3 ? atoi(argv[2]) : 2048, K = argc > 3 ? atoi(argv[3]) : 2048;
    const int reps = argc > 4 ? atoi(argv[4]) : 3;
    cl_platform_id pl; CK(clGetPlatformIDs(1, &pl, nullptr));
    CK(clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr));
    cl_int e; ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e); CK(e);
    {   // low priority (as the engine): on Adreno 630 a kernel that never ends then gets killed instead of freezing the screen
        cl_queue_properties qp[] = {CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_LOW_KHR, 0};
        q = clCreateCommandQueueWithProperties(ctx, dev, qp, &e);
        if (e) q = clCreateCommandQueue(ctx, dev, 0, &e);
        CK(e);
    }

    std::mt19937 rg(1); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> xh((size_t)N * K), wh((size_t)M * K);   // X row-major [n][k], W row-major [m][k]
    for (auto &v : xh) v = toh(nd(rg));
    for (auto &v : wh) v = toh(nd(rg) * 0.02f);
    // W image: texel (og, k2), width M/4, height K/2
    std::vector<uint16_t> wimg((size_t)M * K);
    for (int og = 0; og < M / 4; og++) for (int k2 = 0; k2 < K / 2; k2++) {
        uint16_t *tx = &wimg[((size_t)k2 * (M / 4) + og) * 8];
        // half8: lo = outputs 4og..+3 at input 2k2, hi = same outputs at input 2k2+1
        for (int j = 0; j < 4; j++) { tx[j] = wh[(size_t)(4 * og + j) * K + 2 * k2]; tx[4 + j] = wh[(size_t)(4 * og + j) * K + 2 * k2 + 1]; }
    }
    cl_image_format f{CL_RGBA, CL_UNSIGNED_INT32}; cl_image_desc dx{}, dw{};
    dx.image_type = dw.image_type = CL_MEM_OBJECT_IMAGE2D;
    dx.image_width = N; dx.image_height = K / 8; dw.image_width = M / 4; dw.image_height = K / 2;
    std::vector<uint16_t> ximg((size_t)N * K);   // texel (n, k8) = x[n][8k8 .. 8k8+7], row k8
    for (int k8 = 0; k8 < K / 8; k8++) for (int n = 0; n < N; n++) for (int j = 0; j < 8; j++)
        ximg[((size_t)k8 * N + n) * 8 + j] = xh[(size_t)n * K + 8 * k8 + j];
    cl_mem X = clCreateImage(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &f, &dx, ximg.data(), &e); CK(e);
    cl_mem W = clCreateImage(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &f, &dw, wimg.data(), &e); CK(e);
    cl_mem Y = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)N * M * 4, nullptr, &e); CK(e);

    std::uniform_int_distribution<int> un(0, N - 1), um(0, M - 1);
    std::vector<std::pair<int, int>> pts(256); std::vector<double> ref(256);
    for (int s = 0; s < 256; s++) {
        pts[s] = {un(rg), um(rg)}; double a = 0;
        for (int k = 0; k < K; k++) a += (double)fromh(xh[(size_t)pts[s].first * K + k]) * fromh(wh[(size_t)pts[s].second * K + k]);
        ref[s] = a;
    }
    double refrms = 0; for (double v : ref) refrms += v * v; refrms = sqrt(refrms / 256);

    if (const char *ef = getenv("ENGINE")) {   // ENGINE=<kernels.cl> EOPTS="<options>": run that file's gemml (the engine's args)
        FILE *fp = fopen(ef, "rb"); if (!fp) { perror(ef); return 1; }
        std::string src; char b[65536]; size_t nr; while ((nr = fread(b, 1, sizeof b, fp)) > 0) src.append(b, nr); fclose(fp);
        const char *s = src.c_str(); cl_program pr = clCreateProgramWithSource(ctx, 1, &s, nullptr, &e); CK(e);
        const char *eo = getenv("EOPTS") ? getenv("EOPTS") : "";
        if (clBuildProgram(pr, 1, &dev, eo, nullptr, nullptr)) {
            char lg[8192] = ""; clGetProgramBuildInfo(pr, dev, CL_PROGRAM_BUILD_LOG, sizeof lg, lg, nullptr); printf("did not build: %.500s\n", lg); return 1;
        }
        cl_kernel k = clCreateKernel(pr, "gemml", &e); CK(e);
        cl_image_desc dy{}; dy.image_type = CL_MEM_OBJECT_IMAGE2D; dy.image_width = N; dy.image_height = M / 8;
        cl_mem YI = clCreateImage(ctx, CL_MEM_READ_WRITE, &f, &dy, nullptr, &e); CK(e);
        int z = 0, a = 0; const int TNIe = getenv("ETNI") ? atoi(getenv("ETNI")) : 4, Ge = getenv("EG") ? atoi(getenv("EG")) : 2,
                                   WGe = getenv("EWG") ? atoi(getenv("EWG")) : 128;
        CK(clSetKernelArg(k, a++, sizeof X, &X)); CK(clSetKernelArg(k, a++, sizeof W, &W)); CK(clSetKernelArg(k, a++, 4, &N));
        CK(clSetKernelArg(k, a++, 4, &M)); CK(clSetKernelArg(k, a++, 4, &K));
        for (int i = 0; i < 7; i++) CK(clSetKernelArg(k, a++, 4, &z));
        CK(clSetKernelArg(k, a++, sizeof YI, &YI)); CK(clSetKernelArg(k, a++, 4, &N));
        size_t gl[3] = {(size_t)((N / TNIe + WGe - 1) / WGe * WGe), (size_t)(M / 4 / Ge), 1}, lo[3] = {(size_t)WGe, 1, 1};
        auto t0 = std::chrono::steady_clock::now();
        cl_int r1 = clEnqueueNDRangeKernel(q, k, 3, nullptr, gl, lo, 0, nullptr, nullptr), r2 = clFinish(q);
        const double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
        if (r1 || r2) { printf("engine gemml: enqueue %d finish %d\n", r1, r2); return 1; }
        std::vector<uint16_t> y((size_t)N * M); size_t org[3] = {0, 0, 0}, reg[3] = {(size_t)N, (size_t)M / 8, 1};
        CK(clEnqueueReadImage(q, YI, CL_TRUE, org, reg, 0, 0, y.data(), 0, nullptr, nullptr));
        double err = 0; for (int s2 = 0; s2 < 256; s2++) {
            const int i = pts[s2].first, m = pts[s2].second; double d = fromh(y[((size_t)(m / 8) * N + i) * 8 + m % 8]) - ref[s2]; err += d * d; }
        printf("engine gemml [%s]: %.2f ms (first run), error %.2e\n", eo, ms, sqrt(err / 256) / refrms);
        return 0;
    }
    std::string cands = getenv("CANDS") ? getenv("CANDS") :
        "0 4 2 128 512 1;0 4 2 64 256 0;1 4 2 64 256 0;1 4 4 64 256 0;1 8 2 64 256 0;2 4 2 64 256 0;2 4 4 64 256 0;0 4 4 64 256 0";
    printf("%d x %d x %d (%.1f GFLOP)\n", N, M, K, 2.0 * N * M * K / 1e9);
    size_t p0 = 0;
    while (p0 < cands.size()) {
        size_t p1 = cands.find(';', p0); if (p1 == std::string::npos) p1 = cands.size();
        int mode, tni, g, kc, wg, lw, wy = 1, lxx = 0;
        const int nf = sscanf(cands.substr(p0, p1 - p0).c_str(), "%d %d %d %d %d %d %d %d", &mode, &tni, &g, &kc, &wg, &lw, &wy, &lxx);
        if (nf != 6 && nf != 8) break;
        p0 = p1 + 1;
        char o[256]; snprintf(o, sizeof o, "-DMODE=%d -DTNI=%d -DG=%d -DKC=%d -DWX=%d -DWY=%d -DLOCALW=%d -DLOCALX=%d", mode, tni, g, kc, wg, wy, lw, lxx);
        const char *s = SRC; cl_program pr = clCreateProgramWithSource(ctx, 1, &s, nullptr, &e); CK(e);
        if (clBuildProgram(pr, 1, &dev, o, nullptr, nullptr)) {
            char lg[4096] = ""; clGetProgramBuildInfo(pr, dev, CL_PROGRAM_BUILD_LOG, sizeof lg, lg, nullptr);
            printf("%-40s did not build: %.300s\n", o, lg); continue;
        }
        cl_kernel k = clCreateKernel(pr, "gemm", &e); CK(e);
        CK(clSetKernelArg(k, 0, sizeof X, &X)); CK(clSetKernelArg(k, 1, sizeof W, &W)); CK(clSetKernelArg(k, 2, 4, &N));
        CK(clSetKernelArg(k, 3, 4, &M)); CK(clSetKernelArg(k, 4, 4, &K)); CK(clSetKernelArg(k, 5, sizeof Y, &Y));
        size_t gl[2] = {(size_t)((N / tni + wg - 1) / wg * wg), (size_t)(M / 4 / g)}, lo[2] = {(size_t)wg, (size_t)wy};
        if ((M / 4 / g) % wy) { printf("%-40s does not fit\n", o); continue; }
        if (clEnqueueNDRangeKernel(q, k, 2, nullptr, gl, lo, 0, nullptr, nullptr) || clFinish(q)) { printf("%-40s failed to run\n", o); continue; }
        double best = 1e30; const auto tstart = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; r++) {
            auto t0 = std::chrono::steady_clock::now();
            CK(clEnqueueNDRangeKernel(q, k, 2, nullptr, gl, lo, 0, nullptr, nullptr)); CK(clFinish(q));
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            if (getenv("TRACE") && (r % 10 == 0 || r == reps - 1))   // TRACE=1: every 10th run's time, to see slowdowns under load
                printf("  run %4d at %5.1f s: %.2f ms\n", r, std::chrono::duration<double>(std::chrono::steady_clock::now() - tstart).count(),
                       std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3);
        }
        std::vector<float> y((size_t)N * M); CK(clEnqueueReadBuffer(q, Y, CL_TRUE, 0, y.size() * 4, y.data(), 0, nullptr, nullptr));
        double err = 0; for (int s2 = 0; s2 < 256; s2++) { double d = y[(size_t)pts[s2].first * M + pts[s2].second] - ref[s2]; err += d * d; }
        printf("%-48s %8.2f ms  %4.0f GFLOPS  error %.2e\n", o, best * 1e3, 2.0 * N * M * K / best / 1e9, sqrt(err / 256) / refrms);
        clReleaseKernel(k); clReleaseProgram(pr);
    }
}
