// clpeak: what a phone's graphics chip can really do through OpenCL, to judge our kernels against.
//   fp32 / fp16 multiply-add rate (long independent chains, no memory), and read bandwidth from a plain buffer and
//   from an image (the texture path our kernels use).
// Build: aarch64-linux-android28-clang++ -O2 -std=c++17 -static-libstdc++ -Ithird_party/OpenCLHeaders tools/clpeak.cpp
//        -o clpeak -Lthird_party/lib -lOpenCL
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static cl_context ctx; static cl_command_queue q; static cl_device_id dev;
#define CK(x) do { cl_int e_ = (x); if (e_) { fprintf(stderr, "%s failed: %d (line %d)\n", #x, e_, __LINE__); exit(1); } } while (0)

static cl_kernel build(const std::string &src, const char *name, const char *opts = "") {
    const char *s = src.c_str(); cl_int e;
    cl_program p = clCreateProgramWithSource(ctx, 1, &s, nullptr, &e); CK(e);
    if (clBuildProgram(p, 1, &dev, opts, nullptr, nullptr)) {
        char log[8192]; clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, nullptr);
        fprintf(stderr, "build failed: %s\n", log); exit(1);
    }
    cl_kernel k = clCreateKernel(p, name, &e); CK(e); return k;
}
static double run(cl_kernel k, size_t g, size_t l, int reps) {
    CK(clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, &l, 0, nullptr, nullptr)); CK(clFinish(q));   // warm-up
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
        CK(clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, &l, 0, nullptr, nullptr)); CK(clFinish(q));
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

// each item: V-wide vectors, 8 independent chains, ITER x 8 multiply-adds per chain element
static const char *FMA = R"(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef HALF
typedef half T; typedef half4 T4;
#else
typedef float T; typedef float4 T4;
#endif
__kernel void fma_peak(__global T *out, T a, T b) {
    T4 x0 = (T4)(get_global_id(0) * 1e-6f), x1 = x0 + (T)0.1f, x2 = x0 + (T)0.2f, x3 = x0 + (T)0.3f,
       x4 = x0 + (T)0.4f, x5 = x0 + (T)0.5f, x6 = x0 + (T)0.6f, x7 = x0 + (T)0.7f;
    for (int i = 0; i < ITER; i++) {
        x0 = mad(x0, a, b); x1 = mad(x1, a, b); x2 = mad(x2, a, b); x3 = mad(x3, a, b);
        x4 = mad(x4, a, b); x5 = mad(x5, a, b); x6 = mad(x6, a, b); x7 = mad(x7, a, b);
        x0 = mad(x0, a, b); x1 = mad(x1, a, b); x2 = mad(x2, a, b); x3 = mad(x3, a, b);
        x4 = mad(x4, a, b); x5 = mad(x5, a, b); x6 = mad(x6, a, b); x7 = mad(x7, a, b);
    }
    T4 s = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
    out[get_global_id(0)] = s.x + s.y + s.z + s.w;
}
__kernel void fma_peak8(__global float *out, float af, float bf) {
#ifdef HALF
    typedef half8 T8; const half a = (half)af, b = (half)bf;
#else
    typedef float8 T8; const float a = af, b = bf;
#endif
    T8 x0 = (T8)(get_global_id(0) * 1e-6f), x1 = x0 + 0.1f, x2 = x0 + 0.2f, x3 = x0 + 0.3f;
    for (int i = 0; i < ITER; i++) {
        x0 = mad(x0, a, b); x1 = mad(x1, a, b); x2 = mad(x2, a, b); x3 = mad(x3, a, b);
        x0 = mad(x0, a, b); x1 = mad(x1, a, b); x2 = mad(x2, a, b); x3 = mad(x3, a, b);
        x0 = mad(x0, a, b); x1 = mad(x1, a, b); x2 = mad(x2, a, b); x3 = mad(x3, a, b);
        x0 = mad(x0, a, b); x1 = mad(x1, a, b); x2 = mad(x2, a, b); x3 = mad(x3, a, b);
    }
    T8 s = x0 + x1 + x2 + x3;
    out[get_global_id(0)] = (float)(s.s0 + s.s1 + s.s2 + s.s3 + s.s4 + s.s5 + s.s6 + s.s7);
}
__kernel void rd_buf(__global const float4 *in, __global float *out, int n4) {
    float4 s = 0; for (int i = get_global_id(0); i < n4; i += get_global_size(0)) s += in[i];
    out[get_global_id(0)] = s.x + s.y + s.z + s.w;
}
__kernel void rd_img(__read_only image2d_t im, __global float *out, int w, int h) {
    const sampler_t sm = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    float4 s = 0; const int g = get_global_id(0), G = get_global_size(0);
    for (int i = g; i < w * h; i += G) s += read_imagef(im, sm, (int2)(i % w, i / w));
    out[g] = s.x + s.y + s.z + s.w;
}
)";

int main() {
    cl_platform_id pl; CK(clGetPlatformIDs(1, &pl, nullptr));
    CK(clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr));
    char name[256]; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof name, name, nullptr);
    cl_uint cu, mhz; clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof mhz, &mhz, nullptr);
    printf("%s: %u compute units, %u MHz reported\n", name, cu, mhz);
    cl_int e; ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e); CK(e);
    q = clCreateCommandQueue(ctx, dev, 0, &e); CK(e);

    const int ITER = 256; const size_t G = 64 * 1024, L = 128;
    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, G * 4, nullptr, &e); CK(e);
    for (int half = 0; half < 2; half++) {
        std::string o = "-DITER=" + std::to_string(ITER) + (half ? " -DHALF" : "") + " -cl-fast-relaxed-math";
        cl_kernel k = build(FMA, "fma_peak", o.c_str());
        CK(clSetKernelArg(k, 0, sizeof out, &out));
        if (half) { cl_half a = 0x3bff, b = 0x1000; CK(clSetKernelArg(k, 1, 2, &a)); CK(clSetKernelArg(k, 2, 2, &b)); }
        else { float a = 0.9999f, b = 1e-4f; CK(clSetKernelArg(k, 1, 4, &a)); CK(clSetKernelArg(k, 2, 4, &b)); }
        const double t = run(k, G, L, 5);
        const double flops = (double)G * ITER * 16 * 4 * 2;
        printf("%s multiply-add: %.0f GFLOPS (%.2f ms)\n", half ? "fp16" : "fp32", flops / t / 1e9, t * 1e3);
    }
    for (int half = 0; half < 2; half++) {
        std::string o = "-DITER=" + std::to_string(ITER) + (half ? " -DHALF" : "") + " -cl-fast-relaxed-math";
        cl_kernel k = build(FMA, "fma_peak8", o.c_str());
        float a = 0.9999f, b = 1e-4f;
        CK(clSetKernelArg(k, 0, sizeof out, &out)); CK(clSetKernelArg(k, 1, 4, &a)); CK(clSetKernelArg(k, 2, 4, &b));
        const double t = run(k, G, L, 5);
        printf("%s multiply-add, 8-wide: %.0f GFLOPS (%.2f ms)\n", half ? "fp16" : "fp32", (double)G * ITER * 16 * 8 * 2 / t / 1e9, t * 1e3);
    }
    const size_t MB = 256, n4 = MB * 1024 * 1024 / 16;
    cl_mem big = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n4 * 16, nullptr, &e); CK(e);
    { std::vector<float> z(n4 * 4, 1.f); CK(clEnqueueWriteBuffer(q, big, CL_TRUE, 0, n4 * 16, z.data(), 0, nullptr, nullptr)); }
    cl_kernel kb = build(FMA, "rd_buf", "-DITER=1");
    int n4i = (int)n4; CK(clSetKernelArg(kb, 0, sizeof big, &big)); CK(clSetKernelArg(kb, 1, sizeof out, &out)); CK(clSetKernelArg(kb, 2, 4, &n4i));
    double t = run(kb, G, L, 5);
    printf("read, plain buffer: %.1f GB/s\n", n4 * 16 / t / 1e9);
    const int W = 4096, H = (int)(n4 / W);
    cl_image_format f{CL_RGBA, CL_HALF_FLOAT}; cl_image_desc d{}; d.image_type = CL_MEM_OBJECT_IMAGE2D; d.image_width = W; d.image_height = H;
    cl_mem im = clCreateImage(ctx, CL_MEM_READ_ONLY, &f, &d, nullptr, &e); CK(e);
    { std::vector<uint16_t> z((size_t)W * H * 4, 0x3c00); size_t o[3] = {0, 0, 0}, r[3] = {(size_t)W, (size_t)H, 1};
      CK(clEnqueueWriteImage(q, im, CL_TRUE, o, r, 0, 0, z.data(), 0, nullptr, nullptr)); }
    cl_kernel ki = build(FMA, "rd_img", "-DITER=1");
    CK(clSetKernelArg(ki, 0, sizeof im, &im)); CK(clSetKernelArg(ki, 1, sizeof out, &out)); CK(clSetKernelArg(ki, 2, 4, &W)); CK(clSetKernelArg(ki, 3, 4, &H));
    t = run(ki, G, L, 5);
    printf("read, half-float image: %.1f GB/s\n", (double)W * H * 8 / t / 1e9);
}
