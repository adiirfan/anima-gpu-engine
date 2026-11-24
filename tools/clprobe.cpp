// clprobe: build a kernel file with given options and print what the driver says each named kernel needs
//   clprobe <file.cl> "<build options>" kernel [kernel...]
// (group size it allows, private memory = register spill, local memory). For chasing driver trouble on old phones.
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <cstdio>
#include <string>
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: clprobe file.cl \"options\" kernel...\n"); return 1; }
    FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 1; }
    std::string src; char b[65536]; size_t n; while ((n = fread(b, 1, sizeof b, f)) > 0) src.append(b, n); fclose(f);
    cl_platform_id pl; cl_device_id dev; clGetPlatformIDs(1, &pl, nullptr); clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
    cl_int e; cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e);
    const char *s = src.c_str(); cl_program p = clCreateProgramWithSource(ctx, 1, &s, nullptr, &e);
    if (clBuildProgram(p, 1, &dev, argv[2], nullptr, nullptr)) {
        char lg[16384] = ""; clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, sizeof lg, lg, nullptr); printf("build failed: %s\n", lg); return 1;
    }
    for (int i = 3; i < argc; i++) {
        cl_kernel k = clCreateKernel(p, argv[i], &e); if (e) { printf("%s: not found (%d)\n", argv[i], e); continue; }
        size_t wg = 0; cl_ulong pm = 0, lm = 0;
        clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof wg, &wg, nullptr);
        clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof pm, &pm, nullptr);
        clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_LOCAL_MEM_SIZE, sizeof lm, &lm, nullptr);
        printf("%-8s max group %4zu, private %5llu B, local %5llu B\n", argv[i], wg, (unsigned long long)pm, (unsigned long long)lm);
    }
}
