// Vulkan multiply bench: every setting of one multiply shader on one shape, checked against the processor and timed.
//   vkbench [N M K] [reps] [f16|cm16|cm8|cm8s]      (VKDEV picks the device)
// f16: the plain tiled shader; cm16: the matrix hardware (fp16 products, fp32 totals every KS slices);
// cm8: the matrix hardware on 8-bit whole numbers (exact 32-bit sums); cm8s: cm8 plus the engine's scale per block of 64.
// Prints the chip's features, then per setting: ms per multiply, TFLOPS, worst relative error of 64 sampled outputs.
#include "vkrt.h"
#include "gemm_f16.spv.h"
#include "gemm_cm16.spv.h"
#include "gemm_cm8.spv.h"
#include "gemm_cm8b.spv.h"
#include <cmath>
#include <random>

int main(int argc, char **argv) {
    const uint32_t N = argc > 3 ? atoi(argv[1]) : 2304, M = argc > 3 ? atoi(argv[2]) : 6144, K = argc > 3 ? atoi(argv[3]) : 6144;
    const int reps = argc > 4 ? atoi(argv[4]) : 5;
    const std::string mode = argc > 5 ? argv[5] : "f16";
    setvbuf(stdout, nullptr, _IONBF, 0);
    vk::init();
    vk::describe();
    if (!vk::C.f.storage16) { fprintf(stderr, "no 16-bit buffers on this chip\n"); return 1; }
    if (mode != "f16" && !vk::C.f.coopmat) { fprintf(stderr, "no matrix hardware on this chip\n"); return 1; }
    printf("shape: %u x %u x %u (%.1f GFLOP per multiply), shader %s\n", N, M, K, 2.0 * N * M * K / 1e9, mode.c_str());

    std::mt19937 g(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    const bool i8 = mode == "cm8" || mode == "cm8s", sc = mode == "cm8s";
    const uint32_t GS = getenv("GS") ? atoi(getenv("GS")) : 64, NB = K / GS;
    const size_t eb = i8 ? 1 : 2;
    vk::Buf X = vk::buf((size_t)N * K * eb), W = vk::buf((size_t)M * K * eb), Y = vk::buf((size_t)N * M * 4);
    _Float16 *x = (_Float16 *)X.map, *w = (_Float16 *)W.map;
    int8_t *xq = (int8_t *)X.map, *wq = (int8_t *)W.map;
    std::uniform_int_distribution<int> ud(-127, 127);
    const uint32_t STRIP = getenv("STRIP") ? atoi(getenv("STRIP")) : 0;
    std::vector<int8_t> xr, wr;   // row-major copies (the reference reads these)
    if (i8) {
        xr.resize((size_t)N * K); wr.resize((size_t)M * K);
        for (auto &v : xr) v = (int8_t)ud(g);
        for (auto &v : wr) v = (int8_t)ud(g);
        auto put = [&](int8_t *dst, const std::vector<int8_t> &src, uint32_t R) {
            for (uint32_t r = 0; r < R; r++) for (uint32_t k = 0; k < K; k++)
                dst[STRIP ? ((size_t)(k / 32) * R + r) * 32 + k % 32 : (size_t)r * K + k] = src[(size_t)r * K + k];
        };
        put(xq, xr, N); put(wq, wr, M);
        xq = xr.data(); wq = wr.data();
    } else {
        for (size_t i = 0; i < (size_t)N * K; i++) x[i] = (_Float16)nd(g);
        for (size_t i = 0; i < (size_t)M * K; i++) w[i] = (_Float16)(nd(g) * 0.02f);
    }
    vk::Buf SX = vk::buf((size_t)NB * N * 4), SW = vk::buf((size_t)NB * M * 4);   // block-major scales
    float *sx = (float *)SX.map, *sw = (float *)SW.map;
    std::uniform_real_distribution<float> us(0.5f, 1.5f);
    for (size_t i = 0; i < (size_t)NB * N; i++) sx[i] = us(g) * 0.01f;
    for (size_t i = 0; i < (size_t)NB * M; i++) sw[i] = us(g) * 0.001f;
    std::vector<std::pair<uint32_t, uint32_t>> at;
    std::vector<double> want;
    double rms = 0;
    for (int s = 0; s < 64; s++) {
        uint32_t n = g() % N, m = g() % M;
        double t = 0;
        for (uint32_t b = 0; b < NB; b++) {
            double d = 0;
            for (uint32_t k = b * GS; k < (b + 1) * GS; k++)
                d += i8 ? (double)xq[(size_t)n * K + k] * wq[(size_t)m * K + k] : (double)x[(size_t)n * K + k] * (double)w[(size_t)m * K + k];
            t += sc ? d * sx[(size_t)b * N + n] * sw[(size_t)b * M + m] : d;
        }
        at.push_back({n, m}); want.push_back(t); rms += t * t;
    }
    rms = std::sqrt(rms / 64);

    // a setting: its specialization constants, the work-group size, the grid, and a label
    struct Cfg { std::vector<uint32_t> spec; uint32_t gx, gy; std::string label; bool ok; };
    std::vector<Cfg> cfgs;
    const uint32_t *code; size_t bytes;
    if (mode == "f16") {
        code = gemm_f16_spv; bytes = sizeof gemm_f16_spv;
        const uint32_t t[][5] = {{32, 32, 16, 2, 2}, {64, 64, 16, 4, 4}, {64, 64, 16, 8, 4}, {64, 64, 16, 4, 8}, {128, 64, 16, 8, 4},
                                 {64, 128, 16, 4, 8}, {128, 128, 16, 8, 8}, {128, 128, 8, 8, 8}, {128, 64, 8, 8, 4}};
        for (auto &c : t) {
            const uint32_t th = (c[0] / c[3]) * (c[1] / c[4]);
            char l[96]; snprintf(l, sizeof l, "%3u x %3u, k %2u, per thread %u x %u (%3u threads)", c[0], c[1], c[2], c[3], c[4], th);
            const bool ok = th <= vk::C.props.limits.maxComputeWorkGroupInvocations && (c[0] + c[1]) * c[2] * 2 <= vk::C.f.sharedBytes && K % c[2] == 0;
            cfgs.push_back({{c[0], c[1], c[2], c[3], c[4], th}, (M + c[1] - 1) / c[1], (N + c[0] - 1) / c[0], l, ok});
        }
    } else if (mode == "cm8s") {
        code = gemm_cm8b_spv; bytes = sizeof gemm_cm8b_spv;
        const uint32_t SG = vk::C.f.subgroup, TM = 64, TK = 32;
        for (uint32_t TN : {32u, 64u}) for (uint32_t SM : {1u, 2u, 4u}) for (uint32_t SN : {1u, 2u, 4u}) {
            if (SM * SN * SG > vk::C.props.limits.maxComputeWorkGroupInvocations) continue;
            char l[96]; snprintf(l, sizeof l, "tile 64x%u, %u x %u subgroups", TN, SM, SN);
            const bool ok = N % (TM * SM) == 0 && M % (TN * SN) == 0 && K % GS == 0 && SM * SN * (TM * TN + TM) * 4 <= vk::C.f.sharedBytes;
            cfgs.push_back({{TM, TN, TK, SM, SN, GS, SG, STRIP, SM * SN * SG}, M / (TN * SN), N / (TM * SM), l, ok});
        }
    } else if (mode == "cm8") {
        code = gemm_cm8_spv; bytes = sizeof gemm_cm8_spv;
        const uint32_t SG = vk::C.f.subgroup, TM = 64, TK = 32;
        for (uint32_t TN : {64u, 32u}) for (uint32_t SM : {1u, 2u, 4u}) for (uint32_t SN : {1u, 2u, 4u}) {
            if (SM * SN * SG > vk::C.props.limits.maxComputeWorkGroupInvocations) continue;
            char l[96]; snprintf(l, sizeof l, "tile 64x%u, %u x %u subgroups", TN, SM, SN);
            const bool ok = N % (TM * SM) == 0 && M % (TN * SN) == 0 && K % TK == 0;
            cfgs.push_back({{TM, TN, TK, SM, SN, SM * SN * SG}, M / (TN * SN), N / (TM * SM), l, ok});
        }
    } else {
        code = gemm_cm16_spv; bytes = sizeof gemm_cm16_spv;
        const uint32_t TM = 64, TN = 64, TK = 16, SG = vk::C.f.subgroup;
        for (uint32_t SM : {1u, 2u, 4u}) for (uint32_t SN : {1u, 2u, 4u}) for (uint32_t KS : {1u, 4u}) {
            if (SM * SN * SG > vk::C.props.limits.maxComputeWorkGroupInvocations) continue;
            char l[96]; snprintf(l, sizeof l, "%u x %u subgroups, fp32 total every %3u inputs", SM, SN, KS * TK);
            const bool ok = N % (TM * SM) == 0 && M % (TN * SN) == 0 && K % (TK * KS) == 0;
            cfgs.push_back({{TM, TN, TK, SM, SN, KS, SM * SN * SG}, M / (TN * SN), N / (TM * SM), l, ok});
        }
    }
    double best = 1e30; std::string bl;
    for (auto &c : cfgs) {
        printf("  %s: ", c.label.c_str());
        if (!c.ok) { printf("not possible here\n"); continue; }
        vk::Pipe P;
        if (!vk::pipe(P, code, bytes, sc ? 5 : 3, 12, c.spec)) { printf("the chip refused it\n"); vk::freePipe(P); continue; }
        if (sc) vk::bind(P, {&X, &W, &SX, &SW, &Y}); else vk::bind(P, {&X, &W, &Y});
        const uint32_t pc[3] = {N, M, K};
        memset(Y.map, 0, Y.size);
        vk::run(P, pc, c.gx, c.gy, 1, 1);
        double err = 0;   // worst miss, measured against the typical size of an output
        const float *y = (const float *)Y.map;
        const int32_t *yi = (const int32_t *)Y.map;
        for (size_t s = 0; s < at.size(); s++) {
            const size_t o = (size_t)at[s].first * M + at[s].second;
            const double got = i8 && !sc ? (double)yi[o] : (double)y[o];
            err = std::max(err, std::fabs(got - want[s]) / rms);
        }
        const double ms = vk::run(P, pc, c.gx, c.gy, 1, reps);
        printf("%8.2f ms  %5.2f TFLOPS  error %.1e%s\n", ms, 2.0 * N * M * K / ms / 1e9, err, err > (i8 && !sc ? 0 : 2e-2) ? "  WRONG" : "");
        if (err <= 2e-2 && ms < best) { best = ms; bl = c.label; }
        vk::freePipe(P);
    }
    if (best < 1e30) printf("best: %s: %.2f ms, %.2f TFLOPS\n", bl.c_str(), best, 2.0 * N * M * K / best / 1e9);
    return 0;
}
