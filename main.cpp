// Universal graphics-chip engine for diffusion transformers on phones (2026-11-18).
// Usage: ugpu <weights prefix or .gguf file> <ref dir> [loops] [WxH]
// The model comes from the "#model <name>" line of the weights index (default anima). Settings (environment):
//   Q8=<8-bit weights prefix>  Q8KEEP=<layers kept fp16>  CEN=<centring table>  PVSM=1  TODO=1 [TODOB=a,b]
//   PROF=1  DRAW=<steps> [DRAWCOOL=<mC>]  LCFG/CFG (skip tuning)  TUNEFILE (default <weights>.tune)
#include "model_anima.h"
#include "model_klein.h"
#include "model_wan.h"
#include "model_qwen.h"
#include "model_tae.h"
#include "model_moss.h"
#include "model_fibo.h"
#include "model_vaeenc.h"

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);   // each line reaches the app as soon as it is printed (a pipe is block-buffered otherwise)
    if (argc < 3) { fprintf(stderr, "usage: %s <weights prefix> <ref dir> [loops] [WxH]\n", argv[0]); return 1; }
    const std::string weights = argv[1], R = argv[2];
    const int loops = argc > 3 ? atoi(argv[3]) : 1;
    int PW = 512, PH = 512;
    if (argc > 4) sscanf(argv[4], "%dx%d", &PW, &PH);
    if (getenv("CFG")) sscanf(getenv("CFG"), "%d,%d,%d,%d", &TN, &TM, &L0, &L1);
    if (getenv("LCFG")) sscanf(getenv("LCFG"), "%d,%d,%d,%d,%d", &LOCALGEMM, &TNI, &GG, &WGS, &KCS);
    if (getenv("CACHE_KB")) CACHE = (size_t)atoi(getenv("CACHE_KB")) << 10;
    prof = getenv("PROF") != nullptr;
    STREAM = getenv("STREAM") != nullptr;   // blocks' big tables loaded one block at a time (models bigger than memory)
    if (!getenv("TUNEFILE")) setenv("TUNEFILE", (weights + ".tune").c_str(), 1);

    clInit();
    auto t0 = std::chrono::steady_clock::now();
    KSRC = slurp(getenv("KERNELS") ? getenv("KERNELS") : "kernels.cl");
    std::string model = ops::loadModel(weights);
    if (model.empty()) model = "anima";
    printf("model %s, load %.1f s (%zu weights)\n", model.c_str(),
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), W.size());
    if (getenv("LOADTEST") && !SW.empty()) {   // how fast streamed blocks come in: storage read vs. hand-over to the chip
        using clk = std::chrono::steady_clock;
        auto ms = [](clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); };
        for (int b = 0; b < 6; b++) {
            const std::string pre = "b" + std::to_string(b) + ".";
            size_t bytes = 0; volatile unsigned sum = 0;
            auto a = clk::now();
            for (auto it = SW.lower_bound(pre); it != SW.end() && it->first.compare(0, pre.size(), pre) == 0; ++it) {
                for (size_t o = 0; o < it->second.bytes; o += 4096) sum += (unsigned char)it->second.src[o];
                bytes += it->second.bytes;
            }
            const double tr = ms(a);
            a = clk::now();
            auto m = loadBlock(b); clFinish(q);
            const double tc = ms(a);
            a = clk::now();
            auto m2 = loadBlock(b); clFinish(q);
            const double tc2 = ms(a);
            for (auto &kv : m) clReleaseMemObject(kv.second.m);
            for (auto &kv : m2) clReleaseMemObject(kv.second.m);
            printf("block %d: %.0f MB, read %.0f ms (%.0f MB/s), to chip %.0f ms, again %.0f ms\n", b, bytes / 1e6, tr,
                   bytes / 1e3 / tr, tc, tc2);
        }
        return 0;
    }
    if (model == "anima") return anima::run(weights, R, loops, PW, PH);
    if (model == "klein") return klein::run(R, loops);
    if (model == "wan") return wan::run(R, loops);
    if (model == "qwen") return qwen::run(R, loops);
    if (model == "tae") return tae::run(R, loops);
    if (model == "moss") return moss::run(R, loops);
    if (model == "fibo") return fibo::run(R, loops);
    if (model == "vaeenc") return vaeenc::run(R, loops);
    fprintf(stderr, "no wiring for model %s\n", model.c_str());
    return 1;
}
