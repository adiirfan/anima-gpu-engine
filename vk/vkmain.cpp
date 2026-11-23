// The Vulkan engine: one program for every model it has been taught, picked by the model file's "ugpu.model".
//   vkugpu <model.gguf> <ref dir> [loops] [WxH]     env: PROF=1, VKQ=dot|cm, VKATT=plain, VKNOSG, DRAW=1, VKAUTO=1, KCACHE
// VKAUTO=1 (the app's automatic choice): exit 3 at once when OpenCL would be as fast (scale blocks under 256, or no
// matrix hardware), or when the model has no Vulkan wiring yet.
#include "vk_anima.h"
#include "vk_klein.h"
#include "vk_moss.h"
#include "vk_wan.h"
#include "vk_qwen.h"
#include "vk_fibo.h"

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model.gguf> <ref dir> [loops] [WxH]\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string Rd = argv[2];
    const int loops = argc > 3 ? atoi(argv[3]) : 1;
    int PW = 512, PH = 512;
    if (argc > 4) sscanf(argv[4], "%dx%d", &PW, &PH);
    loadGGUF(argv[1]);
    if (MODEL.empty()) MODEL = "anima";   // the first Anima files carried no model name
    if (MODEL != "anima" && MODEL != "klein" && MODEL != "moss" && MODEL != "wan" && MODEL != "qwen" && MODEL != "fibo") {
        printf("Vulkan skipped: no Vulkan wiring for %s yet\n", MODEL.c_str());
        return 3;
    }
    if (!vkStart()) return getenv("VKAUTO") ? 3 : 1;
    if (MODEL == "anima") return anima::run(Rd, loops, PW, PH);
    if (MODEL == "klein") return klein::run(Rd, loops);
    if (MODEL == "moss") return moss::run(Rd, loops);
    if (MODEL == "qwen") return qwen::run(Rd, loops);
    if (MODEL == "fibo") return fibo::run(Rd, loops);
    return wan::run(Rd, loops);
}
