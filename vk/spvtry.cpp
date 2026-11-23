// Compile test: asks the chip driver to build each SPIR-V file given, one per process, and says whether it took it.
//   spvtry file.spv nbuf [spec values...]
#include "vkrt.h"
#include <fstream>
int main(int argc, char **argv) {
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> d((std::istreambuf_iterator<char>(f)), {});
    std::vector<uint32_t> spec;
    for (int i = 3; i < argc; i++) spec.push_back(atoi(argv[i]));
    vk::init();
    vk::Pipe P;
    printf("%s: %s\n", argv[1], vk::pipe(P, (const uint32_t *)d.data(), d.size(), atoi(argv[2]), 12, spec) ? "built" : "refused");
    return 0;
}
