// Unit check: rotq_sg.comp against rotq.comp (SRC 0, fp16 input) on random numbers; prints how many outputs differ.
#include "vkrt.h"
#include "gen/rotq.spv.h"
#include "gen/rotq_sg.spv.h"
#include "gen/rowstats.spv.h"
#include "gen/rowstats_sg.spv.h"
#include <random>
typedef _Float16 h16;
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    vk::init();
    VkPhysicalDeviceSubgroupProperties ps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; p2.pNext = &ps;
    vkGetPhysicalDeviceProperties2(vk::C.pd, &p2);
    printf("%s: subgroup %u, stages 0x%x, operations 0x%x (shuffle %s, arithmetic %s)\n", vk::C.props.deviceName, ps.subgroupSize,
           ps.supportedStages, ps.supportedOperations, ps.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT ? "yes" : "no",
           ps.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT ? "yes" : "no");
    const uint32_t N = 64, K = 2048, GS = 256;
    vk::Buf in = vk::buf(N * K * 2), d = vk::buf(256), q1 = vk::buf(N * K), q2 = vk::buf(N * K), s1 = vk::buf(N * K / GS * 4), s2 = vk::buf(N * K / GS * 4);
    std::mt19937 g(1); std::normal_distribution<float> nd;
    for (uint32_t i = 0; i < N * K; i++) ((h16 *)in.map)[i] = (h16)nd(g);
    struct { uint32_t N, K, ld, col0, Npad, GS; float eps; } pc{N, K, K, 0, N, GS, 1e-6f};
    vk::Kern a, b;
    vk::kern(a, "rotq", rotq_spv, sizeof rotq_spv, 8, 28, {0, 0});
    vk::kern(b, "rotq_sg", rotq_sg_spv, sizeof rotq_sg_spv, 8, 28, {0, 0, GS / 4});
    vk::Graph G;
    G.add(a, {vk::R(d), vk::R(in), vk::R(d), vk::R(d), vk::R(d), vk::R(q1), vk::R(s1), vk::R(d)}, pc, N, K / 256);
    G.add(b, {vk::R(d), vk::R(in), vk::R(d), vk::R(d), vk::R(d), vk::R(q2), vk::R(s2), vk::R(d)}, pc, N, K / 256);
    G.run();
    int bad = 0, first = -1;
    for (uint32_t i = 0; i < N * K; i++) if (((int8_t *)q1.map)[i] != ((int8_t *)q2.map)[i]) { if (first < 0) first = i; bad++; }
    int sbad = 0;
    for (uint32_t i = 0; i < N * K / GS; i++) if (fabsf(((float *)s1.map)[i] - ((float *)s2.map)[i]) > 1e-6f * fabsf(((float *)s1.map)[i])) sbad++;
    printf("int8 outputs that differ: %d of %u (first at %d: %d vs %d); scales that differ: %d\n", bad, N * K, first,
           first >= 0 ? ((int8_t *)q1.map)[first] : 0, first >= 0 ? ((int8_t *)q2.map)[first] : 0, sbad);
    // SRC 1: layer norm from an fp32 stream, statistics from each rowstats version
    vk::Buf res = vk::buf(N * K * 4), sh = vk::buf(K * 4), sc = vk::buf(K * 4), st1 = vk::buf(N * 8), st2 = vk::buf(N * 8);
    for (uint32_t i = 0; i < N * K; i++) ((float *)res.map)[i] = nd(g) * 3 + 0.5f;
    for (uint32_t i = 0; i < K; i++) { ((float *)sh.map)[i] = nd(g) * 0.1f; ((float *)sc.map)[i] = nd(g) * 0.1f; }
    vk::Kern r1, r2, a1, b1;
    vk::kern(r1, "rowstats", rowstats_spv, sizeof rowstats_spv, 2, 12, {});
    vk::kern(r2, "rowstats_sg", rowstats_sg_spv, sizeof rowstats_sg_spv, 2, 12, {});
    vk::kern(a1, "rotq1", rotq_spv, sizeof rotq_spv, 8, 28, {1, 0});
    vk::kern(b1, "rotq_sg1", rotq_sg_spv, sizeof rotq_sg_spv, 8, 28, {1, 0, GS / 4});
    struct { uint32_t N, K; float eps; } pst{N, K, 1e-6f};
    vk::Graph H;
    H.add(r1, {vk::R(res), vk::R(st1)}, pst, N);
    H.add(r2, {vk::R(res), vk::R(st2)}, pst, (N + 3) / 4);
    H.add(a1, {vk::R(res), vk::R(d), vk::R(sh), vk::R(sc), vk::R(d), vk::R(q1), vk::R(s1), vk::R(st1)}, pc, N, K / 256);
    H.add(b1, {vk::R(res), vk::R(d), vk::R(sh), vk::R(sc), vk::R(d), vk::R(q2), vk::R(s2), vk::R(st1)}, pc, N, K / 256);
    H.run();
    double md = 0, rd = 0;
    for (uint32_t i = 0; i < N; i++) {
        md = std::max(md, (double)fabsf(((float *)st1.map)[2 * i] - ((float *)st2.map)[2 * i]));
        rd = std::max(rd, (double)fabsf(((float *)st1.map)[2 * i + 1] / ((float *)st2.map)[2 * i + 1] - 1));
    }
    printf("statistics: mean differs by up to %.2e, 1/std by up to %.2e (relative); e.g. token 5: %g %g vs %g %g\n", md, rd,
           ((float *)st1.map)[10], ((float *)st1.map)[11], ((float *)st2.map)[10], ((float *)st2.map)[11]);
    bad = 0;
    for (uint32_t i = 0; i < N * K; i++) if (((int8_t *)q1.map)[i] != ((int8_t *)q2.map)[i]) bad++;
    printf("layer-norm path, same statistics: %d of %u int8 outputs differ\n", bad, N * K);
    return 0;
}
