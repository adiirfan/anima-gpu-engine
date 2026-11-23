#!/bin/bash
# Builds the Vulkan shaders (GLSL -> SPIR-V, embedded as C arrays), the bench and the engine (vkugpu: every model it knows), for this computer
# and for phones:  vk/build.sh  -> vk/vkbench_{x86,android}, vk/vkugpu_{x86,android}
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p gen
for s in *.comp; do
  n=${s%.comp}
  # matrix-hardware shaders are never run through the SPIR-V optimiser: its output crashes the Adreno 840 driver
  # (gemm_hc) or fails to build at all; the chip driver optimises anyway
  if grep -q cooperative_matrix "$s"; then glslc --target-env=vulkan1.2 "$s" -o gen/$n.spv
  else glslc -O --target-env=vulkan1.2 "$s" -o gen/$n.spv; fi
  python3 - gen/$n.spv gen/$n.spv.h ${n} <<'PY'
import sys, struct
d = open(sys.argv[1], 'rb').read()
w = struct.unpack('<%dI' % (len(d) // 4), d)
name = sys.argv[3] + '_spv'
lines = ', '.join('0x%08x' % v for v in w)
open(sys.argv[2], 'w').write('// generated from %s.comp by vk/build.sh\nstatic const uint32_t %s[] = {%s};\n' % (sys.argv[3], name, lines))
PY
done
timeout 300 g++ -O2 -std=c++17 -Igen vkbench.cpp -o vkbench_x86 -lvulkan
timeout 300 g++ -O2 -std=c++17 -I. vkmain.cpp -o vkugpu_x86 -lvulkan
timeout 300 g++ -O2 -std=c++17 -I. vktest.cpp -o vktest_x86 -lvulkan
NDK=/home/debian/Android/Sdk/ndk/28.2.13676358/toolchains/llvm/prebuilt/linux-x86_64/bin
timeout 300 $NDK/aarch64-linux-android28-clang++ -O2 -std=c++17 -static-libstdc++ -Igen vkbench.cpp -o vkbench_android -lvulkan
timeout 300 $NDK/aarch64-linux-android28-clang++ -O2 -std=c++17 -static-libstdc++ -I. vkmain.cpp -o vkugpu_android -lvulkan
timeout 300 $NDK/aarch64-linux-android28-clang++ -O2 -std=c++17 -static-libstdc++ -I. vktest.cpp -o vktest_android -lvulkan
ls -la vkbench_x86 vkbench_android vkugpu_x86 vkugpu_android
