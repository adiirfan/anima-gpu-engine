#!/bin/bash
# Build the engine for phones and put it in the Anima app: the program as jniLibs/arm64-v8a/libugpu.so (Android only
# lets apps run programs from their library folder, and only files named lib*.so get unpacked there), its chip code
# (kernels.cl, tae.cl) as assets/ugpu/, and Fibo's prompt reader as libfibotaps.so. Usage: tools/build_app.sh [app folder, default ~/anima-android]
set -euo pipefail
cd "$(dirname "$0")/.."
APP=${1:-/home/debian/anima-android}
CXX=/home/debian/Android/Sdk/ndk/28.2.13676358/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++
mkdir -p "$APP/app/src/main/jniLibs/arm64-v8a" "$APP/app/src/main/assets/ugpu"
timeout 300 "$CXX" -O2 -std=c++17 -static-libstdc++ -I/home/debian/ugpu/third_party/OpenCLHeaders \
    main.cpp -o "$APP/app/src/main/jniLibs/arm64-v8a/libugpu.so" -L/home/debian/ugpu/third_party/lib -lOpenCL
cp kernels.cl tae.cl "$APP/app/src/main/assets/ugpu/"
# Fibo's prompt reader (reader/fibotaps.cpp), linked against the app's own llama.cpp libraries (built from
# ~/llamacpp-hexagon: the headers must be that same version)
L=/home/debian/llamacpp-hexagon
J="$APP/app/src/main/jniLibs/arm64-v8a"
timeout 300 "$CXX" -O2 -std=c++17 -static-libstdc++ reader/fibotaps.cpp -o "$J/libfibotaps.so" \
    -I$L/include -I$L/ggml/include -I$L/tools/mtmd -L"$J" -lmtmd -lllama -lggml -lggml-base
ls -la "$J/libugpu.so" "$J/libfibotaps.so" "$APP/app/src/main/assets/ugpu/"
# Klein's and Anima's prompt readers for the graphics-chip route (reader/texttaps.cpp), same libraries
timeout 300 "$CXX" -O3 -ffast-math -std=c++17 -static-libstdc++ reader/texttaps.cpp -o "$J/libtexttaps.so" \
    -I$L/include -I$L/ggml/include -L"$J" -lllama -lggml -lggml-base
ls -la "$J/libtexttaps.so"
