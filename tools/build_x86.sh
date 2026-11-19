#!/bin/bash
# Build the engine for testing on this computer's Radeon (ROCm OpenCL from ComfyUI's venv) as ./ugpu_x86
set -euo pipefail
cd "$(dirname "$0")/.."
R=/home/debian/ComfyUI/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib
timeout 300 g++ -O2 -std=c++17 -I/home/debian/gpu-engine-test/mnn/3rd_party/OpenCLHeaders -DCL_TARGET_OPENCL_VERSION=300 \
    main.cpp -o ugpu_x86 $R/opencl/libamdocl64.so.2 -Wl,-rpath,$R:$R/opencl -lpthread
