# Anima GPU engine

A small, hand-written engine that runs image and video diffusion models on a phone's graphics chip through
standard OpenCL. It is the fallback route of the [Anima Android app](https://github.com/adiirfan/anima-android) for
phones whose AI chip the app's main route cannot use, and the route the app takes for the largest models.

It is **not** a general GGUF runner. GGUF is only the container: weights are packed by the scripts in `tools/` into
the engine's own layout and number formats, and each model has its own wiring file. A GGUF made for llama.cpp or
ComfyUI will not load.

## Models

| Model | Wiring | Packer | Kind |
|---|---|---|---|
| Anima | `model_anima.h` | (fp16 weight files plus an optional 8-bit set, see `ops.h` `loadModel`) | text to picture |
| FLUX.2 Klein 4B | `model_klein.h` | `tools/pack_klein.py` | text to picture |
| Wan 2.2 5B | `model_wan.h` | `tools/pack_wan.py` | text/picture to video |
| Qwen-Image 2.1 | `model_qwen.h` | `tools/pack_qwen.py` (8-bit or 4-bit) | text to picture, picture editing |
| MOSS | `model_moss.h` | `tools/pack_moss.py` | text to picture |
| Fibo | `model_fibo.h` | `tools/pack_fibo.py` | picture editing |
| Tiny decoders/encoders (TAE) | `model_tae.h`, `tae.cl` | `tools/pack_tae.py` | latents to pixels and back |

The shared parts are in `ops.h` (matrix multiply with 8-bit and 4-bit weights, attention, norms, streaming of
blocks too big for memory) and `rt.h` (OpenCL set-up and the per-chip tuning). The chip code is `kernels.cl`.

Text readers run on the processor, not here: `reader/fibotaps.cpp` reads Fibo's prompt with llama.cpp and hands
the engine the hidden states it needs.

## Design rules

- Standard OpenCL 2.0 with `cl_khr_fp16` only - no vendor extensions on any required path, so it is meant to run on
  Adreno, Mali, Xclipse and desktop GPUs alike. Tested so far: Adreno (Snapdragon 8 Elite and 8 Elite Gen 5 phones)
  and an AMD Radeon desktop card.
- Every chip tunes itself on the first run (tile sizes, work-group sizes, kernel variants, each checked for correct
  answers before it is timed) and keeps the result in `<weights>.tune`.
- Models bigger than the chip's memory stream their blocks in one block ahead, on a helper thread, through two fixed
  slots of chip memory (`STREAM=1`).

## Build

Phone (Android NDK, arm64), straight into the app:

```bash
tools/build_app.sh [path to anima-android]
```

This builds the engine as `jniLibs/arm64-v8a/libugpu.so` (Android only lets apps run programs from their library
folder, named `lib*.so`), copies `kernels.cl` and `tae.cl` to `assets/ugpu/`, and builds Fibo's reader as
`libfibotaps.so` against the app's own llama.cpp libraries. Paths to the NDK and OpenCL headers are at the top of the
script.

Desktop (for checking against the reference scripts): `tools/build_x86.sh`.

## Run

```
ugpu <weights .gguf, or a prefix of loose weight files> <folder> [loops] [WxH]
```

The model is read from the weights file's index. `<folder>` holds the inputs (prompt states, noise, sigmas) and
receives the outputs. Main settings, as environment variables:

| Setting | Meaning |
|---|---|
| `DRAW=<steps>` | run the whole sampling loop instead of one checked step |
| `STREAM=1` | stream blocks from storage (models bigger than memory) |
| `TUNEFILE=<path>` | where the tuning result is kept (default `<weights>.tune`) |
| `SEED`, `SIGLIST`, `EULER` | sampling (Wan) |
| `KERNELS`, `TAEKERNELS` | where `kernels.cl` / `tae.cl` are |
| `PROF=1` | time every kernel |

## Checking against the reference

`tools/ref_*.py` compute one step of each model in PyTorch from the original weights and write the inputs and the
expected outputs to a folder; running the engine on that folder prints how far its answer is from the reference.

## Weights

Packed weights are on Hugging Face, one repository per model, with the files the app downloads.
