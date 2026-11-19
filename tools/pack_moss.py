"""MOSS-Image (SFT + the 8-step DMD add-on) -> one GGUF for the universal phone engine (2026-11-18).
Big layers 4-bit (ConvRot, as pack_qwen.py) on the graphics card; small ones fp16 / fp32. The add-on (strength 0.7,
alpha / rank) is MERGED into the weights in fp32 before rounding: rounding W + dW keeps dW's effect, it only changes
which rounding noise lands where.
Layout for the engine:
  - q, k, v as one layer "qkv" (heads already in split-half order, no reordering); the MLP's up / gate stored
    [gate 8][up 8] per 16 rows for the engine's paired activation (SITU in model_moss.h).
  - norms (n1, n2, final) kept as fp32 vectors, folded into the modulation on the host each step.
  - the 2 prompt blocks (tb0, tb1) and the prompt projection with their biases ("<name>.b").
Usage: python pack_moss.py <out.gguf> [strength 0.7] [lora file]   (ComfyUI's ROCm venv; BITS=8 for 8-bit)"""
import os
import sys
sys.path[:0] = ["/home/debian/gpu-engine-test/tsbuild/anima", "/home/debian/llama.cpp/gguf-py"]
import numpy as np
import torch
from safetensors import safe_open
import anima_qnn
import gguf

OUT = sys.argv[1]
ST = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
LP = sys.argv[3] if len(sys.argv) > 3 else "/home/debian/moss-build/dmd/lora_latest.safetensors"
GS, BITS = 32, int(os.environ.get("BITS", "4"))
SRC = "/home/debian/moss-build/models/moss_image_sft_bf16.safetensors"
D, H, HD, FF, NB = 3072, 24, 128, 8192, 42
dev = "cuda" if torch.cuda.is_available() else "cpu"
HT = anima_qnn._HT.to(dev)
f = safe_open(SRC, "pt")
lf = safe_open(LP, "pt") if ST > 0 else None
g = lambda k: f.get_tensor(k).to(dev).float()
writer = gguf.GGUFWriter(OUT, "ugpu")
index = []

def put(name, kind, a, M, K):
    writer.add_tensor(name, np.ascontiguousarray(a))
    index.append(f"{name} {kind} {M} {K}")
def f16(name, w): put(name, "f16", w.half().contiguous().cpu().numpy(), *w.shape)
def f32(name, v): put(name, "f32", v.float().contiguous().cpu().numpy(), v.numel(), 1)
def rotw(w):
    M, K = w.shape
    return (w.reshape(M, K // HT.shape[0], HT.shape[0]) @ HT).reshape(M, K)
def q8(name, w, b=None):
    M, K = w.shape
    t = rotw(w).view(M, K // GS, GS)
    s = t.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
    q = (t / s).round().clamp(-127, 127).to(torch.int8).view(M, K)
    put(name, "q8", q.view(M, K // 16, 16).permute(1, 0, 2).contiguous().cpu().numpy(), M, K)
    put(name + ".s", "qs", s.view(M, K // GS).contiguous().cpu().numpy().astype(np.float32), M, K // GS)
    if b is not None: f32(name + ".b", b)
def q4(name, w, b=None):
    M, K = w.shape
    t = rotw(w).view(M, K // GS, GS)
    s = t.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7
    q = (t / s).round().clamp(-8, 7).to(torch.int16).view(M, K // 32, 2, 16)
    by = ((q[:, :, 0] & 15) | ((q[:, :, 1] & 15) << 4)).to(torch.uint8)
    put(name, "q4", by.permute(1, 0, 2).contiguous().cpu().numpy().view(np.int8), M, K)
    put(name + ".s", "qs", (s / 16).view(M, K // GS).contiguous().cpu().numpy().astype(np.float32), M, K // GS)
    if b is not None: f32(name + ".b", b)
qb4 = q4 if BITS == 4 else q8
MIX = set(filter(None, os.environ.get("MIX", "").split(",")))   # drawing-block layer kinds kept 8-bit, e.g. "down,o"
def qb(name, w, b=None):
    kind = name.split(".")[-1]
    (q8 if name.startswith("b") and kind in MIX else qb4)(name, w, b)
def pair_rows(gate, up):
    """[gate 8][up 8] per 16 rows, as the engine's paired-activation output expects"""
    n = gate.shape[0]
    return torch.stack([gate.view(n // 8, 8, -1), up.view(n // 8, 8, -1)], 1).reshape(2 * n, -1)
def pair_vec(gate, up):
    n = gate.shape[0]
    return torch.stack([gate.view(n // 8, 8), up.view(n // 8, 8)], 1).reshape(2 * n)
def lora(i, key):
    if lf is None: return 0.0
    p = f"diffusion_model.layers.{i}.proj.{key}."
    dn, up = lf.get_tensor(p + "lora_down.weight").to(dev).float(), lf.get_tensor(p + "lora_up.weight").to(dev).float()
    a = float(lf.get_tensor(p + "alpha")) if p + "alpha" in lf.keys() else dn.shape[0]
    return (up @ dn) * (ST * a / dn.shape[0])

with torch.no_grad():
    for i in range(NB):
        p, o = f"layers.{i}.", f"b{i}."
        qkv = g(p + "proj.qkv.weight")
        qkv = qkv + torch.cat([lora(i, "q"), lora(i, "k"), lora(i, "v")], 0) if lf is not None else qkv
        qb(o + "qkv", qkv)
        qb(o + "o", g(p + "proj.o.weight") + lora(i, "o"))
        ug = g(p + "proj.ffn_up_gate.weight") + lora(i, "ffn_up_gate")
        up, gate = ug.chunk(2, 0)
        qb(o + "mlp_in", pair_rows(gate, up))
        qb(o + "down", g(p + "proj.ffn_down.weight") + lora(i, "ffn_down"))
        f32(o + "qn", g(p + "q_norm.weight")); f32(o + "kn", g(p + "k_norm.weight"))
        f32(o + "n1", g(p + "norm1.weight")); f32(o + "n2", g(p + "norm2.weight"))
        print("block", i, flush=True)
    for j in range(2):
        p, o = f"text_embedder.blocks.{j}.", f"tb{j}."
        qb(o + "qkv", g(p + "qkv.weight"), g(p + "qkv.bias"))
        qb(o + "o", g(p + "proj.weight"), g(p + "proj.bias"))
        qb(o + "mlp_in", pair_rows(g(p + "ffn_gate.weight"), g(p + "ffn_up.weight")),
           pair_vec(g(p + "ffn_gate.bias"), g(p + "ffn_up.bias")))
        qb(o + "down", g(p + "ffn_down.weight"), g(p + "ffn_down.bias"))
        f32(o + "qn", g(p + "q_norm.weight")); f32(o + "kn", g(p + "k_norm.weight"))
        f32(o + "n1", g(p + "norm1.weight")); f32(o + "n2", g(p + "norm2.weight"))
    qb("tproj", g("text_embedder.proj.weight"), g("text_embedder.proj.bias"))
    vw = g("video_embedder.weight")                                              # 128 inputs, padded to the rotation's
    q8("vin", torch.cat([vw, torch.zeros_like(vw)], 1), g("video_embedder.bias"))  # 256 (the engine feeds zeros there)
    q8("fout", g("final_linear_video.weight"))
    f32("fnorm", g("final_norm_video.weight"))
    f16("t1", g("time_embedding.0.weight")); f32("t1.b", g("time_embedding.0.bias"))
    f16("t2", g("time_embedding.2.weight")); f32("t2.b", g("time_embedding.2.bias"))
    f16("mod", g("shared_block_modulation.weight")); f32("mod.b", g("shared_block_modulation.bias"))
    f16("hmod", g("video_head_modulation.weight")); f32("hmod.b", g("video_head_modulation.bias"))
writer.add_string("ugpu.model", "moss")
writer.add_string("ugpu.index", "\n".join(index))
writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
print("wrote", OUT, os.path.getsize(OUT) / 1e9, "GB,", len(index), "tensors")
