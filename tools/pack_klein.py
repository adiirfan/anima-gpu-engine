"""FLUX.2 Klein 4B -> one GGUF file for the universal phone engine (2026-11-18).
The big layers are 8-bit (ConvRot: 256-block Hadamard on the input side, int8 with one scale per output and GS
inputs), the small ones fp16. Layouts are the engine's own, stored as plain GGUF tensors (I8 / F16 / F32); the
metadata key "ugpu.index" lists "name kind M K" per tensor and "ugpu.model" names the wiring ("klein").
Changes made here so the engine's kernels apply unchanged:
  - q and k rows of every head reordered (even channels, then odd), turning Flux's pairwise rope into the
    rotate-half form; the q/k norm scales are reordered the same way (scores are unchanged).
  - SwiGLU inputs stored [gate 8][up 8] per 16 outputs, so the 8-bit multiply finishes silu(gate) * up itself.
  - single blocks' linear1 split into qkv and mlp_in, linear2 into out_a (attention part) and out_m (MLP part).
Usage: py.sh pack_klein.py <out.gguf> [GS]"""
import os
import sys
sys.path[:0] = ["/home/debian/anima-qnn", "/home/debian/llama.cpp/gguf-py"]
import numpy as np
import torch
from safetensors import safe_open
from anima_qnn import rot_weight
import gguf

OUT = sys.argv[1]
GS = int(sys.argv[2]) if len(sys.argv) > 2 else 64
SRC = "/home/debian/models/klein/flux-2-klein-4b.safetensors"
D, HEADS, HD, F = 3072, 24, 128, 9216
f = safe_open(SRC, "pt")
g = lambda k: f.get_tensor(k).float()
writer = gguf.GGUFWriter(OUT, "ugpu")
index = []

HPERM = torch.cat([torch.arange(0, HD, 2), torch.arange(1, HD, 2)])
def qk_perm(w):           # rows of [q; k; v] (3D rows): reorder each head's channels of q and k
    w = w.clone()
    for part in (0, 1):
        for h in range(HEADS):
            r0 = part * D + h * HD
            w[r0:r0 + HD] = w[r0:r0 + HD][HPERM]
    return w
def swiglu_rows(w):       # [gate F; up F] rows -> [gate 8][up 8] per 16 rows
    gt, up = w[:F], w[F:]
    return torch.stack([gt.view(F // 8, 8, -1), up.view(F // 8, 8, -1)], 1).reshape(2 * F, -1)

def put(name, kind, a, M, K):
    writer.add_tensor(name, np.ascontiguousarray(a))
    index.append(f"{name} {kind} {M} {K}")
def q8(name, w):
    M, K = w.shape
    w = rot_weight(w)
    t = w.view(M, K // GS, GS)
    s = t.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
    q = (t / s).round().clamp(-127, 127).to(torch.int8).view(M, K)
    put(name, "q8", q.view(M, K // 16, 16).permute(1, 0, 2).contiguous().numpy(), M, K)
    put(name + ".s", "qs", s.view(M, K // GS).contiguous().numpy().astype(np.float32), M, K // GS)
def pk(name, w):          # fp16 packed weights for the image multiply: texel (og, k2) = 4 outputs x 2 inputs
    M, K = w.shape
    a = w.half().view(M // 4, 4, K // 2, 2).permute(2, 0, 3, 1).reshape(K // 2, M // 4, 8)
    put(name, "pk", a.contiguous().numpy(), M, K)
def f16(name, w):         # row-major fp16 for the small vector multiplies
    put(name, "f16", w.half().contiguous().numpy(), *w.shape)
def f32(name, v):
    put(name, "f32", v.float().contiguous().numpy(), v.numel(), 1)

for i in range(5):
    b, o = f"double_blocks.{i}.", f"d{i}."
    for s in ("img", "txt"):
        q8(o + s + "_qkv", qk_perm(g(b + s + "_attn.qkv.weight")))
        q8(o + s + "_proj", g(b + s + "_attn.proj.weight"))
        q8(o + s + "_mlp_in", swiglu_rows(g(b + s + "_mlp.0.weight")))
        q8(o + s + "_mlp_out", g(b + s + "_mlp.2.weight"))
        f32(o + s + "_qn", g(b + s + "_attn.norm.query_norm.scale")[HPERM])
        f32(o + s + "_kn", g(b + s + "_attn.norm.key_norm.scale")[HPERM])
    print("double", i, flush=True)
for j in range(20):
    b, o = f"single_blocks.{j}.", f"s{j}."
    l1, l2 = g(b + "linear1.weight"), g(b + "linear2.weight")
    q8(o + "qkv", qk_perm(l1[:3 * D]))
    q8(o + "mlp_in", swiglu_rows(l1[3 * D:]))
    q8(o + "out_a", l2[:, :D].contiguous())
    q8(o + "out_m", l2[:, D:].contiguous())
    f32(o + "qn", g(b + "norm.query_norm.scale")[HPERM])
    f32(o + "kn", g(b + "norm.key_norm.scale")[HPERM])
    print("single", j, flush=True)
pk("img_in", g("img_in.weight"))
pk("txt_in", g("txt_in.weight"))
pk("final_linear", g("final_layer.linear.weight"))
f16("time_in1", g("time_in.in_layer.weight"))
f16("time_in2", g("time_in.out_layer.weight"))
f16("mod_img", g("double_stream_modulation_img.lin.weight"))
f16("mod_txt", g("double_stream_modulation_txt.lin.weight"))
f16("mod_single", g("single_stream_modulation.lin.weight"))
f16("final_mod", g("final_layer.adaLN_modulation.1.weight"))
writer.add_string("ugpu.model", "klein")
writer.add_string("ugpu.index", "\n".join(index))
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print("wrote", OUT, os.path.getsize(OUT) / 1e9, "GB,", len(index), "tensors")
