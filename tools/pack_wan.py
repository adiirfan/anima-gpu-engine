"""Wan 2.2 TI2V 5B -> one GGUF file for the universal phone engine (2026-11-18).
Big layers 8-bit (ConvRot, as pack_klein.py) with their biases as "<name>.b" (fp32); small ones fp16. Changes made
here so the engine's kernels apply unchanged:
  - q and k rows of every head reordered (even channels, then odd): Wan's pairwise rope becomes rotate-half; the
    q / k biases and norm weights are reordered the same way (the whole-width rms is unchanged by the order).
  - self-attention q, k, v as one layer "qkv"; cross-attention k, v as one layer "ckv".
  - norm3's weight stored minus one, because the engine's norm multiplies by (1 + scale).
  - the patch layer's bias rides on an extra input channel that is always 1 (input 200 wide: 192 + 1 + 7 zero).
  - the head's 192 outputs reordered to the patch input order (c * 4 + dy * 2 + dx), so a step is x += dt * v.
Usage: py.sh pack_wan.py <out.gguf> [GS]"""
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
SRC = "/home/debian/ComfyUI/models/diffusion_models/Wan2_2-TI2V-5B-Turbo_fp16.safetensors"
D, HEADS, HD, F, NB = 3072, 24, 128, 14336, 30
f = safe_open(SRC, "pt")
g = lambda k: f.get_tensor(k).float()
writer = gguf.GGUFWriter(OUT, "ugpu")
index = []

HPERM = torch.cat([torch.arange(0, HD, 2), torch.arange(1, HD, 2)])
FULLPERM = torch.cat([h * HD + HPERM for h in range(HEADS)])     # every head of a D-wide q or k

def put(name, kind, a, M, K):
    writer.add_tensor(name, np.ascontiguousarray(a))
    index.append(f"{name} {kind} {M} {K}")
def q8(name, w, b=None):
    M, K = w.shape
    w = rot_weight(w)
    t = w.view(M, K // GS, GS)
    s = t.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
    q = (t / s).round().clamp(-127, 127).to(torch.int8).view(M, K)
    put(name, "q8", q.view(M, K // 16, 16).permute(1, 0, 2).contiguous().numpy(), M, K)
    put(name + ".s", "qs", s.view(M, K // GS).contiguous().numpy().astype(np.float32), M, K // GS)
    if b is not None: f32(name + ".b", b)
def pk(name, w):          # fp16 packed weights for the image multiply: texel (og, k2) = 4 outputs x 2 inputs
    M, K = w.shape
    a = w.half().view(M // 4, 4, K // 2, 2).permute(2, 0, 3, 1).reshape(K // 2, M // 4, 8)
    put(name, "pk", a.contiguous().numpy(), M, K)
def f16(name, w):
    put(name, "f16", w.half().contiguous().numpy(), *w.shape)
def f32(name, v):
    put(name, "f32", v.float().contiguous().numpy(), v.numel(), 1)

for i in range(NB):
    b, o = f"blocks.{i}.", f"b{i}."
    sa, ca = b + "self_attn.", b + "cross_attn."
    q8(o + "qkv", torch.cat([g(sa + "q.weight")[FULLPERM], g(sa + "k.weight")[FULLPERM], g(sa + "v.weight")]),
       torch.cat([g(sa + "q.bias")[FULLPERM], g(sa + "k.bias")[FULLPERM], g(sa + "v.bias")]))
    q8(o + "o", g(sa + "o.weight"), g(sa + "o.bias"))
    f32(o + "qn", g(sa + "norm_q.weight")[FULLPERM]); f32(o + "kn", g(sa + "norm_k.weight")[FULLPERM])
    q8(o + "cq", g(ca + "q.weight"), g(ca + "q.bias"))
    q8(o + "ckv", torch.cat([g(ca + "k.weight"), g(ca + "v.weight")]), torch.cat([g(ca + "k.bias"), g(ca + "v.bias")]))
    q8(o + "co", g(ca + "o.weight"), g(ca + "o.bias"))
    f32(o + "cqn", g(ca + "norm_q.weight")); f32(o + "ckn", g(ca + "norm_k.weight"))
    f32(o + "n3w", g(b + "norm3.weight") - 1); f32(o + "n3b", g(b + "norm3.bias"))
    q8(o + "ff1", g(b + "ffn.0.weight"), g(b + "ffn.0.bias"))
    q8(o + "ff2", g(b + "ffn.2.weight"), g(b + "ffn.2.bias"))
    f32(o + "mod", g(b + "modulation").view(6 * D))
    print("block", i, flush=True)

pw = g("patch_embedding.weight").view(D, 192)
pk("patch", torch.cat([pw, g("patch_embedding.bias").view(D, 1), torch.zeros(D, 7)], 1))
hp = torch.tensor([(k % 4) * 48 + k // 4 for k in range(192)])     # engine row c*4+j <- ComfyUI row j*48+c
q8("head", g("head.head.weight")[hp], g("head.head.bias")[hp])
f32("head.mod", g("head.modulation").view(2 * D))
q8("txt1", g("text_embedding.0.weight"), g("text_embedding.0.bias"))
q8("txt2", g("text_embedding.2.weight"), g("text_embedding.2.bias"))
f16("t1", g("time_embedding.0.weight")); f32("t1.b", g("time_embedding.0.bias"))
f16("t2", g("time_embedding.2.weight")); f32("t2.b", g("time_embedding.2.bias"))
f16("tp", g("time_projection.1.weight")); f32("tp.b", g("time_projection.1.bias"))
writer.add_string("ugpu.model", "wan")
writer.add_string("ugpu.index", "\n".join(index))
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print("wrote", OUT, os.path.getsize(OUT) / 1e9, "GB,", len(index), "tensors")
