"""Krea 2 Turbo (krea/Krea-2-Turbo, 12B single-stream DiT) -> one GGUF file for the universal phone engine (2026-11-23).
Same storage as pack_klein.py: big layers 8-bit (256-block Hadamard on the input side, int8, one scale per output and
GS inputs), small ones fp16 / fp32, the engine's own layouts in plain GGUF tensors, "ugpu.index" = "name kind M K".
Changes made here so the engine's kernels apply unchanged:
  - per block one input multiply [q 6144; k 1536; v 1536; gate 6144] (qkvg), q and k rows of every head reordered
    (even channels, then odd: Flux-style pairwise rope -> rotate-half); q/k norm scales reordered the same way and
    stored as the full weight (1 + w, Krea keeps them zero-centred).
  - SwiGLU [gate 8][up 8] per 16 outputs, so the 8-bit multiply finishes silu(gate) * up itself.
  - img_in padded from 64 to 128 inputs, the output layer from 64 to 128 outputs (zero rows), sizes the engine's
    multiplies already handle.
  - the block norms (norm1 / norm2), final norm, per-block modulation tables stay fp32 as stored; the engine folds
    (1 + norm) into the modulation's scale once per step.
  - prompt side (text fusion + txt_in) packed too, 8-bit with fp32 biases ("<name>.b").
Usage: py.sh pack_krea.py <out.gguf> [GS]"""
import os
import sys
sys.path[:0] = ["/home/debian/anima-build81/build/anima", "/home/debian/llama.cpp/gguf-py"]
import glob
import numpy as np
import torch
from safetensors import safe_open
from anima_qnn import rot_weight
import gguf

OUT = sys.argv[1]
GS = int(sys.argv[2]) if len(sys.argv) > 2 else 64
SRC = "/home/debian/models/krea2-turbo/transformer"
D, HD, NB, F = 6144, 128, 28, 16384
TD, TF = 2560, 6912
torch.set_num_threads(16)

files = {}
for fn in sorted(glob.glob(SRC + "/*.safetensors")):
    s = safe_open(fn, "pt")
    for k in s.keys(): files[k] = s
g = lambda k: files[k].get_tensor(k).float()
writer = gguf.GGUFWriter(OUT, "ugpu")
index = []

HPERM = torch.cat([torch.arange(0, HD, 2), torch.arange(1, HD, 2)])
def head_perm(w):        # rows of a q or k projection: reorder each head's channels
    return w.view(-1, HD, w.shape[1])[:, HPERM].reshape(w.shape)
def swiglu_rows(gt, up):  # [gate F; up F] -> [gate 8][up 8] per 16 rows
    n = gt.shape[0]
    return torch.stack([gt.view(n // 8, 8, -1), up.view(n // 8, 8, -1)], 1).reshape(2 * n, -1)

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
def q4(name, w):
    """4-bit (as pack_qwen.py): per row and block of GS inputs one scale; texel (o, k32) = 16 bytes, byte j = input j
    (low half) and input 16 + j (high half). The kernel reads each half as nibble * 16, so the scale carries the 1/16."""
    M, K = w.shape
    w = rot_weight(w)
    t = w.view(M, K // GS, GS)
    s = t.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7
    q = (t / s).round().clamp(-8, 7).to(torch.int16).view(M, K // 32, 2, 16)
    b = ((q[:, :, 0] & 15) | ((q[:, :, 1] & 15) << 4)).to(torch.uint8)
    put(name, "q4", b.permute(1, 0, 2).contiguous().numpy().view(np.int8), M, K)
    put(name + ".s", "qs", (s / 16).view(M, K // GS).contiguous().numpy().astype(np.float32), M, K // GS)
def q4k(name, w):
    """4-bit with two-level scales for the Vulkan engine (Q4K=1, with GS 256): as pack_qwen.py's q4k. Per row and
    block of 256 inputs one fp32 scale d ("<name>.s"), per 32 inputs a whole-number sub-scale 1..15 ("<name>.sc"),
    the values -8..7 in the q4 layout above; weight = d * sc * q. OpenCL folds d * sc / 16 back into per-32 scales."""
    M, K = w.shape
    t = rot_weight(w).view(M, K // 256, 8, 32)
    s32 = t.abs().amax(-1).clamp_min(1e-12) / 7
    best = None
    for fct in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75):
        d = s32.amax(-1, keepdim=True) / 15 * fct
        sc = (s32 / d).round().clamp(1, 15)
        st = (d * sc).unsqueeze(-1)
        q = (t / st).round().clamp(-8, 7)
        e = ((q * st - t) ** 2).sum((-1, -2))
        if best is None: best = [e, d, sc, q]
        else:
            gm = e < best[0]
            best = [torch.where(gm, e, best[0]), torch.where(gm[..., None], d, best[1]), torch.where(gm[..., None], sc, best[2]),
                    torch.where(gm[..., None, None], q, best[3])]
    _, d, sc, q = best
    q = q.to(torch.int16).view(M, K // 32, 2, 16)
    by = ((q[:, :, 0] & 15) | ((q[:, :, 1] & 15) << 4)).to(torch.uint8)
    put(name, "q4", by.permute(1, 0, 2).contiguous().numpy().view(np.int8), M, K)
    put(name + ".s", "qs", d.view(M, K // 256).contiguous().numpy().astype(np.float32), M, K // 256)
    put(name + ".sc", "q4sc", sc.to(torch.uint8).view(M, K // 32).contiguous().numpy().view(np.int8), M, K // 32)
BITS = int(os.environ.get("BITS", "8"))   # 4: the drawing blocks' four big tables at 4 bits (~7 GB instead of 13.9)
# Q4K=1 (with GS 256): the Vulkan layout, read by both engines: 4-bit tables with two-level scales, 8-bit ones in
# blocks of 256 inputs
if os.environ.get("Q4K") and GS != 256: sys.exit("Q4K=1 needs GS 256 (the 8-bit tables' blocks)")
qb = (q4k if os.environ.get("Q4K") else q4) if BITS == 4 else q8
def pk(name, w):          # fp16 packed weights for the image multiply: texel (og, k2) = 4 outputs x 2 inputs
    M, K = w.shape
    a = w.half().view(M // 4, 4, K // 2, 2).permute(2, 0, 3, 1).reshape(K // 2, M // 4, 8)
    put(name, "pk", a.contiguous().numpy(), M, K)
def f16(name, w):         # row-major fp16 for the small vector multiplies
    put(name, "f16", w.half().contiguous().numpy(), *w.shape)
def f32(name, v):
    put(name, "f32", v.float().contiguous().numpy(), v.numel(), 1)

def attn_mlp(b, o, heads_q, heads_k, dim, ff, qq=q8):
    """One block's big layers: qkvg, wo, mlp_in (SwiGLU), down; q/k norms (full weight, reordered)."""
    q = head_perm(g(b + "attn.to_q.weight")); k = head_perm(g(b + "attn.to_k.weight"))
    qq(o + "qkvg", torch.cat([q, k, g(b + "attn.to_v.weight"), g(b + "attn.to_gate.weight")], 0))
    qq(o + "wo", g(b + "attn.to_out.0.weight"))
    qq(o + "mlp_in", swiglu_rows(g(b + "ff.gate.weight"), g(b + "ff.up.weight")))
    qq(o + "down", g(b + "ff.down.weight"))
    f32(o + "qn", (1 + g(b + "attn.norm_q.weight"))[HPERM])
    f32(o + "kn", (1 + g(b + "attn.norm_k.weight"))[HPERM])

for i in range(NB):
    b = f"transformer_blocks.{i}."
    attn_mlp(b, f"b{i}.", 48, 12, D, F, qb)
    f32(f"m{i}.lin", g(b + "scale_shift_table"))   # (6, D): prescale, preshift, pregate, postscale, postshift, postgate
    f32(f"m{i}.n1", g(b + "norm1.weight"))
    f32(f"m{i}.n2", g(b + "norm2.weight"))
    print("block", i, flush=True)

# prompt side: text fusion (2 blocks across the 12 reader layers per token, projector 12 -> 1, 2 refiner blocks)
for kind, n in (("layerwise_blocks", "tl"), ("refiner_blocks", "tr")):
    for j in range(2):
        b, o = f"text_fusion.{kind}.{j}.", f"{n}{j}."
        attn_mlp(b, o, 20, 20, TD, TF)
        f32(o + "n1", g(b + "norm1.weight"))
        f32(o + "n2", g(b + "norm2.weight"))
f32("tf.proj", g("text_fusion.projector.weight"))
f32("txt.n", g("txt_in.norm.weight"))
q8("txt.l1", g("txt_in.linear_1.weight")); f32("txt.l1.b", g("txt_in.linear_1.bias"))
q8("txt.l2", g("txt_in.linear_2.weight")); f32("txt.l2.b", g("txt_in.linear_2.bias"))
print("prompt side", flush=True)

# picture in / out, noise level
w = torch.zeros(D, 128); w[:, :64] = g("img_in.weight"); pk("first", w); f32("first.b", g("img_in.bias"))
w = torch.zeros(128, D); w[:64] = g("final_layer.linear.weight"); q8("last", w)
f32("last.b", g("final_layer.linear.bias"))
f32("last.n", g("final_layer.norm.weight"))
f32("last.lin", g("final_layer.scale_shift_table"))   # (2, D): scale, shift
f16("t1", g("time_embed.linear_1.weight")); f32("t1.b", g("time_embed.linear_1.bias"))
f16("t2", g("time_embed.linear_2.weight")); f32("t2.b", g("time_embed.linear_2.bias"))
f16("tproj", g("time_mod_proj.weight")); f32("tproj.b", g("time_mod_proj.bias"))

writer.add_string("ugpu.model", "krea")
writer.add_string("ugpu.index", "\n".join(index))
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print("wrote", OUT, os.path.getsize(OUT) / 1e9, "GB,", len(index), "tensors")
