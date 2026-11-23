"""Qwen-Image 2.1 (Turbo) -> one GGUF file for the universal phone engine (2026-11-18).
Big layers 8-bit (ConvRot, as pack_klein.py), done on the graphics card; small ones fp16. No biases in this model.
Changes made here so the engine's kernels apply unchanged:
  - q and k rows of every head reordered (even channels, then odd): pairwise rope -> rotate-half; norm scales too.
  - q, k, v as one layer "qkv"; the MLP's gate_up stored [gate 8][up 8] per 16 rows for the SwiGLU multiply.
  - txt_in's norm weight stored with its +1 (ComfyUI's RMSNorm offset), applied on the host with the prompt.
Usage: python pack_qwen.py <out.gguf> [GS]   (ComfyUI's ROCm venv; the Radeon does the rounding)"""
import os
import sys
sys.path[:0] = ["/home/debian/anima-qnn", "/home/debian/llama.cpp/gguf-py"]
import numpy as np
import torch
from safetensors import safe_open
import anima_qnn
import gguf

OUT = sys.argv[1]
GS = int(sys.argv[2]) if len(sys.argv) > 2 else 32
BITS = int(os.environ.get("BITS", "4"))   # 4: two weights per byte, ~4.4 GB instead of 7.7 (the phone has 11 GB)
SRC = "/home/debian/phone-v81/qwen/models/qwen_image_2.1_turbo_bf16.safetensors"
D, H, HD, FF, NB = 4096, 32, 128, 12288, 32
dev = "cuda" if torch.cuda.is_available() else "cpu"
HT = anima_qnn._HT.to(dev)
f = safe_open(SRC, "pt")
g = lambda k: f.get_tensor(k).to(dev).float()
writer = gguf.GGUFWriter(OUT, "ugpu")
index = []

HPERM = torch.cat([torch.arange(0, HD, 2), torch.arange(1, HD, 2)]).to(dev)
FULL = torch.cat([h * HD + HPERM for h in range(H)])

def put(name, kind, a, M, K):
    writer.add_tensor(name, np.ascontiguousarray(a))
    index.append(f"{name} {kind} {M} {K}")
def q8(name, w):
    M, K = w.shape
    w = (w.reshape(M, K // HT.shape[0], HT.shape[0]) @ HT).reshape(M, K)      # rot_weight
    t = w.view(M, K // GS, GS)
    s = t.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
    q = (t / s).round().clamp(-127, 127).to(torch.int8).view(M, K)
    put(name, "q8", q.view(M, K // 16, 16).permute(1, 0, 2).contiguous().cpu().numpy(), M, K)
    put(name + ".s", "qs", s.view(M, K // GS).contiguous().cpu().numpy().astype(np.float32), M, K // GS)
def q4(name, w):
    """4-bit: per row and block of GS inputs one scale; texel (o, k32) = 16 bytes, byte j = input j (low half) and
    input 16 + j (high half). The kernel reads each half as nibble * 16, so the stored scale carries the 1/16."""
    M, K = w.shape
    w = (w.reshape(M, K // HT.shape[0], HT.shape[0]) @ HT).reshape(M, K)
    t = w.view(M, K // GS, GS)
    s = t.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7
    q = (t / s).round().clamp(-8, 7).to(torch.int16).view(M, K // 32, 2, 16)
    b = ((q[:, :, 0] & 15) | ((q[:, :, 1] & 15) << 4)).to(torch.uint8)          # (M, K/32, 16)
    put(name, "q4", b.permute(1, 0, 2).contiguous().cpu().numpy().view(np.int8), M, K)
    put(name + ".s", "qs", (s / 16).view(M, K // GS).contiguous().cpu().numpy().astype(np.float32), M, K // GS)
def q4k(name, w):
    """4-bit with two-level scales (Q4K=1, for the Vulkan engine's 8-bit matrix hardware): per row and block of 256
    inputs one fp32 scale d ("<name>.s"), per 32 inputs a whole-number sub-scale sc 1..15 ("<name>.sc", one byte), and
    the values q -8..7 in the q4 layout above; weight = d * sc * q, and q * sc always fits in 8 bits. d is searched
    per 256 (0.75..1 x the largest sub-block scale / 15) for the smallest error. OpenCL folds d * sc / 16 back into
    plain per-32 scales as it loads."""
    M, K = w.shape
    w = (w.reshape(M, K // HT.shape[0], HT.shape[0]) @ HT).reshape(M, K)
    t = w.view(M, K // 256, 8, 32)
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
            g = e < best[0]
            best = [torch.where(g, e, best[0]), torch.where(g[..., None], d, best[1]), torch.where(g[..., None], sc, best[2]),
                    torch.where(g[..., None, None], q, best[3])]
    _, d, sc, q = best
    q = q.to(torch.int16).view(M, K // 32, 2, 16)
    b = ((q[:, :, 0] & 15) | ((q[:, :, 1] & 15) << 4)).to(torch.uint8)
    put(name, "q4", b.permute(1, 0, 2).contiguous().cpu().numpy().view(np.int8), M, K)
    put(name + ".s", "qs", d.view(M, K // 256).contiguous().cpu().numpy().astype(np.float32), M, K // 256)
    put(name + ".sc", "q4sc", sc.to(torch.uint8).view(M, K // 32).contiguous().cpu().numpy().view(np.int8), M, K // 32)
def pk(name, w):
    M, K = w.shape
    a = w.half().view(M // 4, 4, K // 2, 2).permute(2, 0, 3, 1).reshape(K // 2, M // 4, 8)
    put(name, "pk", a.contiguous().cpu().numpy(), M, K)
def f16(name, w):
    put(name, "f16", w.half().contiguous().cpu().numpy(), *w.shape)
def f32(name, v):
    put(name, "f32", v.float().contiguous().cpu().numpy(), v.numel(), 1)
def swiglu_rows(w):
    gt, up = w[:FF], w[FF:]
    return torch.stack([gt.view(FF // 8, 8, -1), up.view(FF // 8, 8, -1)], 1).reshape(2 * FF, -1)

qb = (q4k if os.environ.get("Q4K") else q4) if BITS == 4 else q8
with torch.no_grad():
    for i in range(NB):
        p, o = f"transformer_blocks.{i}.", f"b{i}."
        qb(o + "qkv", torch.cat([g(p + "attn.to_q.weight")[FULL], g(p + "attn.to_k.weight")[FULL], g(p + "attn.to_v.weight")]))
        qb(o + "o", g(p + "attn.to_out.0.weight"))
        f32(o + "qn", g(p + "attn.norm_q.weight")[HPERM]); f32(o + "kn", g(p + "attn.norm_k.weight")[HPERM])
        qb(o + "mlp_in", swiglu_rows(g(p + "img_mlp.gate_up.weight")))
        qb(o + "down", g(p + "img_mlp.out.weight"))
        print("block", i, flush=True)
    qb("txt_in1", g("txt_in.in_layer.weight")); qb("txt_in2", g("txt_in.out_layer.weight"))
    f32("txt_norm", g("txt_in.text_norm.weight") + 1.0)
    pk("img_in", g("img_in.weight"))
    pk("proj_out", g("proj_out.weight"))
    f16("t1", g("time_text_embed.timestep_embedder.linear_1.weight"))
    f16("t2", g("time_text_embed.timestep_embedder.linear_2.weight"))
    f16("mod", g("modulation.1.weight"))
    f16("out_scale", g("norm_out.linear.weight"))
writer.add_string("ugpu.model", "qwen")
writer.add_string("ugpu.index", "\n".join(index))
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print("wrote", OUT, os.path.getsize(OUT) / 1e9, "GB,", len(index), "tensors")
