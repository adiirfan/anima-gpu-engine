"""Fibo 2.0 edit-turbo -> one GGUF for the universal phone engine (2026-11-18). 8-bit ConvRot (BITS=4 for 4-bit),
meant to be STREAMED (STREAM=1): ~8 GB, more than a phone holds at once.
Blocks are numbered in running order: b0, b1 = the noise refiner, b2..b45 = the 44 main layers (injection layers
6, 14, 22, 30, 38 are b8, b16, b24, b32, b40).
Per block: qkv (q, k rows reordered to the half-split rotation layout), o, mlp_in ([gate 8][up 8] per 16 rows), down;
qn, kn (reordered); mod (4D x 256, fp16) + mod.b; the norm weights n1 n2 f1 f2 and z1 z2 zf1 zf2 = the same with the
t = 0 modulation folded in (for gist and edit-picture rows; gates through tanh). Injection blocks also: cq, ckv
(k rows then v rows, 7680 inputs), co, cqn, ckn, cn1, cn2 (= norm weight * tanh(gate)).
Perceiver p0 / p1: cq, ckv, co, cqn, ckn (cross to bundle 0), sq, skv, so, sqn, skn (self), mlp_in, down, cn, sn, fn;
queries (512 x D), pnorm. Small: x_in (padded to 256 inputs) + b, t1 t2 out_scale (fp16) + b, out + b.
Usage: python pack_fibo.py <out.gguf>   (ComfyUI's ROCm venv)"""
import os
import sys
sys.path[:0] = ["/home/debian/gpu-engine-test/tsbuild/anima", "/home/debian/llama.cpp/gguf-py"]
import numpy as np
import torch
from safetensors import safe_open
import anima_qnn
import gguf

OUT = sys.argv[1]
GS, BITS = 32, int(os.environ.get("BITS", "8"))
SRC = "/home/debian/fibo-build/models/fibo2_edit_turbo_bf16.safetensors"
D, H, HD, FF, NL, NR = 3584, 28, 128, 9728, 44, 2
INJ = (6, 14, 22, 30, 38)
dev = "cuda" if torch.cuda.is_available() else "cpu"
HT = anima_qnn._HT.to(dev)
f = safe_open(SRC, "pt")
pre = next((p for p in ("model.diffusion_model.", "") if f"{p}x_embedder.weight" in f.keys()), "")
g = lambda k: f.get_tensor(pre + k).to(dev).float()
writer = gguf.GGUFWriter(OUT, "ugpu")
index = []
PERM = torch.cat([torch.arange(0, HD, 2), torch.arange(1, HD, 2)]).to(dev)
FULL = torch.cat([h * HD + PERM for h in range(H)])

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
qb = q4 if BITS == 4 else q8
def pair_rows(gate, up):
    n = gate.shape[0]
    return torch.stack([gate.view(n // 8, 8, -1), up.view(n // 8, 8, -1)], 1).reshape(2 * n, -1)
def cross(o, c, ctxin=True):
    """cross / self attention c -> o{q,kv,o,qn,kn}"""
    kv = g(c + "to_kv.weight")
    qb(o + "q", g(c + "to_q.weight")[FULL])
    qb(o + "kv", torch.cat([kv[:D][FULL], kv[D:]], 0))
    qb(o + "o", g(c + "to_out.weight"))
    f32(o + "qn", g(c + "norm_q.weight")[PERM]); f32(o + "kn", g(c + "norm_k.weight")[PERM])

with torch.no_grad():
    t1w, t1b, t2w, t2b = g("t_embedder.mlp.0.weight"), g("t_embedder.mlp.0.bias"), g("t_embedder.mlp.2.weight"), g("t_embedder.mlp.2.bias")
    half = 128
    fr = torch.exp(-np.log(10000) * torch.arange(half, dtype=torch.float32, device=dev) / half)
    ts0 = torch.cat([torch.cos(0 * fr), torch.sin(0 * fr)])[None]
    t0 = torch.nn.functional.silu(ts0 @ t1w.T + t1b) @ t2w.T + t2b            # temb at t = 0
    blocks = [("noise_refiner", i) for i in range(NR)] + [("layers", i) for i in range(NL)]
    for k, (kind, i) in enumerate(blocks):
        p, o = f"{kind}.{i}.", f"b{k}."
        qkv = g(p + "attention.qkv.weight")
        qb(o + "qkv", torch.cat([qkv[:D][FULL], qkv[D:2 * D][FULL], qkv[2 * D:]], 0))
        qb(o + "o", g(p + "attention.out.weight"))
        w13 = g(p + "feed_forward.w13.weight")
        qb(o + "mlp_in", pair_rows(w13[:FF], w13[FF:]))
        qb(o + "down", g(p + "feed_forward.w2.weight"))
        f32(o + "qn", g(p + "attention.q_norm.weight")[PERM]); f32(o + "kn", g(p + "attention.k_norm.weight")[PERM])
        mw, mb = g(p + "adaLN_modulation.0.weight"), g(p + "adaLN_modulation.0.bias")
        f16(o + "mod", mw); f32(o + "mod.b", mb)
        s1, g1, s2, g2 = (t0 @ mw.T + mb)[0].chunk(4)
        n1, n2, f1, f2 = (g(p + n + ".weight") for n in ("attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2"))
        for n, v in (("n1", n1), ("n2", n2), ("f1", f1), ("f2", f2), ("z1", n1 * (1 + s1)), ("z2", n2 * g1.tanh()),
                     ("zf1", f1 * (1 + s2)), ("zf2", f2 * g2.tanh())):
            f32(o + n, v)
        if kind == "layers" and i in INJ:
            cross(o + "c", p + "cross_attention.")
            f32(o + "cn1", g(p + "cross_attn_norm1.weight"))
            f32(o + "cn2", g(p + "cross_attn_norm2.weight") * torch.tanh(g(p + "cross_attn_gate")))
        print("block", k, flush=True)
    for j in range(2):
        p, o = f"perceiver.layers.{j}.", f"p{j}."
        cross(o + "c", p + "cross_attention."); cross(o + "s", p + "self_attention.")
        w13 = g(p + "ffn.w13.weight")
        qb(o + "mlp_in", pair_rows(w13[:FF], w13[FF:])); qb(o + "down", g(p + "ffn.w2.weight"))
        f32(o + "cn", g(p + "cross_attn_norm.weight")); f32(o + "sn", g(p + "self_attn_norm.weight"))
        f32(o + "fn", g(p + "ffn_norm.weight"))
    f32("queries", g("perceiver.latent_queries")); f32("pnorm", g("perceiver.norm_out.weight"))
    xw = g("x_embedder.weight")
    q8("x_in", torch.cat([xw, torch.zeros_like(xw)], 1), g("x_embedder.bias"))
    f16("t1", t1w); f32("t1.b", t1b); f16("t2", t2w); f32("t2.b", t2b)
    f16("out_scale", g("final_layer.adaLN_modulation.1.weight")); f32("out_scale.b", g("final_layer.adaLN_modulation.1.bias"))
    q8("out", g("final_layer.linear.weight"), g("final_layer.linear.bias"))
writer.add_string("ugpu.model", "fibo")
writer.add_string("ugpu.index", "\n".join(index))
writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
print("wrote", OUT, os.path.getsize(OUT) / 1e9, "GB,", len(index), "tensors")
