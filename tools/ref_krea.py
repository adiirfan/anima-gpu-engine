# Krea 2 Turbo reference on the box (ComfyUI's model code, its ROCm venv, the R9700): writes what the phone engine
# is checked against. Usage (from ~/ComfyUI, with ~/ComfyUI/venv/bin/python):
#   ref_krea.py <out dir> "<prompt>" [seed] [steps] [WxH]
# Out dir gets: ctx.f32 (rows x 12*2560, the prompt reader's 12 tapped layers), txt.f32 (rows x 6144, after the
# model's text fusion + text MLP), noise.f32 (16 x h x w, model space), sigmas.f32, out0.f32 (first step's model
# output), x_final.f32 (16 x h x w, model space), lat.f32 + info.txt (2h x 2w... for the small Wan 2.1 decoder:
# h x w x 16, channels innermost).
# Weights: ~/models/krea2-turbo (diffusers names, translated here); prompt reader: ComfyUI's qwen3vl_4b_bf16
# (byte-identical to Krea's text_encoder). QKONE=1: the q/k norm weights are stored as the full weight (1 taken off).
import os, sys, json, glob, math
sys.path.insert(0, os.path.expanduser("~/ComfyUI"))
import numpy as np
import torch
from safetensors import safe_open
import comfy.sd, comfy.ops, comfy.model_management, comfy.latent_formats
import comfy.ldm.krea2.model as K

out = sys.argv[1]; prompt = sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 8
W, H = (int(v) for v in (sys.argv[5] if len(sys.argv) > 5 else "512x512").split("x"))
os.makedirs(out, exist_ok=True)
dev = torch.device("cuda")
dt = torch.bfloat16
def save(name, t): t.detach().float().cpu().contiguous().numpy().tofile(os.path.join(out, name))

# ---- prompt reader: 12 tapped layers of Qwen3-VL-4B, ComfyUI's Krea template and prefix strip ----
clip = comfy.sd.load_clip(ckpt_paths=[os.path.expanduser("~/ComfyUI/models/text_encoders/qwen3vl_4b_bf16.safetensors")],
                          clip_type=comfy.sd.CLIPType.KREA2)
cond = clip.encode_from_tokens(clip.tokenize(prompt))   # (1, seq, 12*2560)
cond = cond[0] if isinstance(cond, (tuple, list)) else cond
print("prompt rows", cond.shape[1]); save("ctx.f32", cond[0])
del clip; comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()

# ---- the drawing model, weights translated from the diffusers names ----
model = K.SingleStreamDiT(device=dev, dtype=dt, operations=comfy.ops.disable_weight_init).eval()
blk = {"attn.to_q": "attn.wq", "attn.to_k": "attn.wk", "attn.to_v": "attn.wv", "attn.to_gate": "attn.gate",
       "attn.to_out.0": "attn.wo", "ff.gate": "mlp.gate", "ff.up": "mlp.up", "ff.down": "mlp.down",
       "norm1": "prenorm.scale", "norm2": "postnorm.scale", "attn.norm_q": "attn.qknorm.qnorm.scale",
       "attn.norm_k": "attn.qknorm.knorm.scale", "scale_shift_table": "mod.lin"}
basic = {"img_in": "first", "time_embed.linear_1": "tmlp.0", "time_embed.linear_2": "tmlp.2", "time_mod_proj": "tproj.1",
         "txt_in.norm": "txtmlp.0.scale", "txt_in.linear_1": "txtmlp.1", "txt_in.linear_2": "txtmlp.3",
         "text_fusion.projector": "txtfusion.projector", "final_layer.linear": "last.linear",
         "final_layer.norm": "last.norm.scale", "final_layer.scale_shift_table": "last.modulation.lin"}
def name(k):
    stem, leaf = k.rsplit(".", 1)
    for pre_d, pre_c in (("transformer_blocks.", "blocks."), ("text_fusion.layerwise_blocks.", "txtfusion.layerwise_blocks."),
                         ("text_fusion.refiner_blocks.", "txtfusion.refiner_blocks.")):
        if stem.startswith(pre_d):
            i, rest = stem[len(pre_d):].split(".", 1) if "." in stem[len(pre_d):] else (stem[len(pre_d):], "")
            if k.endswith("scale_shift_table"): return pre_c + i + ".mod.lin"
            c = blk[rest]
            return pre_c + i + "." + c + ("" if c.endswith(".scale") else "." + leaf)
    if k == "final_layer.scale_shift_table": return "last.modulation.lin"
    c = basic[stem]
    return c + ("" if c.endswith(".scale") or c.endswith(".lin") else "." + leaf)
params = dict(model.named_parameters())
seen = set()
for f in sorted(glob.glob(os.path.expanduser("~/models/krea2-turbo/transformer/*.safetensors"))):
    with safe_open(f, "pt") as s:
        for k in s.keys():
            n = name(k); t = s.get_tensor(k)
            if os.environ.get("QKONE") and ("norm_q" in k or "norm_k" in k): t = t.float() - 1.0
            p = params[n]
            with torch.no_grad(): p.copy_(t.reshape(p.shape).to(p.dtype))
            seen.add(n)
missing = [n for n in params if n not in seen]
print("weights loaded,", len(seen), "missing:", missing[:5])
assert not missing

# W4=1: the drawing blocks' tables rounded as the AI chip's 4-bit build does (symmetric int4, blocks of 32 inputs per
# output, each block's range clipped at the best of 0.5..1.0 x its largest value), to see the picture it gives
if os.environ.get("W4"):
    def fake4(w, block=32):
        o, i = w.shape
        out = torch.empty_like(w)
        for r0 in range(0, o, 1024):
            wb = w[r0:r0 + 1024].float().reshape(-1, i // block, block)
            amax = wb.abs().amax(-1, keepdim=True)
            best_err = best_q = None
            for c in torch.linspace(0.5, 1.0, 11):
                sc = (amax * c).clamp_min(1e-8) / 7
                q = (wb / sc).round().clamp(-8, 7) * sc
                err = ((q - wb) ** 2).sum(-1, keepdim=True)
                if best_err is None: best_err, best_q = err, q
                else:
                    better = err < best_err
                    best_err = torch.where(better, err, best_err); best_q = torch.where(better, q, best_q)
            out[r0:r0 + 1024] = best_q.reshape(-1, i).to(w.dtype)
        return out
    n = 0
    with torch.no_grad():
        for name, mod in list(model.blocks.named_modules()) + (list(model.txtfusion.named_modules()) if os.environ.get("W4TXT") else []):
            if isinstance(mod, torch.nn.Linear) and mod.weight.shape[1] % 32 == 0:
                mod.weight.copy_(fake4(mod.weight)); n += 1
    print("4-bit rounding on", n, "tables", flush=True)

# ---- sampling: Euler on the flow (x - sigma * v is the clean picture), Krea's exponential shift mu 1.15 ----
mu = 1.15
t = np.linspace(1.0, 1.0 / 1000, steps)
sig = np.exp(mu) / (np.exp(mu) + (1.0 / t - 1.0))
sig = np.append(sig, 0.0).astype(np.float32)
sig.tofile(os.path.join(out, "sigmas.f32")); print("sigmas", np.round(sig, 4))
h, w = H // 8, W // 8
g = torch.Generator().manual_seed(seed)
x = torch.randn((1, 16, h, w), generator=g).to(dev, torch.float32)
save("noise.f32", x[0])
ctx = cond.to(dev, dt)
with torch.no_grad():
    txt = model.txtmlp(model.txtfusion(model._unpack_context(ctx), mask=None))
    save("txt.f32", txt[0])
    # engine checkpoints for the first step (rows in the engine's order: picture, prompt, padding to a multiple of 8)
    Lt = txt.shape[1]; Ni = (h // 2) * (w // 2); Lp = (Ni + Lt + 7) // 8 * 8 - Ni - Lt
    grab = {}
    def hook(name):
        def f(mod, inp, out): grab.setdefault(name, out.detach().float())
        return f
    hs = [model.blocks[0].register_forward_hook(hook("b0")), model.blocks[-1].register_forward_hook(hook("bl")),
          model.tmlp.register_forward_hook(hook("t")), model.tproj.register_forward_hook(hook("tvec"))]
    def eng(t):
        t = t[0]
        return torch.cat([t[Lt:], t[:Lt], torch.zeros(Lp, t.shape[1], device=t.device)], 0)
    patches = lambda z: z[0].reshape(16, h // 2, 2, w // 2, 2).permute(1, 3, 0, 2, 4).reshape(Ni, 64)
    for i in range(steps):
        s0, s1 = float(sig[i]), float(sig[i + 1])
        v = model(x.to(dt), torch.tensor([s0], device=dev, dtype=dt), ctx).float()
        if i == 0:
            save("out0.f32", v[0]); save("img0.f32", patches(x)); save("out0p.f32", patches(v))
            save("x_block0.f32", eng(grab["b0"])); save("x_blocklast.f32", eng(grab["bl"]))
            save("t.f32", grab["t"].reshape(-1)); save("tvec.f32", grab["tvec"].reshape(-1))
            open(os.path.join(out, "engine.txt"), "w").write(f"{Ni} {Lt} {Lp} {W} {H}\n")
            for hh in hs: hh.remove()
        x = x + (s1 - s0) * v
        print(f"step {i}: sigma {s0:.4f} -> {s1:.4f}, |v| {v.std().item():.3f}", flush=True)
save("x_final.f32", x[0])
# for the small Wan 2.1 decoder: model-space latent, (h, w, 16) channels innermost
x[0].permute(1, 2, 0).contiguous().float().cpu().numpy().tofile(os.path.join(out, "lat.f32"))
open(os.path.join(out, "info.txt"), "w").write(f"{h} {w}\n")
print("done", out)
