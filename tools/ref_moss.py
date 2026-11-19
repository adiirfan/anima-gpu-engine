"""Reference for model_moss.h: MOSS-Image + the 8-step add-on (0.7, separate tables, matches ComfyUI merged to 2e-4)
in fp32 on the graphics card, Euler on the add-on's 8 levels, no guidance. Writes into <out dir>:
  noise.f32 (N x 128 latent rows, row-major y, x), ctx.f32 (T x 4096 prompt rows from the reader), sigmas.f32,
  out.f32 (the direction at the first level), final.f32 (the finished latent), info.txt "T W H".
Usage: ref_moss.py <out dir> [WxH, default 768x768]   env PROMPT_PT (default ~/moss-build/ref/cat.pt), SEED"""
import os, sys, torch
sys.path.insert(0, "/home/debian/moss-build")
torch.set_grad_enabled(False)
import numpy as np
import moss_model as M
from safetensors import safe_open
OUT = sys.argv[1]
W, H = (int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "768x768").split("x"))
h, w = H // 16, W // 16
dev = "cuda"
os.makedirs(OUT, exist_ok=True)
save = lambda n, t: t.detach().float().cpu().contiguous().numpy().astype(np.float32).tofile(f"{OUT}/{n}.f32")
f = safe_open("/home/debian/moss-build/models/moss_image_sft_bf16.safetensors", "pt")
sig = [float(v) for v in open("/home/debian/moss-build/dmd/sigmas8.txt").read().split()] + [0.0]
ctx = torch.load(os.environ.get("PROMPT_PT", "/home/debian/moss-build/ref/cat.pt"))["h"].float()
m = M.load(M.Model(), lambda k: f.get_tensor(k)).eval()
M.load_lora(m, "/home/debian/moss-build/dmd/lora_latest.safetensors", 0.7)
m = m.to(dev)
_rt = M.rope_tables
M.rope_tables = lambda ids: tuple(t.to(dev) for t in _rt(ids))
z = torch.randn(h * w, 128, generator=torch.Generator().manual_seed(int(os.environ.get("SEED", 12345)))).to(dev)
save("noise", z); save("ctx", ctx); save("sigmas", torch.tensor(sig[:-1]))
ctx = ctx.to(dev)
for i in range(8):
    v = m(z, ctx, M.temb_sin(sig[i]).to(dev), h, w)
    if i == 0: save("out", v)
    z = z + (sig[i + 1] - sig[i]) * v
    print("step", i + 1, flush=True)
save("final", z)
open(f"{OUT}/info.txt", "w").write(f"{ctx.shape[0]} {W} {H}\n")
print("saved", OUT, "prompt rows", ctx.shape[0])
