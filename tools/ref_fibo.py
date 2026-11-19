"""Reference for model_fibo.h: Fibo 2.0 edit-turbo in fp32 on the graphics card (fibo_model, one block on the card at
a time, as ~/fibo-build/check_real.py), Euler on the edit-turbo's 4 levels from seeded noise, edit picture and prompt
from ~/fibo-build/ctx_real.pt. Writes into <out dir>:
  ctx.f32 (T x 46080 reader bundles), ref.f32 (R x 128 edit picture latent), noise.f32 (N x 128), sigmas.f32,
  out.f32 (direction at the first level), final.f32, info.txt "T G h w".
Usage: ref_fibo.py <out dir>"""
import os, sys, torch
sys.path.insert(0, "/home/debian/ComfyUI"); sys.path.insert(0, "/home/debian/fibo-build")
OUT = sys.argv[1]
sys.argv = [sys.argv[0]]
import numpy as np
import comfy.options; comfy.options.enable_args_parsing()
import comfy.sd, comfy.utils, comfy.latent_formats
from safetensors import safe_open
import fibo_model as fm
torch.set_grad_enabled(False)
dev = torch.device("cuda")
os.makedirs(OUT, exist_ok=True)
save = lambda n, t: t.detach().float().cpu().contiguous().numpy().astype(np.float32).tofile(f"{OUT}/{n}.f32")
f = safe_open("/home/debian/fibo-build/models/fibo2_edit_turbo_bf16.safetensors", "pt")
pre = next((p for p in ("model.diffusion_model.", "") if f"{p}x_embedder.weight" in f.keys()), "")
sd = lambda k: f.get_tensor(pre + k)
data = torch.load("/home/debian/fibo-build/ctx_real.pt")
ctx, img = data["ctx"].float(), data["img"]
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file("/home/debian/ComfyUI/models/vae/flux2-vae.safetensors"))
ref = comfy.latent_formats.Flux2().process_in(vae.encode(img[None]).float())[0]
del vae; torch.cuda.empty_cache()
h, w = ref.shape[1:]
ref = ref.flatten(1).T.contiguous()
NL, NR = fm.NL, fm.NR
fm.NL = fm.NR = 0
top = fm.load(fm.Model(), sd, blocks=set())
fm.NL, fm.NR = NL, NR
t0 = top.temb(fm.temb_sin(0.0))
top = top.to(dev)
def block(kind, i):
    b = fm.Block(kind == "layers" and i in fm.INJ)
    p = f"{'noise_refiner' if kind == 'refiner' else 'layers'}.{i}."
    wt, bb = sd(p + "adaLN_modulation.0.weight").float(), sd(p + "adaLN_modulation.0.bias").float()
    s1, g1, s2, g2 = (t0 @ wt.T + bb).chunk(4, -1)
    fm._fold_block(b, sd, p, (s1[0], g1[0].tanh(), s2[0], g2[0].tanh()))
    return b.eval().to(dev)
T = ctx.shape[0]; G = fm.gist_count(T)
gid, iid, rids = fm.ids_of(G, h, w)
PQ = [tuple(t.to(dev) for t in fm.rope_tables(i)) for i in (gid, iid, rids[0])]
sig = [1.0, 0.904531, 0.759511, 0.512844, 0.0]
z = torch.randn(h * w, 128, generator=torch.Generator().manual_seed(1))
save("ctx", ctx); save("ref", ref); save("noise", z); save("sigmas", torch.tensor(sig[:-1]))
open(f"{OUT}/info.txt", "w").write(f"{T} {G} {h} {w}\n")
blocks = [("refiner", i) for i in range(NR)] + [("layers", i) for i in range(NL)]
bundles = ctx.to(dev).chunk(1 + len(fm.INJ), -1)
xg0 = top.gist(bundles[0], G, None)
ref = ref.to(dev); z = z.to(dev)
for s in range(4):
    temb = top.temb(fm.temb_sin(sig[s]).to(dev))
    xi, xr, xg = top.x_in(z), top.x_in(ref), xg0
    for kind, i in blocks:
        b = block(kind, i)
        if kind == "refiner":
            xi, xr = b([xi, xr], 0, PQ[1:], [None, None], temb)
        else:
            ckv = b.cross.kv(bundles[1 + fm.INJ.index(i)]) if b.inj else None
            xg, xi, xr = b([xg, xi, xr], 1, PQ, [None, None, None], temb, ckv, None)
        del b
    v = top.final(xi, temb)
    if s == 0: save("out", v)
    z = z + (sig[s + 1] - sig[s]) * v
    print("step", s + 1, flush=True)
save("final", z)
print("saved", OUT, "T", T, "G", G, "picture", h, w)
