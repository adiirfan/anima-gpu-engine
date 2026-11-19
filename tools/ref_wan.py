"""Reference step for Wan 2.2 TI2V 5B on the computer (fp32, processor), for checking the phone engine.
The prompt goes through umt5-xxl, the video is drawn for real for STEPS0 steps from seeded noise, then the next step's
inputs and outputs are saved in <out dir> (tokens in ComfyUI's order: frame, row, column):
  img.f32 (L x 192 patches, index c*4 + dy*2 + dx), ctx.f32 (Lt x 4096), sigma.f32, sigmas.f32, noise.f32,
  rope_P.f32 / rope_Q.f32 (L x 128, head channels in the engine's rotate-half order), out.f32 (L x 192),
  x_block0 / x_block_mid / x_block_last .f32 (stream after those blocks), info.txt (L Lt T Hp Wp W H frames).
Usage: py.sh ref_wan.py <out dir> [WxHxF]   env STEPS (4), STEPS0 (2), PROMPT"""
import os
import sys
sys.path[:0] = ["/home/debian/gpu-engine-test/tsbuild/common"]
from settings import use_comfyui
OUT = sys.argv[1]
SIZE = sys.argv[2] if len(sys.argv) > 2 else "512x512x17"
sys.argv = [sys.argv[0]] + ([] if os.environ.get("GPU") else ["--cpu"])   # GPU=1: run on the graphics card
use_comfyui()
import comfy.options
comfy.options.enable_args_parsing()
import numpy as np
import torch
import comfy.sd, comfy.samplers

torch.set_num_threads(12)
W, H, F = (int(v) for v in SIZE.split("x"))
STEPS, STEPS0 = int(os.environ.get("STEPS", "4")), int(os.environ.get("STEPS0", "2"))
PROMPT = os.environ.get("PROMPT", "a girl with short silver hair in a red coat walks through a snowy street at dusk, "
                                  "warm shop lights, snow falling, anime illustration")
os.makedirs(OUT, exist_ok=True)
save = lambda n, t: t.detach().float().contiguous().numpy().astype(np.float32).tofile(f"{OUT}/{n}.f32")

clip = comfy.sd.load_clip(["/home/debian/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"],
                          clip_type=comfy.sd.CLIPType.WAN)
cond = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))[0][0].float()      # (1, Lt, 4096)
print("prompt: shape", tuple(cond.shape), "max", float(cond.abs().max()), "nan", int(cond.isnan().sum()), flush=True)
assert not cond.isnan().any()
del clip
mp = comfy.sd.load_diffusion_model("/home/debian/ComfyUI/models/diffusion_models/Wan2_2-TI2V-5B-Turbo_fp16.safetensors",
                                   model_options={"dtype": torch.float32})
DEV = "cuda" if os.environ.get("GPU") else "cpu"
dm = mp.model.diffusion_model.float().eval().to(DEV)
sig = comfy.samplers.calculate_sigmas(mp.model.model_sampling, "simple", STEPS).float()
if os.environ.get("SIGMAS"):    # a schedule file, e.g. the app's 8 levels (sigmas.txt), with 0 appended
    sig = torch.tensor([float(v) for v in open(os.environ["SIGMAS"]).read().split()] + [0.0])
    STEPS = len(sig) - 1
print("sigmas", [round(float(s), 4) for s in sig], flush=True)

T, Hl, Wl = (F - 1) // 4 + 1, H // 16, W // 16
lat = torch.randn(1, 48, T, Hl, Wl, generator=torch.Generator().manual_seed(42))
Hp, Wp = Hl // 2, Wl // 2
L = T * Hp * Wp
def patches(v):   # (1, 48, T, Hl, Wl) -> (L, 192), index c*4 + dy*2 + dx (the patch convolution's input order)
    return v.view(48, T, Hp, 2, Wp, 2).permute(1, 2, 4, 0, 3, 5).reshape(L, 192)
x = lat * sig[0]
save("noise", patches(x)); save("sigmas", sig)
# IMAGE=<picture>: image-to-video as the app does it - the picture is the first latent frame, kept clean (timestep 0)
# at every step while the rest is drawn
first = None
if os.environ.get("IMAGE"):
    import comfy.utils, comfy.latent_formats
    from PIL import Image
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file("/home/debian/ComfyUI/models/vae/wan2.2_vae.safetensors"))
    im = torch.from_numpy(np.array(Image.open(os.environ["IMAGE"]).convert("RGB").resize((W, H)))).float()[None] / 255
    first = comfy.latent_formats.Wan22().process_in(vae.encode(im).float().cpu())[:, :, :1]     # (1, 48, 1, Hl, Wl)
    del vae
    x[:, :, :1] = first
    save("first", first.view(48, Hp, 2, Wp, 2).permute(1, 3, 0, 2, 4).reshape(Hp * Wp, 192))
grab = {}
def hook(name):
    def f(m, inp, out):
        grab[name] = out.detach().float().cpu()
    return f
nb = len(dm.blocks)
dm.blocks[0].register_forward_hook(hook("b0"))
dm.blocks[nb // 2].register_forward_hook(hook("bm"))
dm.blocks[nb - 1].register_forward_hook(hook("bl"))
orig_pe = dm.rope_embedder.forward
def pe_fwd(ids):
    grab["pe"] = orig_pe(ids)
    return grab["pe"]
dm.rope_embedder.forward = pe_fwd
ts = mp.model.model_sampling.timestep
with torch.no_grad():
    for s_i in range(STEPS0 + 1):
        s = sig[s_i]
        if s_i == STEPS0:
            save("sigma", s.view(1))
        tt = ts(s.view(1))
        if first is not None:           # one timestep per frame: 0 for the picture's frame
            tt = tt.view(1, 1).repeat(1, T); tt[:, 0] = 0
        out = dm(x.to(DEV), tt.to(DEV), cond.to(DEV)).float().cpu()    # velocity, (1, 48, T, Hl, Wl)
        if s_i == STEPS0:
            if os.environ.get("FULL"):     # the whole video: finish this last step and keep the latent
                xf = x + (sig[s_i + 1] - s) * out
                if first is not None: xf[:, :, :1] = first
                save("final", patches(xf))
            break
        x = x + (sig[s_i + 1] - s) * out
        if first is not None: x[:, :, :1] = first
        print(f"step {s_i}: sigma {float(s):.4f}", flush=True)
    print("timestep fed to the model:", float(ts(sig[STEPS0].view(1))))

    Lt = cond.shape[1]
    save("img", patches(x)); save("ctx", cond[0]); save("out", patches(out))
    pe = (grab["pe"][0, 0] if grab["pe"].ndim == 5 else grab["pe"][0]).float().cpu()     # (L, 64, 2, 2)
    pe = pe.reshape(L, 64, 2, 2)
    P = torch.cat([pe[:, :, 0, 0], pe[:, :, 1, 1]], 1)
    Q = torch.cat([pe[:, :, 0, 1], pe[:, :, 1, 0]], 1)
    save("rope_P", P); save("rope_Q", Q)
    save("x_block0", grab["b0"][0]); save("x_block_mid", grab["bm"][0]); save("x_block_last", grab["bl"][0])
    open(f"{OUT}/info.txt", "w").write(f"{L} {Lt} {T} {Hp} {Wp} {W} {H} {F}\n")
    print("saved", OUT, "L", L, "Lt", Lt, "sigma", float(sig[STEPS0]), "mid block", nb // 2)
