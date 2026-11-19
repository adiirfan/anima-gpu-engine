"""Reference step for FLUX.2 Klein 4B on the computer (fp32, processor), for checking the phone engine.
A test prompt is encoded with the Qwen3-VL 4B reader, the picture is drawn for real for STEPS0 steps from seeded
noise, then the next step's inputs and outputs are saved in <out dir>:
  img.f32 (Ni x 128 patches), ctx.f32 (Lt x 7680), sigma.f32, rope_P.f32 / rope_Q.f32 (engine order: picture tokens,
  then prompt tokens, then padding to a multiple of 8; head channels in the engine's rotate-half order),
  out.f32 (Ni x 128), x_double0 / x_double_last / x_single0 .f32 (whole stream after those blocks, engine order),
  info.txt (Ni Lt Lpad W H).
Usage: py.sh ref_klein.py <out dir> [WxH]   env STEPS (4), STEPS0 (2), PROMPT"""
import os
import sys
sys.path[:0] = ["/home/debian/gpu-engine-test/tsbuild/common"]
from settings import use_comfyui
OUT = sys.argv[1]
SIZE = sys.argv[2] if len(sys.argv) > 2 else "512x512"
sys.argv = [sys.argv[0], "--cpu"]
use_comfyui()
import comfy.options
comfy.options.enable_args_parsing()
import numpy as np
import torch
import comfy.sd, comfy.samplers

torch.set_num_threads(12)
W, H = (int(v) for v in SIZE.split("x"))
STEPS, STEPS0 = int(os.environ.get("STEPS", "4")), int(os.environ.get("STEPS0", "2"))
PROMPT = os.environ.get("PROMPT", "a girl with short silver hair in a red coat standing in a snowy street at dusk, "
                                  "warm shop lights, anime illustration")
os.makedirs(OUT, exist_ok=True)
save = lambda n, t: t.detach().float().contiguous().numpy().astype(np.float32).tofile(f"{OUT}/{n}.f32")

TE = "/home/debian/ComfyUI/models/text_encoders/qwen3vl_4b_bf16.safetensors"
clip = comfy.sd.load_clip([TE], clip_type=comfy.sd.CLIPType.FLUX2)
# ComfyUI's Klein branch forgets to rename Qwen3-VL's keys, so the reader stays empty: load them renamed here
import comfy.utils
sd = comfy.utils.state_dict_prefix_replace(comfy.utils.load_torch_file(TE),
                                           {"model.language_model.": "model.", "model.visual.": "visual.", "lm_head.": "model.lm_head."})
m, u = clip.load_sd(sd)
print("reader: missing", len(m), [k for k in m][:5])
del sd
cond = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))[0][0].float()      # (1, Lt, 7680)
print("prompt: shape", tuple(cond.shape), "max", float(cond.abs().max()), "nan", int(cond.isnan().sum()))
assert not cond.isnan().any()
del clip
mp = comfy.sd.load_diffusion_model("/home/debian/models/klein/flux-2-klein-4b.safetensors", model_options={"dtype": torch.float32})
dm = mp.model.diffusion_model.float().eval()
sig = comfy.samplers.calculate_sigmas(mp.model.model_sampling, "simple", STEPS).float()
print("sigmas", [round(float(s), 4) for s in sig])

lat = torch.randn(1, 128, H // 16, W // 16, generator=torch.Generator().manual_seed(42))   # ComfyUI's Flux 2 latent: 128 ch at 1/16
x = lat * sig[0]
save("noise", x.view(128, -1).T); save("sigmas", sig)      # for DRAW on the phone: same start, same schedule
grab = {}
def hook(name):
    def f(m, inp, out):
        # copies: the blocks add into these tensors in place later, which would overwrite what we keep
        grab[name] = tuple(o.clone() for o in out) if isinstance(out, tuple) else out.clone()
    return f
nd, ns = len(dm.double_blocks), len(dm.single_blocks)
dm.double_blocks[0].register_forward_hook(hook("d0"))
dm.double_blocks[nd - 1].register_forward_hook(hook("dl"))
dm.single_blocks[0].register_forward_hook(hook("s0"))
orig_pe = dm.pe_embedder.forward
def pe_fwd(ids):
    grab["pe"] = orig_pe(ids)
    return grab["pe"]
dm.pe_embedder.forward = pe_fwd
with torch.no_grad():
    for s_i in range(STEPS0 + 1):
        s = sig[s_i]
        if s_i == STEPS0:   # the test step: keep its inputs
            save("sigma", s.view(1))
        out = dm(x, s.view(1), cond)          # velocity, (1, 128, H/16, W/16)
        if s_i == STEPS0:
            break
        x = x + (sig[s_i + 1] - s) * out
        print(f"step {s_i}: sigma {float(s):.4f}", flush=True)

    Lt = cond.shape[1]
    Ni = (W // 16) * (H // 16)
    Lp = (Ni + Lt + 7) // 8 * 8 - Ni - Lt
    # inputs
    img = x.view(128, Ni).T                                     # patch size 1: "b c h w -> b (h w) c"
    save("img", img); save("ctx", cond[0])
    outp = out.view(128, Ni).T
    save("out", outp)
    # rope: ComfyUI order (prompt, picture), pairs (2i, 2i+1) -> engine order (picture, prompt, pad), channels
    # rotate-half: engine channel i < 64 = pair i's first element, 64 + i = its second
    pe = grab["pe"][0, 0]            # (L, 64, 2, 2)
    f = torch.cat([pe[Lt:], pe[:Lt], torch.eye(2).expand(Lp, 64, 2, 2)], 0)
    P = torch.cat([f[:, :, 0, 0], f[:, :, 1, 1]], 1)
    Q = torch.cat([f[:, :, 0, 1], f[:, :, 1, 0]], 1)
    save("rope_P", P); save("rope_Q", Q)
    def eng(t):  # stream (1, Lt + Ni, 3072) in ComfyUI order -> engine order with zero padding rows
        t = t[0]
        return torch.cat([t[Lt:], t[:Lt], torch.zeros(Lp, t.shape[1])], 0)
    save("x_double0", eng(torch.cat([grab["d0"][1], grab["d0"][0]], 1)))   # double blocks return (img, txt)
    save("x_double_last", eng(torch.cat([grab["dl"][1], grab["dl"][0]], 1)))
    save("x_single0", eng(grab["s0"]))
    open(f"{OUT}/info.txt", "w").write(f"{Ni} {Lt} {Lp} {W} {H}\n")
    print("saved", OUT, "Ni", Ni, "Lt", Lt, "pad", Lp, "sigma", float(sig[STEPS0]))
