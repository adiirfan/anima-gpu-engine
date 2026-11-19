"""Reference for the small encoder on the engine (model_tae.h with an ENC=1 pack): encodes the Fibo test's edit picture
with the TAESD reference code (fp32) and with the full Flux 2 VAE (what ref_fibo.py uses, ref.f32).
Writes <out>/lat.f32 (the picture, H x W x 4 in [0, 1], 4th channel 0), <out>/rgb.f32 (the small encoder's latent,
h x w x 32), <out>/info.txt "H W", <out>/full.f32 (the full VAE's latent, unpatchified to h x w x 32).
Usage: ref_taee.py <taef2 file> <out dir>"""
import os, sys
sys.path[:0] = ["/home/debian/phone-v81/qwen/build/common"]
import numpy as np, torch
from safetensors.torch import load_file
from taesd import TAESD
src, out = sys.argv[1], sys.argv[2]
t = TAESD(encoder_path=None, decoder_path=None, latent_channels=32, arch_variant="flux_2", image_channels=3)
sd = load_file(src)
t.encoder.load_state_dict({k[len("encoder.layers."):]: v.float() for k, v in sd.items() if k.startswith("encoder.")})
img = torch.load("/home/debian/fibo-build/ctx_real.pt")["img"].float()       # H x W x 3 in [0, 1]
H, W = img.shape[:2]
os.makedirs(out, exist_ok=True)
x = torch.cat([img, torch.zeros(H, W, 1)], -1)
x.numpy().astype(np.float32).tofile(f"{out}/lat.f32")
with torch.no_grad():
    z = t.encoder(img.permute(2, 0, 1)[None])[0]                                # 32 x H/8 x W/8
z.permute(1, 2, 0).contiguous().numpy().astype(np.float32).tofile(f"{out}/rgb.f32")
ref = np.fromfile("/home/debian/ugpu/ref_fibo/ref.f32", np.float32)              # (h/2 * w/2) x 128, patchified
h2, w2 = H // 16, W // 16
full = ref.reshape(h2, w2, 32, 2, 2).transpose(0, 3, 1, 4, 2).reshape(h2 * 2, w2 * 2, 32)
full.astype(np.float32).tofile(f"{out}/full.f32")
zz = z.permute(1, 2, 0).numpy()
print("small vs full encoder: relative error", np.linalg.norm(zz - full) / np.linalg.norm(full))
open(f"{out}/info.txt", "w").write(f"{H} {W}\n")
print("saved", out, tuple(z.shape))
