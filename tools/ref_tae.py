"""Reference for model_tae.h: decode one latent with the TAESD reference code (fp32).
Writes <out>/lat.f32 (h, w, C), <out>/rgb.f32 (H, W, first 3 channels, not clamped), <out>/info.txt "h w".
Usage: ref_tae.py <taesd file> <latent channels> <arch or ""> <image channels> <latent .f32 (h*w, C)> <h> <w> <out dir>"""
import os, sys
sys.path[:0] = ["/home/debian/phone-v81/qwen/build/common"]
import numpy as np, torch
from safetensors.torch import load_file
from taesd import TAESD
src, LC, ARCH, IC, latf, h, w, out = sys.argv[1], int(sys.argv[2]), sys.argv[3] or None, int(sys.argv[4]), sys.argv[5], int(sys.argv[6]), int(sys.argv[7]), sys.argv[8]
t = TAESD(encoder_path=None, decoder_path=None, latent_channels=LC, arch_variant=ARCH, image_channels=IC)
sd = load_file(src)
if any(k.startswith("decoder.layers.") for k in sd):
    sd = {".".join([e, str(int(i) + (1 if e == "decoder" else 0)), *s]): v for k, v in sd.items() for e, _l, i, *s in [k.split(".")]}
t.load_state_dict({k: v.float() for k, v in sd.items()})
os.makedirs(out, exist_ok=True)
z = np.fromfile(latf, dtype=np.float32).reshape(h, w, LC)
z.tofile(f"{out}/lat.f32")
with torch.no_grad():
    im = t.decoder(torch.from_numpy(z).permute(2, 0, 1)[None])[0, :3]
im.permute(1, 2, 0).contiguous().numpy().astype(np.float32).tofile(f"{out}/rgb.f32")
open(f"{out}/info.txt", "w").write(f"{h} {w}\n")
print("saved", out, tuple(im.shape))
