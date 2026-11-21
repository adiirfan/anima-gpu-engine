"""The Wan 2.1 tiny decoder (taew2_1, ComfyUI's TAEHV) for one picture -> one GGUF for the engine (model_tae.h).
One latent frame grows into 4 frames inside the decoder (TGrow); each "memory" block of a frame also reads the same
block's input of the frame before (zeros for the first); the picture is the last frame (as the AI-chip export,
gpu-engine-test/tsbuild/anima/export_dec.py). Ops on top of pack_tae.py's:
  mem <name> <ci> <co> <skip>       relu(c2(relu(c1(relu(c0x(x) + c0p(past))))) + skip(x))
  tgrow <name> <c> <stride> <up>    (input doubled first if up) 1x1 filters <name>.<j>: frame f -> frames f*stride + j
Wan 2.2's (lighttaew2_2 / taew2_2: 48 latent channels) folds its output 2 x 2 ("shuffle 2") and the whole clip is kept
("frames all": every frame, the first 2^tgrows - 1 dropped as ComfyUI does).
Usage: pack_taehv.py <taew2_1 or (light)taew2_2 .safetensors> <out.gguf> [latent channels, default 16]"""
import sys
sys.path[:0] = ["/home/debian/ComfyUI", "/home/debian/llama.cpp/gguf-py"]
sys.argv, args = [sys.argv[0], "--cpu"], sys.argv[1:]
import comfy.options; comfy.options.enable_args_parsing()
import numpy as np, torch, torch.nn as nn
import comfy.utils
from comfy.taesd.taehv import TAEHV, MemBlock, TGrow, Clamp
import gguf

SRC, OUT = args[:2]
LC = int(args[2]) if len(args) > 2 else 16
t = TAEHV(latent_channels=LC, latent_format=None)
missing = t.load_state_dict({k: v.float() for k, v in comfy.utils.load_torch_file(SRC).items()}, strict=False).missing_keys
if [k for k in missing if k.startswith("decoder.")]: raise SystemExit(f"missing {missing[:4]}")
writer = gguf.GGUFWriter(OUT, "ugpu")
index, prog = [], []
def put(name, kind, a, M, K):
    writer.add_tensor(name, np.ascontiguousarray(a)); index.append(f"{name} {kind} {M} {K}")
def filtw(name, w, b):
    w = w.detach().float().permute(0, 2, 3, 1).contiguous()          # [out][ky][kx][in]
    if w.shape[-1] % 4: w = torch.cat([w, w.new_zeros(*w.shape[:-1], 4 - w.shape[-1] % 4)], -1)
    co = (w.shape[0] + 3) // 4 * 4
    oc = w.shape[0]
    w = torch.cat([w, w.new_zeros(co - oc, *w.shape[1:])])
    put(name, "f16", w.half().reshape(co, -1).numpy(), co, w[0].numel())
    if b is not None:
        put(name + ".b", "f32", torch.cat([b.detach().float(), torch.zeros(co - oc)]).numpy(), co, 1)
    return co
def ops_of(m): return m.operations if hasattr(m, "operations") else m
layers, up, i = list(t.decoder), 0, 0
while i < len(layers):
    m = layers[i]
    relu_next = i + 1 < len(layers) and isinstance(layers[i + 1], nn.ReLU)
    if isinstance(m, Clamp): prog.append("clamp")
    elif isinstance(m, nn.Upsample): up = 1 if m.scale_factor in (2, 2.0) else 0
    elif isinstance(m, nn.Conv2d):
        co = filtw(f"l{i}", m.weight, m.bias)
        prog.append(f"conv l{i} {(m.in_channels + 3) // 4 * 4} {co} {m.kernel_size[0]} {int(m.bias is not None)} {int(relu_next)} {up}")
        up = 0; i += int(relu_next)
    elif isinstance(m, MemBlock):
        c = [x for x in m.conv if isinstance(x, nn.Conv2d)]
        n = c[0].in_channels // 2
        filtw(f"l{i}.c0x", c[0].weight[:, :n], c[0].bias); filtw(f"l{i}.c0p", c[0].weight[:, n:], None)
        filtw(f"l{i}.c1", c[1].weight, c[1].bias); filtw(f"l{i}.c2", c[2].weight, c[2].bias)
        skip = int(isinstance(m.skip, nn.Conv2d))
        if skip: filtw(f"l{i}.skip", m.skip.weight, None)
        prog.append(f"mem l{i} {n} {c[0].out_channels} {skip}")
    elif isinstance(m, TGrow):
        C, s = m.conv.in_channels, m.stride
        for j in range(s): filtw(f"l{i}.{j}", m.conv.weight[j * C:(j + 1) * C], None)
        prog.append(f"tgrow l{i} {C} {s} {up}"); up = 0
    elif isinstance(m, nn.ReLU): raise SystemExit(f"ReLU at {i} not after a filter")
    else: raise SystemExit(f"layer {i} ({type(m).__name__}) not supported")
    i += 1
if t.patch_size > 1: prog += [f"shuffle {t.patch_size}", "frames all"]
prog.append("out 3")
writer.add_string("ugpu.model", "tae"); writer.add_string("ugpu.index", "\n".join(index)); writer.add_string("ugpu.program", "\n".join(prog))
writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
print("\n".join(prog)); print("wrote", OUT, len(index), "tensors")
