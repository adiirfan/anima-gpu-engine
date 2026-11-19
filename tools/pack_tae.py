"""Tiny picture decoders (the TAESD family: taeqi2_1 for Qwen, taef2 for Klein, ...) -> one GGUF for the engine.
The decoder is rebuilt with the reference code and walked layer by layer; the walk is written as a small program
("ugpu.program", one op per line) that model_tae.h runs:
  clamp                                  x = 3 tanh(x / 3)
  conv <name> <ci> <co> <k> <bias> <relu> <up>   k x k filter, padding k/2; up = 1: input doubled first (nearest),
                                         up = -1: stride 2 (an encoder's shrinking filter)
  pool <name> <ci> <cg> <groups>         x += p3(relu(groupnorm(p0(x)))), p0: ci -> cg 1x1, p3 back (flux_2)
  block <name> <ci> <co> <skip>          relu(c2(relu(c1(relu(c0(x))))) + skip(x)), skip = 1x1 filter or x itself
  shuffle <r>                            channels -> r x r pixels (PixelShuffle)
  out <channels>                         channels kept in the picture
Filters are stored fp16 as [out][ky][kx][in]; biases fp32.
ENC=1 packs the encoder instead (input: the picture in [0, 1], channels padded to 4; output: the latent).
Usage: python pack_tae.py <taesd file> <out.gguf> <latent channels> <arch variant or ""> <image channels>"""
import os
import sys
sys.path[:0] = ["/home/debian/phone-v81/qwen/build/common", "/home/debian/llama.cpp/gguf-py"]
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from taesd import TAESD, Clamp, Block
import gguf

SRC, OUT, LC, ARCH, IC = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] or None, int(sys.argv[5])
t = TAESD(encoder_path=None, decoder_path=None, latent_channels=LC, arch_variant=ARCH, image_channels=IC)
sd = load_file(SRC)
ENC = os.environ.get("ENC") == "1"
if any(k.startswith("decoder.layers.") or k.startswith("encoder.layers.") for k in sd):     # the Sequential's index 0 (Clamp) has no weights in these files
    sd = {".".join([e, str(int(i) + (1 if e == "decoder" else 0)), *s]): v
          for k, v in sd.items() for e, _l, i, *s in [k.split(".")]}
half = "encoder." if ENC else "decoder."
sd = {k: v for k, v in sd.items() if k.startswith(half)}
missing = [k for k in t.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False).missing_keys if k.startswith(half)]
if missing: raise SystemExit(f"weights missing from {SRC}: {missing[:5]}")
writer = gguf.GGUFWriter(OUT, "ugpu")
index, prog = [], []

def put(name, kind, a, M, K):
    writer.add_tensor(name, np.ascontiguousarray(a))
    index.append(f"{name} {kind} {M} {K}")
def filt(name, c):
    w = c.weight.detach().permute(0, 2, 3, 1).contiguous()          # [out][ky][kx][in]
    if w.shape[-1] % 4:                                              # the picture's 3 colours -> 4 (a zero channel)
        w = torch.cat([w, w.new_zeros(*w.shape[:-1], 4 - w.shape[-1] % 4)], -1)
    co = (c.out_channels + 3) // 4 * 4                               # the engine works in groups of 4 channels:
    w = torch.cat([w, w.new_zeros(co - w.shape[0], *w.shape[1:])])   # pad with zero outputs (e.g. RGB -> RGB0)
    put(name, "f16", w.half().reshape(w.shape[0], -1).numpy(), w.shape[0], w[0].numel())
    if c.bias is not None:
        b = torch.cat([c.bias.detach().float(), torch.zeros(co - c.out_channels)])
        put(name + ".b", "f32", b.numpy(), co, 1)
    return (c.in_channels + 3) // 4 * 4, co, c.kernel_size[0], int(c.bias is not None)

layers, up, i = list(t.encoder if ENC else t.decoder), 0, 0
while i < len(layers):
    m = layers[i]
    if isinstance(m, Clamp): prog.append("clamp")
    elif isinstance(m, nn.Upsample): up = 1
    elif isinstance(m, nn.Conv2d):
        assert m.stride in ((1, 1), (2, 2)) and m.padding[0] == m.kernel_size[0] // 2
        if m.stride == (2, 2): up = -1
        relu = int(i + 1 < len(layers) and isinstance(layers[i + 1], nn.ReLU))
        ci, co, k, b = filt(f"l{i}", m)
        prog.append(f"conv l{i} {ci} {co} {k} {b} {relu} {up}")
        up = 0
        i += relu
    elif isinstance(m, Block):
        if m.pool is not None:     # flux_2: x += p3(relu(groupnorm(p0(x)))) before the block
            p0, gn, p3 = m.pool[0], m.pool[1], m.pool[3]
            filt(f"l{i}.p0", p0); filt(f"l{i}.p3", p3)
            put(f"l{i}.gn.w", "f32", gn.weight.detach().float().numpy(), gn.num_channels, 1)
            put(f"l{i}.gn.b", "f32", gn.bias.detach().float().numpy(), gn.num_channels, 1)
            prog.append(f"pool l{i} {p0.in_channels} {p0.out_channels} {gn.num_groups}")
        c = [x for x in m.conv if isinstance(x, nn.Conv2d)]
        for j, cj in enumerate(c): filt(f"l{i}.c{j}", cj)
        skip = int(isinstance(m.skip, nn.Conv2d))
        if skip: filt(f"l{i}.skip", m.skip)
        prog.append(f"block l{i} {c[0].in_channels} {c[0].out_channels} {skip}")
    elif isinstance(m, nn.PixelShuffle): prog.append(f"shuffle {m.upscale_factor}")
    elif isinstance(m, nn.ReLU): raise SystemExit(f"ReLU at {i} not after a filter")
    else: raise SystemExit(f"layer {i} ({type(m).__name__}) not supported")
    i += 1
prog.append(f"out {LC if ENC else min(IC, 3)}")
writer.add_string("ugpu.model", "tae")
writer.add_string("ugpu.index", "\n".join(index))
writer.add_string("ugpu.program", "\n".join(prog))
writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
print("\n".join(prog))
print("wrote", OUT, len(index), "tensors")
