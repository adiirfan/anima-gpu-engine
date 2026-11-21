"""The full Wan 2.2 VAE encoder for ONE picture (the start frame of a clip) -> one GGUF for the engine (model_vaeenc.h).
Walks ~/wan-build/wan_vae_enc.py's Enc2D (the causal 3D convolutions cut to their last time slice; checked 0.99998
against ComfyUI) and writes a layer list ("ugpu.program") over named slots:
  conv <w> <ci> <co> <k> <mode>   mode 0: padding k / 2; mode -2: stride 2, padded only right / bottom (also AvgDown's
                                  pixel-unshuffle + 1x1, written as a 2 x 2 stride-2 filter)
  convr <w> <ci> <co> <k> <slot>  same (mode 0), plus the slot (residual)
  norm <w> <c> <silu>             RMS norm over the channels of each pixel (the weight carries sqrt(C)), SiLU after
  attn <name> <c>                 x + proj(attention(qkv(norm(x)))), one head over all pixels
  save <slot> / load <slot> / add <slot>
  out <c>
Input (made by the caller): the picture * 2 - 1, pixel-unshuffled 2 x 2 -> (H/2, W/2, 12), channel c*4 + dy*2 + dx.
Output: the raw latent mu (H/16, W/16, 48), not yet (z - mean) / std.
Usage: pack_wanenc.py <out.gguf>   (reads ComfyUI's models/vae/wan2.2_vae.safetensors)"""
import sys
sys.path[:0] = ["/home/debian/wan-build", "/home/debian/ComfyUI", "/home/debian/llama.cpp/gguf-py"]
OUT = sys.argv[1]
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options; comfy.options.enable_args_parsing()
import numpy as np, torch
import gguf
import wan_vae_enc as E

torch.set_grad_enabled(False)
vae, enc = E.load()
writer = gguf.GGUFWriter(OUT, "ugpu")
index, prog = [], []


def put(name, kind, a, M, K):
    writer.add_tensor(name, np.ascontiguousarray(a)); index.append(f"{name} {kind} {M} {K}")


def filt(name, w, b):
    """[out][ky][kx][in] fp16, out padded to 4 (in is already a multiple of 4 everywhere here)"""
    w = w.detach().float().permute(0, 2, 3, 1).contiguous()
    assert w.shape[-1] % 4 == 0, (name, tuple(w.shape))
    co = (w.shape[0] + 3) // 4 * 4
    oc = w.shape[0]
    w = torch.cat([w, w.new_zeros(co - oc, *w.shape[1:])])
    put(name, "f16", w.half().reshape(co, -1).numpy(), co, w[0].numel())
    if b is not None:
        put(name + ".b", "f32", torch.cat([b.detach().float(), torch.zeros(co - oc)]).numpy(), co, 1)
    return co


def conv(name, c, mode=0, res=None):
    m = c.conv
    k = m.kernel_size[0]
    if mode == 0:
        assert m.stride == (1, 1) and m.padding == (k // 2, k // 2), (name, m.stride, m.padding)
    co = filt(name, m.weight, m.bias)
    prog.append(f"convr {name} {m.in_channels} {co} {k} {res}" if res else f"conv {name} {m.in_channels} {co} {k} {mode}")
    return co


def norm(name, n, silu):
    g = n.g.detach().float().reshape(-1)
    put(name, "f32", g.numpy(), g.numel(), 1)
    prog.append(f"norm {name} {g.numel()} {int(silu)}")


def res(name, r):
    prog.append("save x")
    sk = "x"
    if r.sc is not None:
        conv(name + ".sc", r.sc); prog.extend(["save s", "load x"]); sk = "s"
    norm(name + ".n1", r.n1, True); conv(name + ".c1", r.c1)
    norm(name + ".n2", r.n2, True); conv(name + ".c2", r.c2, res=sk)


def avg(name, a):
    w = a.conv.conv.weight.detach().float()          # (cout, cin * fs * fs, 1, 1) on the unshuffled channels
    fs = a.fs
    if fs == 1:
        co = filt(name, w, None); prog.append(f"conv {name} {w.shape[1]} {co} 1 0"); return
    cout, cin = w.shape[0], w.shape[1] // (fs * fs)
    w4 = w[:, :, 0, 0].view(cout, cin, fs, fs)       # unshuffled channel (c * fs + i) * fs + j -> tap (i, j) of c
    co = filt(name, w4, None)
    prog.append(f"conv {name} {cin} {co} {fs} -2")


conv("conv1", enc.conv1)
for bi, blk in enumerate(enc.down):
    p = f"d{bi}"
    prog.append("save i")
    avg(p + ".avg", blk.avg); prog += ["save a", "load i"]
    for li, l in enumerate(blk.body):
        if isinstance(l, E.Res): res(f"{p}.r{li}", l)
        else:
            co = filt(f"{p}.down{li}", l.conv.conv.weight, l.conv.conv.bias)
            prog.append(f"conv {p}.down{li} {l.conv.conv.in_channels} {co} 3 -2")
    prog.append("add a")
res("m0", enc.mid[0])
a = enc.mid[1]
C = a.qkv.conv.in_channels
g = a.norm.g.detach().float().reshape(-1); put("m1.n", "f32", g.numpy(), g.numel(), 1)   # the attn op norms itself
wq, bq = a.qkv.conv.weight, a.qkv.conv.bias   # q, k, v as three filters: one 3C-wide activation would be wider
for j, n in enumerate("qkv"):                  # than the chip's pictures allow (48 x 480 texels at 768)
    filt(f"m1.{n}", wq[j * C:(j + 1) * C], bq[j * C:(j + 1) * C])
filt("m1.proj", a.proj.conv.weight, a.proj.conv.bias)
prog.append(f"attn m1 {C}")
res("m2", enc.mid[2])
norm("hn", enc.hn, True); conv("hc", enc.hc); conv("out", enc.out)
prog.append("out 48")
writer.add_string("ugpu.model", "vaeenc"); writer.add_string("ugpu.index", "\n".join(index)); writer.add_string("ugpu.program", "\n".join(prog))
writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
print("\n".join(prog)); print("wrote", OUT, len(index), "tensors")
