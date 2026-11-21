"""Anima's prompt adapter (llm_adapter: 6 blocks + T5 word table) for the graphics-chip route, where it runs on the
processor after the Qwen3-0.6B reader (reader/texttaps.cpp, mode anima). Weights kept in bf16, names without the
"model.diffusion_model.llm_adapter." prefix.
Usage: pack_anima_text.py <anima safetensors> <out .gguf>"""
import sys
sys.path.insert(0, "/home/debian/llamacpp-hexagon/gguf-py")
import gguf
import numpy as np
from safetensors import safe_open

src, out = sys.argv[1:3]
P = "model.diffusion_model.llm_adapter."
w = gguf.GGUFWriter(out, "anima_adapter")
with safe_open(src, "pt") as f:
    names = sorted(k for k in f.keys() if k.startswith(P))
    for k in names:
        t = f.get_tensor(k)
        raw = t.view(__import__("torch").int16).numpy().view(np.uint16)
        w.add_tensor(k[len(P):], raw, raw_shape=list(t.shape), raw_dtype=gguf.GGMLQuantizationType.BF16)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(len(names), "tensors ->", out)
