"""Anima's 8-bit set as one engine GGUF (what the app downloads): the 8-bit layers (and their scales) from
<q8 prefix>.idx/.bin, every other tensor from the fp16 set <fp16 prefix>.idx/.bin; the fp16 copies of 8-bit layers are
left out (the engine never loads them when the 8-bit one is there).
Usage: pack_anima_gguf.py <fp16 prefix> <q8 prefix | none> <out .gguf>   ("none": the plain fp16 set, for chips
without a fast 8-bit dot product; stream it with STREAM=1 STREAM_PK=1)"""
import sys
sys.path[:0] = ["/home/debian/llama.cpp/gguf-py"]
import numpy as np
import gguf

fp, q8, out = sys.argv[1:4]
def read(prefix):
    raw = np.memmap(prefix + ".bin", np.uint8, "r")
    rows = []
    for line in open(prefix + ".idx"):
        if line.startswith("#") or not line.strip(): continue
        n, k, M, K, off, nb = line.split()
        rows.append((n, k, int(M), int(K), raw[int(off):int(off) + int(nb)]))
    return rows
DT = {"q8": np.int8, "qs": np.float32, "pk": np.float16, "h": np.float16, "f": np.float32, "f16": np.float16, "f32": np.float32}
w = gguf.GGUFWriter(out, "ugpu")
index, have = [], set()
for n, k, M, K, b in (read(q8) if q8 != "none" else []):
    w.add_tensor(n, np.frombuffer(b, DT[k])); index.append(f"{n} {k} {M} {K}"); have.add(n)
for n, k, M, K, b in read(fp):
    if n in have: continue
    w.add_tensor(n, np.frombuffer(b, DT[k])); index.append(f"{n} {k} {M} {K}")
w.add_string("ugpu.model", "anima")
w.add_string("ugpu.index", "\n".join(index))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(len(index), "tensors ->", out)
