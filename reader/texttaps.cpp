// Prompt readers for the graphics-chip route, on the processor (llama.cpp): the app hands over the prompt's token ids
// (already wrapped and tokenized, as for the AI-chip route) and gets the reader's inner states back as float32 rows.
//
//   texttaps <reader.gguf> <ids file: int32> <out dir> <threads> <mode>
//
// mode "klein": FLUX.2 klein's reader (Qwen3-4B), as ComfyUI's Qwen3_4BModel with layer=[9, 18, 27]: the stream
//   just before layers 9, 18 and 27 (= llama.cpp's "l_out-8/17/26"), no norm, side by side per row
//   -> <out>/ctx.f32 (T x 3*n_embd)
// mode "last": the last layer after the final norm (llama.cpp's "result_norm"), as ComfyUI's layer="last"
//   (Anima's Qwen3-0.6B) -> <out>/hidden.f32 (T x n_embd)
// mode "qimg": Qwen-Image 2.1's reader (Qwen3-VL-8B's language half): the last layer's output WITHOUT the final
//   norm (llama.cpp's "l_out-<last>"), as the AI-chip route's reader -> <out>/hidden.f32 (T x n_embd); the caller
//   drops the system turn's rows
// mode "anima" <adapter.gguf> <T5 ids: int32> <T5 weights: float32>: "last", then Anima's prompt adapter
//   (anima_adapter.h) -> <out>/ctx.f32 (256 x 1024, rows past the T5 prompt zero)
// Exit code 0 on success.
#include "llama.h"
#include "anima_adapter.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct Taps {
    std::vector<std::string> names;                // tensors to keep, in output order
    int n_embd = 0;
    std::map<std::string, std::vector<float>> rows;
};

static bool onTensor(struct ggml_tensor *t, bool ask, void *ud) {
    auto *tp = (Taps *)ud;
    bool want = false;
    for (auto &n : tp->names) want |= n == t->name;
    if (ask) return want;
    if (!want) return true;
    if (t->type != GGML_TYPE_F32) { fprintf(stderr, "%s is not f32\n", t->name); return false; }
    const int ne0 = (int)t->ne[0], ne1 = (int)t->ne[1];
    tp->n_embd = ne0;
    auto &v = tp->rows[t->name];
    const size_t at = v.size();
    v.resize(at + (size_t)ne0 * ne1);
    ggml_backend_tensor_get(t, v.data() + at, 0, (size_t)ne0 * ne1 * sizeof(float));
    return true;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <reader.gguf> <ids file> <out dir> <threads> <klein|last|anima> [adapter.gguf t5 ids t5 weights]\n", argv[0]);
        return 1;
    }
    const std::string out = argv[3], mode = argv[5];
    const int threads = atoi(argv[4]) > 0 ? atoi(argv[4]) : 4;
    auto t0 = std::chrono::steady_clock::now();
    auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

    std::vector<llama_token> ids;
    { FILE *f = fopen(argv[2], "rb");
      if (!f) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
      int32_t v; while (fread(&v, 4, 1, f) == 1) ids.push_back(v); fclose(f); }
    const size_t T = ids.size();
    if (T == 0 || T > 2048) { fprintf(stderr, "%zu prompt ids (1 to 2048 allowed)\n", T); return 1; }

    Taps taps;
    if (mode == "klein") taps.names = {"l_out-8", "l_out-17", "l_out-26"};
    else if (mode == "qimg") taps.names = {};   // set once the model's layer count is known
    else if (mode == "last" || mode == "anima") taps.names = {"result_norm"};   // filled from the embeddings below
    else { fprintf(stderr, "unknown mode %s\n", mode.c_str()); return 1; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;   // the processor: this runs on phones without the AI chip
    // the processor only: the app's llama.cpp also carries the AI-chip backend, which would otherwise open a session
    // on the AI chip (and fail when the app holds it)
    static ggml_backend_dev_t cpuOnly[1] = {nullptr};   // no offload devices at all, as llama.cpp's "--device none"
    mp.devices = cpuOnly;
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    if (mode == "qimg") taps.names = {"l_out-" + std::to_string(llama_model_n_layer(model) - 1)};
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (auto id : ids) if (id < 0 || id >= nv) { fprintf(stderr, "id %d outside the reader's %d words\n", id, nv); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.op_offload = false;
    const uint32_t nc = (uint32_t)((T + 63) / 64 * 64);   // only as much room as the prompt needs (memory)
    cp.n_ctx = nc; cp.n_batch = nc; cp.n_ubatch = nc; cp.n_threads = cp.n_threads_batch = threads;
    const bool tapped = mode == "klein" || mode == "qimg";
    cp.embeddings = !tapped;   // keeps result_norm for every row (no word guessing at the end)
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    if (tapped) { cp.cb_eval = onTensor; cp.cb_eval_user_data = &taps; }
    llama_context *lctx = llama_init_from_model(model, cp);
    if (!lctx) { fprintf(stderr, "cannot make the reader's context\n"); return 1; }
    printf("loaded in %.1f s\n", secs()); fflush(stdout);

    llama_batch b = llama_batch_init((int)T, 0, 1);
    for (size_t i = 0; i < T; i++) {
        b.token[i] = ids[i]; b.pos[i] = (llama_pos)i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = true;
    }
    b.n_tokens = (int)T;
    if (llama_decode(lctx, b)) { fprintf(stderr, "reading failed\n"); return 1; }
    printf("read %zu rows in %.1f s\n", T, secs()); fflush(stdout);

    if (!tapped) {   // the final norm's rows come back through llama.cpp's embeddings (pooling none)
        const int ne = llama_model_n_embd(model);
        auto &v = taps.rows["result_norm"]; v.resize(T * (size_t)ne);
        for (size_t i = 0; i < T; i++) {
            const float *e = llama_get_embeddings_ith(lctx, (int32_t)i);
            if (!e) { fprintf(stderr, "no output for row %zu\n", i); return 1; }
            memcpy(&v[i * ne], e, ne * sizeof(float));
        }
        taps.n_embd = ne;
    }
    const int E = taps.n_embd;
    for (auto &n : taps.names) {
        const size_t got = taps.rows[n].size() / (E ? E : 1);
        if (got != T) { fprintf(stderr, "%s gave %zu rows, not %zu\n", n.c_str(), got, T); return 1; }
    }
    const size_t K = taps.names.size(), RW = K * E;
    std::vector<float> o(T * RW);
    for (size_t r = 0; r < T; r++)
        for (size_t k = 0; k < K; k++)
            memcpy(&o[r * RW + k * E], &taps.rows[taps.names[k]][r * E], E * sizeof(float));
    if (mode == "anima") {
        if (argc < 9) { fprintf(stderr, "anima needs <adapter.gguf> <T5 ids> <T5 weights>\n"); return 1; }
        std::vector<int32_t> t5; std::vector<float> t5w;
        { FILE *g = fopen(argv[7], "rb"); int32_t v; while (g && fread(&v, 4, 1, g) == 1) t5.push_back(v); if (g) fclose(g); }
        { FILE *g = fopen(argv[8], "rb"); float v; while (g && fread(&v, 4, 1, g) == 1) t5w.push_back(v); if (g) fclose(g); }
        if (t5.empty() || t5.size() >= (size_t)anima_adapter::NT || t5w.size() != t5.size()) {
            fprintf(stderr, "%zu T5 ids, %zu weights (1 to %d, same count)\n", t5.size(), t5w.size(), anima_adapter::NT - 1); return 1;
        }
        for (auto id : t5) if (id < 0 || id >= 32128) { fprintf(stderr, "bad T5 id %d\n", id); return 1; }
        anima_adapter::Weights w; std::vector<uint16_t> embed;
        if (!anima_adapter::load(argv[6], &w, &embed) || embed.size() != (size_t)32128 * anima_adapter::H) return 1;
        llama_batch_free(b); llama_free(lctx); llama_model_free(model);   // the reader's memory, before the adapter's
        auto ctx = anima_adapter::run(threads, w, embed, o, (int)T, t5, t5w);
        FILE *f = fopen((out + "/ctx.f32").c_str(), "wb");
        if (!f || fwrite(ctx.data(), sizeof(float), ctx.size(), f) != ctx.size()) { fprintf(stderr, "cannot write ctx.f32\n"); return 1; }
        fclose(f);
        printf("done: %zu reader rows, %zu prompt rows, %.1f s\n", T, t5.size(), secs());
        llama_backend_free();
        return 0;
    }
    const std::string path = out + (mode == "klein" ? "/ctx.f32" : "/hidden.f32");
    FILE *f = fopen(path.c_str(), "wb");
    if (!f || fwrite(o.data(), sizeof(float), o.size(), f) != o.size()) { fprintf(stderr, "cannot write %s\n", path.c_str()); return 1; }
    fclose(f);
    printf("done: %zu rows x %zu, %.1f s\n", T, RW, secs());

    llama_batch_free(b); llama_free(lctx); llama_model_free(model); llama_backend_free();
    return 0;
}
