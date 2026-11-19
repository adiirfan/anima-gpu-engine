// Fibo 2.0's prompt reader on the processor (llama.cpp): Qwen3-VL-4B reads the edit instruction with the picture to
// edit, and 15 of its inner states are handed to Fibo, as ComfyUI's Fibo2TEModel does (comfy/text_encoders/fibo2.py):
// the stream before layers 5, 7, 9, 11, 13, 16, 18, 20, 22, 24, 27, 29, 31, 33, 35 (= llama.cpp's "l_out-<n - 1>"),
// each at unit RMS (no weight, eps 1e-6) and rounded to bf16, laid out as six bundles of three.
//
//   fibotaps <reader.gguf> <mmproj.gguf> <picture.rgb> <width> <height> <instruction file> <out dir> [threads]
//
// picture.rgb: raw 8-bit RGB rows (the app's 768 x 768 picture). Writes <out>/ctx.f32 (T x 46080) and <out>/info.txt
// "T G h w" for the engine (h, w = the drawing's latent rows: picture / 16). Exit code 0 on success.
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static const int BUNDLES[6][3] = {{9, 20, 31}, {5, 16, 27}, {7, 18, 29}, {9, 20, 31}, {11, 22, 33}, {13, 24, 35}};
static const int TAPS[15] = {5, 7, 9, 11, 13, 16, 18, 20, 22, 24, 27, 29, 31, 33, 35};
static const int MAXG = 512;

struct Taps {
    int n_embd = 0;
    std::map<int, std::vector<float>> rows;   // tap layer -> rows x n_embd, in the order the reader ran them
};

// asked for every node; keeps the outputs of the layers just before a tap
static bool onTensor(struct ggml_tensor *t, bool ask, void *ud) {
    auto *tp = (Taps *)ud;
    int il = -1;
    if (sscanf(t->name, "l_out-%d", &il) != 1) return ask ? false : true;
    const int tap = il + 1;
    bool want = false;
    for (int x : TAPS) want |= x == tap;
    if (ask) return want;
    if (!want) return true;
    if (t->type != GGML_TYPE_F32) { fprintf(stderr, "l_out-%d is not f32\n", il); return false; }
    const int ne0 = (int)t->ne[0], ne1 = (int)t->ne[1];
    tp->n_embd = ne0;
    auto &v = tp->rows[tap];
    const size_t at = v.size();
    v.resize(at + (size_t)ne0 * ne1);
    ggml_backend_tensor_get(t, v.data() + at, 0, (size_t)ne0 * ne1 * sizeof(float));
    return true;
}

static float bf16(float x) {   // round to nearest even, as torch's .to(bfloat16)
    uint32_t u; memcpy(&u, &x, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    u &= 0xFFFF0000u;
    float y; memcpy(&y, &u, 4); return y;
}

static std::string jsonString(const std::string &s) {   // as Python's json.dumps(..., ensure_ascii=False)
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            case '\b': o += "\\b"; break;
            case '\f': o += "\\f"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += (char)c;
        }
    }
    return o + "\"";
}

int main(int argc, char **argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: %s <reader.gguf> <mmproj.gguf> <picture.rgb> <width> <height> <instruction file> <out dir> [threads]\n", argv[0]);
        return 1;
    }
    const char *modelPath = argv[1], *mmprojPath = argv[2], *picPath = argv[3], *instrPath = argv[6];
    const int W = atoi(argv[4]), H = atoi(argv[5]);
    const std::string out = argv[7];
    const int threads = argc > 8 ? atoi(argv[8]) : 4;
    auto t0 = std::chrono::steady_clock::now();
    auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

    std::vector<unsigned char> rgb((size_t)W * H * 3);
    { FILE *f = fopen(picPath, "rb");
      if (!f || fread(rgb.data(), 1, rgb.size(), f) != rgb.size()) { fprintf(stderr, "cannot read %s (%dx%d RGB)\n", picPath, W, H); return 1; }
      fclose(f); }
    std::string instr;
    { FILE *f = fopen(instrPath, "rb");
      if (!f) { fprintf(stderr, "cannot read %s\n", instrPath); return 1; }
      char b[4096]; size_t n; while ((n = fread(b, 1, sizeof b, f)) > 0) instr.append(b, n); fclose(f); }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;   // the processor: this runs on phones without the AI chip
    llama_model *model = llama_model_load_from_file(modelPath, mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", modelPath); return 1; }

    Taps taps;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 1024; cp.n_ubatch = 1024; cp.n_threads = cp.n_threads_batch = threads;
    cp.cb_eval = onTensor; cp.cb_eval_user_data = &taps;
    llama_context *lctx = llama_init_from_model(model, cp);
    if (!lctx) { fprintf(stderr, "cannot make the reader's context\n"); return 1; }

    mtmd_context_params mcp = mtmd_context_params_default();
    mcp.use_gpu = false; mcp.n_threads = threads; mcp.print_timings = false; mcp.warmup = false;
    mtmd_context *mctx = mtmd_init_from_file(mmprojPath, model, mcp);
    if (!mctx) { fprintf(stderr, "cannot load %s\n", mmprojPath); return 1; }
    printf("loaded in %.1f s\n", secs()); fflush(stdout);

    // {"image":"<picture>","edit_instruction":"..."} in the chat turn, as Fibo2Tokenizer writes it; mtmd puts
    // <|vision_start|> ... <|vision_end|> around the picture's rows itself
    const std::string prompt = std::string("<|im_start|>user\n{\"image\":\"") + mtmd_default_marker() +
                               "\",\"edit_instruction\":" + jsonString(instr) + "}<|im_end|>\n<|im_start|>assistant\n";
    mtmd_bitmap *bmp = mtmd_bitmap_init(W, H, rgb.data());
    mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    const mtmd_bitmap *bl[1] = {bmp};
    if (int r = mtmd_tokenize(mctx, chunks, &txt, bl, 1)) { fprintf(stderr, "tokenize failed (%d)\n", r); return 1; }
    const size_t T = mtmd_helper_get_n_tokens(chunks);
    printf("reading %zu rows\n", T); fflush(stdout);
    llama_pos np = 0;
    if (mtmd_helper_eval_chunks(mctx, lctx, chunks, 0, 0, cp.n_batch, false, &np)) { fprintf(stderr, "reading failed\n"); return 1; }
    printf("read in %.1f s\n", secs()); fflush(stdout);

    const int E = taps.n_embd;
    for (int tap : TAPS) {
        const size_t got = taps.rows[tap].size() / (E ? E : 1);
        if (got != T) { fprintf(stderr, "layer %d gave %zu rows, not %zu\n", tap, got, T); return 1; }
    }
    // unit RMS + bf16 per tap, then the six bundles side by side per row
    for (auto &kv : taps.rows) {
        auto &v = kv.second;
        for (size_t r = 0; r < T; r++) {
            float *x = v.data() + r * E;
            double ss = 0; for (int k = 0; k < E; k++) ss += (double)x[k] * x[k];
            const float s = (float)(1.0 / std::sqrt(ss / E + 1e-6));
            for (int k = 0; k < E; k++) x[k] = bf16(x[k] * s);
        }
    }
    const size_t RW = (size_t)18 * E;
    std::vector<float> ctx(T * RW);
    for (size_t r = 0; r < T; r++)
        for (int b = 0; b < 6; b++) for (int j = 0; j < 3; j++)
            memcpy(&ctx[r * RW + (size_t)(b * 3 + j) * E], &taps.rows[BUNDLES[b][j]][r * E], E * sizeof(float));
    FILE *f = fopen((out + "/ctx.f32").c_str(), "wb");
    if (!f || fwrite(ctx.data(), sizeof(float), ctx.size(), f) != ctx.size()) { fprintf(stderr, "cannot write %s/ctx.f32\n", out.c_str()); return 1; }
    fclose(f);
    // Fibo's gist size grows with the prompt (fibo_model.gist_count)
    const double rr = std::min(1.0, std::max(0.0, ((double)T - 256) / (2048 - 256)));
    const int G = std::max(256, std::min(MAXG, (int)std::ceil((256 + rr * (MAXG - 256)) / 64) * 64));
    f = fopen((out + "/info.txt").c_str(), "w");
    fprintf(f, "%zu %d %d %d\n", T, G, H / 16, W / 16); fclose(f);
    printf("done: %zu rows x %zu, gist %d, %.1f s\n", T, RW, G, secs());

    mtmd_bitmap_free(bmp); mtmd_input_chunks_free(chunks); mtmd_free(mctx);
    llama_free(lctx); llama_model_free(model); llama_backend_free();
    return 0;
}
