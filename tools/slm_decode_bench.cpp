// SPDX-License-Identifier: Apache-2.0
// Per-token decode latency bench for a real SLM (llama.cpp backend).
//
// The GEMV decode proxy reproduces the bandwidth pattern of LLM decode; this
// tool measures the real thing (Qwen2.5-1.5B Q4_K_M) so the paper's EMC x SLM
// claims don't rest on a synthetic alone. One iteration = sample(greedy) +
// llama_decode(1 token); per-iteration wall time is recorded into a
// preallocated array (no I/O in the timed loop — see the pilot_trans stdio
// flush lesson) and written as CSV afterwards, with a JSON percentile summary
// matching the harness keys (p50/p90/p99/p99.9/p99.99).
//
// RT setup mirrors harness/rt_utils: pin + SCHED_FIFO + mlockall, warn and
// continue if denied (needs root). Greedy sampling can hit EOS mid-run; EOS
// is replaced with a fixed filler token so decode mechanics (KV growth,
// weight streaming) stay uniform across the run. KV grows from prompt_len to
// prompt_len+warmup+tokens over the run — recorded in the JSON meta since
// per-token cost rises slowly with context.
//
// Build (after llama.cpp is built with CUDA):
//   g++ -O2 -o tools/slm_decode_bench tools/slm_decode_bench.cpp \
//       -I$LLAMA/include -I$LLAMA/ggml/include \
//       -L$LLAMA/build/bin -lllama -Wl,-rpath,$LLAMA/build/bin
// Run:
//   sudo ./tools/slm_decode_bench --model ~/mobile/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
//       --tokens 1000 --warmup 100 --cpu 5 --prio 80 --out results/slm_smoke

#define _GNU_SOURCE 1
#include <sched.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>

#include <sys/mman.h>
#include <unistd.h>

#include "llama.h"

static long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static double pct(std::vector<long long> &sorted, double q) {
    size_t i = (size_t)(q / 100.0 * (sorted.size() - 1));
    return sorted[i] / 1000.0;   // us
}

int main(int argc, char **argv) {
    const char *model_path = nullptr, *out_prefix = nullptr;
    int n_tokens = 1000, n_warmup = 100, prompt_len = 64;
    int cpu = -1, prio = 0, ngl = 99;

    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--model")) model_path = argv[++i];
        else if (!strcmp(argv[i], "--tokens")) n_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")) n_warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prompt-tokens")) prompt_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpu")) cpu = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prio")) prio = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ngl")) ngl = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out")) out_prefix = argv[++i];
    }
    if (!model_path || !out_prefix) {
        fprintf(stderr, "usage: %s --model GGUF --out PREFIX [--tokens N] "
                "[--warmup W] [--prompt-tokens P] [--cpu C] [--prio P] [--ngl N]\n",
                argv[0]);
        return 1;
    }

    if (cpu >= 0) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0)
            fprintf(stderr, "slm_bench: warning: pin to cpu%d failed\n", cpu);
    }
    if (prio > 0) {
        struct sched_param sp = {}; sp.sched_priority = prio;
        if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0)
            fprintf(stderr, "slm_bench: warning: SCHED_FIFO denied (need root)\n");
    }
    // mlockall must come AFTER CUDA/model init: MCL_FUTURE makes CUDA's large
    // virtual mappings fail with "out of memory" on Tegra (the harness orders
    // it the same way: backend first, then rt setup).
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "slm_bench: model load failed\n"); return 1; }
    const llama_vocab *vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = prompt_len + n_warmup + n_tokens + 16;
    cparams.n_batch = 512;
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "slm_bench: context init failed\n"); return 1; }

    if (mlockall(MCL_CURRENT) != 0)
        fprintf(stderr, "slm_bench: warning: mlockall failed\n");

    // Build a prompt with at least prompt_len tokens, truncate exactly.
    std::string text;
    while ((int)text.size() < prompt_len * 8)
        text += "The quick brown fox jumps over the lazy dog. ";
    std::vector<llama_token> toks(prompt_len * 2 + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int)text.size(),
                           toks.data(), (int)toks.size(), true, false);
    if (n < prompt_len) {
        fprintf(stderr, "slm_bench: tokenizer produced %d < %d tokens\n",
                n, prompt_len);
        return 1;
    }
    toks.resize(prompt_len);

    // Filler for EOS so decode mechanics stay uniform.
    llama_token filler[8];
    int nf = llama_tokenize(vocab, " the", 4, filler, 8, false, false);
    llama_token filler_tok = nf > 0 ? filler[0] : toks[0];

    long long t_pf0 = now_ns();
    llama_batch batch = llama_batch_get_one(toks.data(), (int)toks.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "slm_bench: prefill decode failed\n");
        return 1;
    }
    double prefill_ms = (now_ns() - t_pf0) / 1e6;

    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    std::vector<long long> dur(n_tokens, 0);
    int total = n_warmup + n_tokens;
    for (int i = 0; i < total; i++) {
        long long t0 = now_ns();
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) tok = filler_tok;
        llama_batch b = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "slm_bench: decode failed at step %d\n", i);
            return 1;
        }
        long long t1 = now_ns();
        if (i >= n_warmup) dur[i - n_warmup] = t1 - t0;
    }

    char path[1024];
    snprintf(path, sizeof(path), "%s.csv", out_prefix);
    FILE *csv = fopen(path, "w");
    if (!csv) { perror("slm_bench: fopen csv"); return 1; }
    fprintf(csv, "token,dur_ns\n");
    for (int i = 0; i < n_tokens; i++)
        fprintf(csv, "%d,%lld\n", i, dur[i]);
    fclose(csv);

    std::vector<long long> sorted = dur;
    std::sort(sorted.begin(), sorted.end());
    double mean = 0;
    for (long long d : dur) mean += d;
    mean /= dur.size() * 1000.0;

    snprintf(path, sizeof(path), "%s.json", out_prefix);
    FILE *js = fopen(path, "w");
    if (!js) { perror("slm_bench: fopen json"); return 1; }
    fprintf(js,
        "{\n"
        "  \"meta\": {\"model\": \"%s\", \"ngl\": %d, \"prompt_tokens\": %d,\n"
        "    \"warmup\": %d, \"tokens\": %d, \"prefill_ms\": %.2f,\n"
        "    \"kv_range\": [%d, %d], \"cpu\": %d, \"prio\": %d},\n"
        "  \"compute_us\": {\"p50\": %.3f, \"p90\": %.3f, \"p99\": %.3f,\n"
        "    \"p99.9\": %.3f, \"p99.99\": %.3f, \"mean\": %.3f,\n"
        "    \"min\": %.3f, \"max\": %.3f}\n"
        "}\n",
        model_path, ngl, prompt_len, n_warmup, n_tokens, prefill_ms,
        prompt_len + n_warmup, prompt_len + n_warmup + n_tokens, cpu, prio,
        pct(sorted, 50), pct(sorted, 90), pct(sorted, 99),
        pct(sorted, 99.9), pct(sorted, 99.99), mean,
        sorted.front() / 1000.0, sorted.back() / 1000.0);
    fclose(js);

    printf("[slm] p50=%.1f p99=%.1f max=%.1f us/token, prefill=%.1f ms\n",
           pct(sorted, 50), pct(sorted, 99), sorted.back() / 1000.0, prefill_ms);

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
