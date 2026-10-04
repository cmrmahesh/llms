# Inference, Opened

A single-page, interactive walkthrough of LLM inference. A real open-weight model
(SmolLM2-135M-Instruct by default) runs **inside the browser tab**. Every number
on the page comes from it: token IDs, embedding rows, RoPE rotations, attention
weights, feed-forward activations, logits, KV cache contents and timings.

```
python3 serve.py          # then open http://localhost:8000
```

The page needs to be served over HTTP (ES modules and a Web Worker don't load
from `file://`). `serve.py` is a plain static server with caching turned off.
Any static server works. The first load downloads the weights (269 MB) from
Hugging Face. After that they come from the browser's Cache Storage.

## What the page covers, in order

It's written for someone new to the topic. Chapter 0 covers vectors, matrices,
tensors, matrix multiplication, weights, forward/backward passes and the hardware. Every chapter opens
with an "in plain words" explanation and keeps the technical version in a
"Go deeper" section. Underlined terms show a definition on hover, focus or tap
(`js/glossary.js`).

| # | Chapter | Where it runs on a real server |
|---|---------|--------------------------------|
| 0 | The ideas you need first: vectors, tensors, matrix × vector, weights, forward and backward passes (a 9-weight model you can train), hardware | — |
| 1 | What's in the download: config, tokenizer, safetensors header, model anatomy, quantization | disk → RAM → HBM |
| 2 | Tokenizer: vocabulary explorer, pre-split, BPE glue steps for any word, ID lookup | CPU |
| 3 | Embedding lookup and nearest neighbours | HBM gather |
| 4 | Word order via RoPE, shown as clock hands using the model's real vectors | SM registers |
| 5 | The transformer loop (attention → add → feed-forward → add, × 30, with Back/Next/Autoplay) and "follow the shape": one token's 576 × 1 vector through every matrix, drawn to scale, with a running ledger of all 134,515,008 weights | HBM → SM |
| 6 | Attention: Q/K/V, heads as slices, real attention maps, attention sinks and previous-word heads | HBM → SM |
| 7 | Feed-forward (SwiGLU), the residual stream, logit lens | HBM → SM |
| 8 | Logits → softmax → temperature / top-k / top-p → token → bytes → text | SM, then 4 bytes to CPU |
| 9 | Your whole prompt through all 30 layers as a T × 576 matrix: the matrix after every layer, every step of a layer on all rows (Q/K/V, each head's T × T attention table, combine + add, normalize, feed-forward), every row's guess for its next token, final softmax | HBM → SM |
| 10 | Decode loop and the KV cache (live grid, one layer's real numbers, with/without cache timings) | HBM |
| 11 | Why decode is memory-bound: arithmetic intensity, H100 roofline, CPU benchmark | HBM vs SM |
| 12 | Prefix caching: hash-chained 16-token blocks, measured time-to-first-token | HBM |
| 13 | One request, every byte: phase-by-phase table for your last run | all |

A sticky **machine map** (CPU, RAM, PCIe, HBM, L2, SMs) follows the chapters and
shows where each step happens and what crosses each link.

## How it works

```
main thread  ("CPU")                       Web Worker  ("GPU")
─────────────────────                      ──────────────────────────────────
tokenizer.js  BPE with a merge trace       model.js     Llama forward pass + KV cache,
util.js       sampling (T, top-k, top-p)                 records a trace of every step
chapters/*    one module per chapter       runtime.js   WebAssembly memory = "HBM"
viz/          machine map, three.js hero   kernels.wat  hand-written SIMD matmul kernel
        ── token IDs ──▶   ◀── logits + trace ──
```

- `js/engine/model.js` implements the Llama architecture: RMSNorm, Q/K/V
  projections, RoPE, grouped-query attention with a KV cache, SwiGLU feed-forward,
  and a tied LM head. Matrix multiplies go through `runtime.matmul`. Everything
  else is plain JavaScript, written to be read.
- `js/engine/kernels.wat` is the matmul kernel in WebAssembly text format. It
  processes 4 tokens per weight load, which makes prefill measurably faster per
  token than decode (chapter 9 benchmarks this). Rebuild it after editing:
  `npx -p wabt wat2wasm js/engine/kernels.wat -o js/engine/kernels.wasm`.
- `js/engine/tokenizer.js` reads a Hugging Face `tokenizer.json` (byte-level BPE).

### Correctness

Checked against the reference implementations:

- Tokenizer: identical IDs to `@huggingface/transformers` on mixed text,
  including digits, emoji, CJK, code and chat special tokens.
- Model: final-position logits match the official ONNX export run in
  onnxruntime to within 1e-4 (logits range ±31), with the same argmax.

### Speed (in this tab, on a 6-core i5-8500T)

About 20 tokens/s decode and about 0.5 s prefill for a 37-token prompt. The
weights are held as float32 in WebAssembly memory (551 MB), even though the file
stores bfloat16.

## Adding a model

Append an entry to `js/models.js`. Any Hugging Face repo with a
`LlamaForCausalLM` config, a BPE `tokenizer.json` and a single
`model.safetensors` in BF16, F16 or F32 should work. Hidden size must be a
multiple of 8. A model's float32 weights must fit in browser memory, so in
practice that means up to roughly 500M parameters.

## Sources

- Karthika Raghavan, [What Is LLM Inference, Really?](https://kraghavan.ca/llm-infrastructure/inference/2026/04/14/re-introduction-to-inference.html) (`kragavan.md` in this repo)
- [Inference Engineering](https://inferenceengineering.tech/chapters/): hardware numbers (H100: 3.35 TB/s HBM, 50 MB L2, 256 KB L1/SMEM per SM), prefill vs decode, arithmetic intensity
- [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) by Hugging Face TB
