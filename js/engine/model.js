// A Llama-architecture decoder (Llama 2/3, SmolLM2, TinyLlama, ...) written as
// plainly as possible, so every step can be traced and shown on the page.
//
// Per layer:  x += Wo · Attention(RoPE(Wq·norm(x)), RoPE(Wk·norm(x)), Wv·norm(x))
//             x += Wdown · (silu(Wgate·norm(x)) ⊙ Wup·norm(x))
// Then:       logits = lm_head · norm(x_last)
//
// Matrix multiplies run through `runtime.matmul` (a WebAssembly SIMD kernel).
// Everything else (norms, RoPE, attention softmax) is plain JS.

export const MAX_TOKENS_PER_PASS = 512;
const ATTN_TRACE_LIMIT = 160 * 160; // keep full T×N attention maps up to this size

function rmsnorm(out, x, w, T, D, eps) {
  for (let t = 0; t < T; t++) {
    const o = t * D;
    let ss = 0;
    for (let i = 0; i < D; i++) ss += x[o + i] * x[o + i];
    const inv = 1 / Math.sqrt(ss / D + eps);
    for (let i = 0; i < D; i++) out[o + i] = x[o + i] * inv * w[i];
  }
}

// One query head at one position: scores against keys 0..pos, softmax, weighted
// sum of values. Kept small and separate so the JIT optimises it early.
// Leaves the softmax probabilities in `scores`.
function attendOne(q, qo, K, V, kvo, kvDim, HD, pos, scale, scores, out) {
  let max = -Infinity;
  for (let j = 0; j <= pos; j++) {
    const ko = j * kvDim + kvo;
    let s = 0;
    for (let d = 0; d < HD; d++) s += q[qo + d] * K[ko + d];
    s *= scale;
    scores[j] = s;
    if (s > max) max = s;
  }
  let sum = 0;
  for (let j = 0; j <= pos; j++) { const e = Math.exp(scores[j] - max); scores[j] = e; sum += e; }
  const inv = 1 / sum;
  for (let j = 0; j <= pos; j++) {
    const p = scores[j] * inv;
    scores[j] = p;
    const vo = j * kvDim + kvo;
    for (let d = 0; d < HD; d++) out[qo + d] += p * V[vo + d];
  }
}

export function modelDims(c) {
  const D = c.hidden_size, H = c.num_attention_heads;
  const KVH = c.num_key_value_heads ?? H;
  const HD = c.head_dim ?? D / H;
  return { D, L: c.num_hidden_layers, H, KVH, HD, I: c.intermediate_size, V: c.vocab_size };
}

// Floats of scratch activations the model allocates from the runtime arena.
export function scratchFloats(c, maxT = MAX_TOKENS_PER_PASS) {
  const { D, KVH, HD, I, V } = modelDims(c);
  return maxT * (5 * D + 2 * KVH * HD + 2 * I) + 2 * D + 2 * V + 4096;
}

export class LlamaModel {
  // weights: Map name -> Float32Array (views into the runtime arena)
  constructor(config, weights, runtime) {
    const c = config;
    this.config = c;
    Object.assign(this, modelDims(c));
    this.rt = runtime;
    this.eps = c.rms_norm_eps ?? 1e-5;
    this.theta = c.rope_theta ?? 10000;
    const g = n => {
      const w = weights.get(n);
      if (!w) throw new Error(`Missing tensor ${n}`);
      return w;
    };
    this.embed = g('model.embed_tokens.weight');
    this.lmHead = weights.get('lm_head.weight') || this.embed; // tied embeddings
    this.finalNorm = g('model.norm.weight');
    this.layers = [];
    for (let l = 0; l < this.L; l++) {
      const p = `model.layers.${l}.`;
      this.layers.push({
        inNorm: g(p + 'input_layernorm.weight'),
        wq: g(p + 'self_attn.q_proj.weight'),
        wk: g(p + 'self_attn.k_proj.weight'),
        wv: g(p + 'self_attn.v_proj.weight'),
        wo: g(p + 'self_attn.o_proj.weight'),
        postNorm: g(p + 'post_attention_layernorm.weight'),
        wg: g(p + 'mlp.gate_proj.weight'),
        wu: g(p + 'mlp.up_proj.weight'),
        wd: g(p + 'mlp.down_proj.weight'),
      });
    }
    // RoPE inverse frequencies: θ^(-2i/d) for each rotated pair i.
    const half = this.HD / 2;
    this.invFreq = new Float64Array(half);
    for (let i = 0; i < half; i++) this.invFreq[i] = Math.pow(this.theta, -(2 * i) / this.HD);

    // Scratch activations live in the arena so the kernel can read/write them.
    const { D, KVH, HD, I, V } = this;
    const M = MAX_TOKENS_PER_PASS, kvDim = KVH * HD;
    const a = n => runtime.alloc(n);
    this.s = {
      x: a(M * D), h: a(M * D), q: a(M * D), att: a(M * D), ao: a(M * D),
      k: a(M * kvDim), v: a(M * kvDim), gate: a(M * I), up: a(M * I),
      last: a(D), nrm: a(D), logits: a(V), lens: a(V),
    };
    this.resetCache();
  }

  resetCache(capacity = 256) {
    const kvDim = this.KVH * this.HD;
    this.cap = capacity;
    this.kcache = Array.from({ length: this.L }, () => new Float32Array(capacity * kvDim));
    this.vcache = Array.from({ length: this.L }, () => new Float32Array(capacity * kvDim));
    this.cacheLen = 0;
  }

  ensureCapacity(n) {
    if (n <= this.cap) return;
    let cap = this.cap;
    while (cap < n) cap *= 2;
    const kvDim = this.KVH * this.HD;
    for (let l = 0; l < this.L; l++) {
      const k = new Float32Array(cap * kvDim); k.set(this.kcache[l]); this.kcache[l] = k;
      const v = new Float32Array(cap * kvDim); v.set(this.vcache[l]); this.vcache[l] = v;
    }
    this.cap = cap;
  }

  // Rotate pairs (i, i + d/2) of each head by angle pos·invFreq[i] (HF "rotate_half").
  rope(buf, T, nHeads, startPos) {
    const HD = this.HD, half = HD / 2, stride = nHeads * HD;
    for (let t = 0; t < T; t++) {
      const pos = startPos + t;
      for (let i = 0; i < half; i++) {
        const ang = pos * this.invFreq[i];
        const cos = Math.cos(ang), sin = Math.sin(ang);
        for (let h = 0; h < nHeads; h++) {
          const o = t * stride + h * HD;
          const x1 = buf[o + i], x2 = buf[o + i + half];
          buf[o + i] = x1 * cos - x2 * sin;
          buf[o + i + half] = x2 * cos + x1 * sin;
        }
      }
    }
  }

  // Process T tokens starting at position startPos, appending to the KV cache.
  // opts.trace: collect intermediate values for visualisation.
  // opts.onLayer(l): progress callback after each layer.
  forward(tokens, startPos, opts = {}) {
    const { D, L, H, KVH, HD, I, V, eps, rt } = this;
    const T = tokens.length;
    if (T > MAX_TOKENS_PER_PASS) throw new Error(`At most ${MAX_TOKENS_PER_PASS} tokens per pass`);
    const N = startPos + T; // total context after this step
    const kvDim = KVH * HD;
    const group = H / KVH;
    const scale = 1 / Math.sqrt(HD);
    const s = this.s;
    const x = s.x.subarray(0, T * D), h = s.h.subarray(0, T * D), q = s.q.subarray(0, T * D);
    const att = s.att.subarray(0, T * D), ao = s.ao.subarray(0, T * D);
    const k = s.k.subarray(0, T * kvDim), v = s.v.subarray(0, T * kvDim);
    const gate = s.gate.subarray(0, T * I), up = s.up.subarray(0, T * I);
    const scores = new Float32Array(N);
    this.ensureCapacity(N);
    this.cacheLen = startPos;

    const fullAttn = T * N <= ATTN_TRACE_LIMIT;
    const tr = opts.trace ? {
      T, N, startPos, fullAttn,
      embed: new Float32Array(T * D),
      // [layer][head][query][key]; if the map would be too big, only the last query row.
      attn: new Float32Array(L * H * (fullAttn ? T : 1) * N),
      resNorm: new Float32Array((L + 1) * T),
      ffnAct: new Float32Array(L * I),       // last token, silu(gate) ⊙ up
      hidden: new Float32Array(L * D),       // last token residual after each layer
      layerMs: new Float64Array(L),
      opMs: { norm: 0, qkv: 0, rope: 0, attn: 0, oproj: 0, ffn: 0, head: 0, embed: 0 },
      qkv: null,
    } : null;
    const op = tr ? tr.opMs : { norm: 0, qkv: 0, rope: 0, attn: 0, oproj: 0, ffn: 0, head: 0, embed: 0 };
    let c = performance.now();
    const lap = key => { const n = performance.now(); op[key] += n - c; c = n; };

    const t0 = c;
    // 1. Embedding lookup: copy row tokens[t] of the [V × D] embedding matrix.
    for (let t = 0; t < T; t++) x.set(this.embed.subarray(tokens[t] * D, tokens[t] * D + D), t * D);
    lap('embed');
    if (tr) {
      tr.embed.set(x);
      for (let t = 0; t < T; t++) tr.resNorm[t] = norm(x, t * D, D);
    }

    for (let l = 0; l < L; l++) {
      const lt = performance.now();
      c = lt;
      const W = this.layers[l];
      // --- attention block ---
      rmsnorm(h, x, W.inNorm, T, D, eps); lap('norm');
      rt.matmul(q, h, W.wq, T, D, D);
      rt.matmul(k, h, W.wk, T, D, kvDim);
      rt.matmul(v, h, W.wv, T, D, kvDim);
      lap('qkv');
      let qPre, kPre;
      if (tr && l === 0) { qPre = q.slice((T - 1) * D, (T - 1) * D + HD); kPre = k.slice((T - 1) * kvDim, (T - 1) * kvDim + HD); }
      this.rope(q, T, H, startPos);
      this.rope(k, T, KVH, startPos);
      if (tr && l === 0) {
        tr.qkv = {
          pos: N - 1, qPre, kPre,
          q: q.slice((T - 1) * D, (T - 1) * D + HD),
          k: k.slice((T - 1) * kvDim, (T - 1) * kvDim + HD),
          v: v.slice((T - 1) * kvDim, (T - 1) * kvDim + HD),
          // all heads, for showing how the vectors split into heads
          qAll: q.slice((T - 1) * D, T * D),
          kAll: k.slice((T - 1) * kvDim, T * kvDim),
          vAll: v.slice((T - 1) * kvDim, T * kvDim),
        };
      }
      // Append this step's K and V to the cache.
      this.kcache[l].set(k, startPos * kvDim);
      this.vcache[l].set(v, startPos * kvDim);
      lap('rope');
      const K = this.kcache[l], Vc = this.vcache[l];
      att.fill(0);
      for (let hh = 0; hh < H; hh++) {
        const kvo = ((hh / group) | 0) * HD; // grouped-query attention: heads share K/V
        for (let t = 0; t < T; t++) {
          const pos = startPos + t; // causal: attend to keys 0..pos only
          const qo = t * D + hh * HD;
          attendOne(q, qo, K, Vc, kvo, kvDim, HD, pos, scale, scores, att);
          if (tr && (fullAttn || t === T - 1)) {
            tr.attn.set(scores.subarray(0, pos + 1), fullAttn ? ((l * H + hh) * T + t) * N : (l * H + hh) * N);
          }
        }
      }
      lap('attn');
      rt.matmul(ao, att, W.wo, T, D, D);
      for (let i = 0; i < T * D; i++) x[i] += ao[i]; // residual
      lap('oproj');
      // --- feed-forward block (SwiGLU) ---
      rmsnorm(h, x, W.postNorm, T, D, eps); lap('norm');
      rt.matmul(gate, h, W.wg, T, D, I);
      rt.matmul(up, h, W.wu, T, D, I);
      for (let i = 0; i < T * I; i++) {
        const g = gate[i];
        gate[i] = (g / (1 + Math.exp(-g))) * up[i]; // silu(g) ⊙ u
      }
      if (tr) tr.ffnAct.set(gate.subarray((T - 1) * I, T * I), l * I);
      rt.matmul(ao, gate, W.wd, T, I, D);
      for (let i = 0; i < T * D; i++) x[i] += ao[i]; // residual
      lap('ffn');
      if (tr) {
        for (let t = 0; t < T; t++) tr.resNorm[(l + 1) * T + t] = norm(x, t * D, D);
        tr.hidden.set(x.subarray((T - 1) * D, T * D), l * D);
        tr.layerMs[l] = performance.now() - lt;
      }
      opts.onLayer?.(l);
    }
    this.cacheLen = N;

    // Final norm + LM head, only for the last position (the one we predict from).
    c = performance.now();
    rmsnorm(s.last, x.subarray((T - 1) * D), this.finalNorm, 1, D, eps);
    rt.matmul(s.logits, s.last, this.lmHead, 1, D, V);
    lap('head');
    const logits = s.logits.slice();
    if (tr) tr.totalMs = performance.now() - t0;
    return { logits, trace: tr, ms: performance.now() - t0 };
  }

  // "Logit lens": what would the model predict if we stopped after layer l?
  logitLens(hidden, topK = 5) {
    const { D, L, V } = this;
    const out = [];
    for (let l = 0; l < L; l++) {
      rmsnorm(this.s.nrm, hidden.subarray(l * D, l * D + D), this.finalNorm, 1, D, this.eps);
      this.rt.matmul(this.s.lens, this.s.nrm, this.lmHead, 1, D, V);
      out.push(topk(softmax(this.s.lens), topK));
    }
    return out;
  }

  get paramCount() {
    const { D, L, KVH, HD, I, V } = this;
    const perLayer = D * D * 2 + D * KVH * HD * 2 + 3 * D * I + 2 * D;
    return V * D + L * perLayer + D + (this.lmHead === this.embed ? 0 : V * D);
  }
}

function norm(a, o, n) {
  let s = 0;
  for (let i = 0; i < n; i++) s += a[o + i] * a[o + i];
  return Math.sqrt(s);
}

export function softmax(logits, temperature = 1) {
  const n = logits.length;
  const out = new Float32Array(n);
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < n; i++) { const e = Math.exp((logits[i] - max) / temperature); out[i] = e; sum += e; }
  for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}

export function topk(arr, k) {
  const idx = [];
  for (let i = 0; i < arr.length; i++) {
    if (idx.length < k) { idx.push(i); idx.sort((a, b) => arr[b] - arr[a]); }
    else if (arr[i] > arr[idx[k - 1]]) { idx[k - 1] = i; idx.sort((a, b) => arr[b] - arr[a]); }
  }
  return idx.map(i => ({ id: i, p: arr[i] }));
}
