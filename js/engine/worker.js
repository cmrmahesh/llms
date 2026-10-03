// The Web Worker plays the GPU. The main thread (the "CPU") tokenizes, samples and
// renders; this worker owns the weights and the KV cache and runs forward passes.

import { parseHeader, toFloat32, numel } from './safetensors.js';
import { LlamaModel, scratchFloats, MAX_TOKENS_PER_PASS } from './model.js';
import { createRuntime } from './runtime.js';

const BLOCK = 16; // prefix-cache block size in tokens (vLLM's default)
const CACHE_NAME = 'inference-opened-weights-v1';

let model = null;
let runtime = null;
let seq = [];        // token ids whose K/V are currently in the cache, by position
let lastHidden = null;
let prefillHidden = null; // residual stream of the last prompt token, per layer
let previews = null;

const post = (msg, transfer = []) => self.postMessage(msg, transfer);

async function download(url, id) {
  let cache = null;
  try { cache = await caches.open(CACHE_NAME); } catch { /* no Cache Storage */ }
  const hit = cache && await cache.match(url);
  if (hit) {
    post({ type: 'progress', id, phase: 'cache', loaded: 1, total: 1 });
    return { buffer: await hit.arrayBuffer(), fromCache: true };
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Download failed: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  let buf = total ? new Uint8Array(total) : null;
  const chunks = [];
  let loaded = 0, lastPost = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (buf && loaded + value.length <= total) buf.set(value, loaded); else chunks.push(value);
    loaded += value.length;
    const now = performance.now();
    if (now - lastPost > 100) { post({ type: 'progress', id, phase: 'download', loaded, total }); lastPost = now; }
  }
  if (!buf || chunks.length) {
    buf = new Uint8Array(loaded);
    let o = 0;
    for (const c of chunks) { buf.set(c, o); o += c.length; }
  }
  post({ type: 'progress', id, phase: 'download', loaded, total: loaded });
  if (cache) {
    try { await cache.put(url, new Response(buf, { headers: { 'content-type': 'application/octet-stream' } })); }
    catch (e) { console.warn('Could not cache weights', e); }
  }
  return { buffer: buf.buffer, fromCache: false };
}

// A small point-sampled grid of each 2-D tensor, plus summary stats, for drawing.
function makePreviews(weights, tensors) {
  const out = [];
  for (const t of tensors) {
    const w = weights.get(t.name);
    let mean = 0, sq = 0, absmax = 0;
    const stride = Math.max(1, Math.floor(w.length / 200000));
    let n = 0;
    for (let i = 0; i < w.length; i += stride) {
      const v = w[i]; mean += v; sq += v * v; n++;
      if (Math.abs(v) > absmax) absmax = Math.abs(v);
    }
    mean /= n;
    const std = Math.sqrt(Math.max(0, sq / n - mean * mean));
    let grid = null, gr = 0, gc = 0;
    if (t.shape.length === 2) {
      const [R, C] = t.shape;
      gr = Math.max(8, Math.min(256, Math.round(64 * Math.sqrt(R / C))));
      gc = Math.max(8, Math.min(256, Math.round(64 * Math.sqrt(C / R))));
      grid = new Float32Array(gr * gc);
      for (let i = 0; i < gr; i++) {
        const r = Math.floor((i + 0.5) * R / gr);
        for (let j = 0; j < gc; j++) grid[i * gc + j] = w[r * C + Math.floor((j + 0.5) * C / gc)];
      }
    }
    out.push({ name: t.name, mean, std, absmax, grid, gr, gc });
  }
  return out;
}

async function load({ id, configUrl, weightsUrl }) {
  const t0 = performance.now();
  const config = await (await fetch(configUrl)).json();
  if (!runtime) runtime = await createRuntime(new URL('./kernels.wasm', import.meta.url));
  const { buffer, fromCache } = await download(weightsUrl, id);
  const tDownload = performance.now() - t0;
  const header = parseHeader(buffer);
  const floats = header.tensors.reduce((a, t) => a + numel(t) + 4, 0) + scratchFloats(config);
  runtime.reserve(floats);
  const weights = new Map();
  const tc = performance.now();
  let done = 0;
  for (const t of header.tensors) {
    weights.set(t.name, toFloat32(buffer, t, runtime.alloc(numel(t))));
    if (++done % 16 === 0) post({ type: 'progress', id, phase: 'convert', loaded: done, total: header.tensors.length });
  }
  const tConvert = performance.now() - tc;
  model = new LlamaModel(config, weights, runtime);
  // Warm-up pass, like engines do at startup (compile kernels, capture CUDA graphs):
  // here it lets the JavaScript JIT optimise the attention loop before your first prompt.
  post({ type: 'progress', id, phase: 'warmup', loaded: 0, total: 1 });
  const tw = performance.now();
  const warm = Array.from({ length: 24 }, (_, i) => 1000 + i);
  model.forward(warm, 0, { trace: true });
  model.forward(warm.slice(16), 16, { trace: true });
  for (let i = 0; i < 4; i++) model.forward([1100 + i], 24 + i, { trace: true });
  model.resetCache();
  const tWarm = performance.now() - tw;
  seq = [];
  previews = makePreviews(weights, header.tensors);
  post({
    id, type: 'loaded',
    fromCache, tDownload, tConvert, tWarm,
    fileBytes: buffer.byteLength,
    deviceBytes: runtime.bytesUsed,
    runtime: runtime.kind,
    params: model.paramCount,
    previews,
  });
}

function collectBuffers(tr) {
  if (!tr) return [];
  const b = [tr.embed.buffer, tr.attn.buffer, tr.resNorm.buffer, tr.ffnAct.buffer, tr.hidden.buffer, tr.layerMs.buffer];
  if (tr.qkv) for (const k of ['qPre', 'kPre', 'q', 'k', 'v', 'qAll', 'kAll', 'vAll']) b.push(tr.qkv[k].buffer);
  return b;
}

// Run `tokens` starting at `start`, in chunks if needed. Only the last chunk is traced.
function run(tokens, start, trace, reportLayers) {
  let res = null;
  let ms = 0;
  for (let i = 0; i < tokens.length; i += MAX_TOKENS_PER_PASS) {
    const chunk = tokens.slice(i, i + MAX_TOKENS_PER_PASS);
    const last = i + MAX_TOKENS_PER_PASS >= tokens.length;
    res = model.forward(chunk, start + i, {
      trace: trace && last,
      onLayer: reportLayers ? l => post({ type: 'layer', l, L: model.L }) : undefined,
    });
    ms += res.ms;
  }
  if (res.trace) lastHidden = res.trace.hidden.slice();
  return { ...res, ms };
}

function finish(id, kind, res, extra) {
  const msg = { id, type: kind, logits: res.logits, trace: res.trace, ms: res.ms, seqLen: seq.length, ...extra };
  post(msg, [res.logits.buffer, ...collectBuffers(res.trace)]);
}

function prefill({ id, ids, prefixCache, trace = true }) {
  let reuse = 0;
  if (prefixCache) {
    while (reuse < ids.length && reuse < seq.length && ids[reuse] === seq[reuse]) reuse++;
    reuse = Math.floor(reuse / BLOCK) * BLOCK;
    // Always compute at least the last token: we need its logits.
    if (reuse >= ids.length) reuse = Math.floor((ids.length - 1) / BLOCK) * BLOCK;
  }
  const cachedBefore = seq.length;
  const res = run(ids.slice(reuse), reuse, trace, true);
  seq = ids.slice();
  prefillHidden = lastHidden;
  finish(id, 'prefilled', res, { reused: reuse, computed: ids.length - reuse, cachedBefore });
}

function decode({ id, token, pos, noCache, trace = true }) {
  if (pos !== undefined && pos < seq.length) seq = seq.slice(0, pos); // branch: drop later K/V
  seq.push(token);
  const res = noCache
    ? run(seq, 0, trace, false)                     // recompute every K and V from scratch
    : run([token], seq.length - 1, trace, false);   // one new token against the cache
  finish(id, 'decoded', res, { noCache: !!noCache });
}

function lens({ id }) {
  if (!prefillHidden) return post({ id, type: 'lens', layers: [] });
  post({ id, type: 'lens', layers: model.logitLens(prefillHidden, 5) });
}

// Read a block of real weights, e.g. for the weight inspector.
function slice({ id, name, r0 = 0, c0 = 0, rows, cols }) {
  const w = findWeight(name);
  if (!w) return post({ id, type: 'slice', error: `No tensor ${name}` });
  const C = shapeOf(name)[1];
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[r * cols + c] = w[(r0 + r) * C + c0 + c];
  post({ id, type: 'slice', data: out }, [out.buffer]);
}

function shapeOf(name) {
  const { D, KVH, HD, I, V } = model;
  if (name.endsWith('embed_tokens.weight')) return [V, D];
  if (name.endsWith('q_proj.weight') || name.endsWith('o_proj.weight')) return [D, D];
  if (name.endsWith('k_proj.weight') || name.endsWith('v_proj.weight')) return [KVH * HD, D];
  if (name.endsWith('gate_proj.weight') || name.endsWith('up_proj.weight')) return [I, D];
  if (name.endsWith('down_proj.weight')) return [D, I];
  return [1, D];
}

function findWeight(name) {
  if (name === 'model.embed_tokens.weight') return model.embed;
  if (name === 'model.norm.weight') return model.finalNorm;
  if (name === 'lm_head.weight') return model.lmHead;
  const m = name.match(/^model\.layers\.(\d+)\.(.+)$/);
  if (!m) return null;
  const L = model.layers[+m[1]];
  const map = {
    'input_layernorm.weight': 'inNorm', 'post_attention_layernorm.weight': 'postNorm',
    'self_attn.q_proj.weight': 'wq', 'self_attn.k_proj.weight': 'wk', 'self_attn.v_proj.weight': 'wv',
    'self_attn.o_proj.weight': 'wo', 'mlp.gate_proj.weight': 'wg', 'mlp.up_proj.weight': 'wu',
    'mlp.down_proj.weight': 'wd',
  };
  return L[map[m[2]]];
}

// The stored keys and values of one layer: [positions × kvHeads·headDim] each.
function kvLayer({ id, layer }) {
  const n = model.cacheLen, kvDim = model.KVH * model.HD;
  const K = model.kcache[layer].slice(0, n * kvDim);
  const V = model.vcache[layer].slice(0, n * kvDim);
  post({ id, type: 'kv', K, V, n }, [K.buffer, V.buffer]);
}

function embedRows({ id, ids }) {
  const D = model.D;
  const out = new Float32Array(ids.length * D);
  ids.forEach((t, i) => out.set(model.embed.subarray(t * D, t * D + D), i * D));
  post({ id, type: 'rows', data: out }, [out.buffer]);
}

// Cosine similarity of one embedding row against all of them: one [V × D] matvec.
let embedNorms = null;
function neighbors({ id, token, k = 12 }) {
  const { D, V } = model;
  if (!embedNorms) {
    embedNorms = new Float32Array(V);
    for (let r = 0; r < V; r++) {
      let s = 0;
      for (let i = r * D, e = i + D; i < e; i++) s += model.embed[i] * model.embed[i];
      embedNorms[r] = Math.sqrt(s) || 1;
    }
  }
  model.s.nrm.set(model.embed.subarray(token * D, token * D + D));
  runtime.matmul(model.s.lens, model.s.nrm, model.embed, 1, D, V);
  const sims = [];
  const qn = embedNorms[token];
  for (let r = 0; r < V; r++) {
    if (r === token) continue;
    const s = model.s.lens[r] / (qn * embedNorms[r]);
    if (sims.length < k || s > sims[sims.length - 1].s) {
      sims.push({ id: r, s });
      sims.sort((a, b) => b.s - a.s);
      if (sims.length > k) sims.pop();
    }
  }
  post({ id, type: 'neighbors', list: sims });
}

// Measure the matmul kernel at T = 1 (decode-shaped) and T = 32 (prefill-shaped).
function benchKernel({ id }) {
  const { D, I } = model;
  const W = model.layers[0].wu;
  const x = model.s.h, out = model.s.gate;
  const res = {};
  for (const T of [1, 4, 32]) {
    runtime.matmul(out, x, W, T, D, I);
    const reps = T === 1 ? 200 : 20;
    const t = performance.now();
    for (let r = 0; r < reps; r++) runtime.matmul(out, x, W, T, D, I);
    const ms = (performance.now() - t) / reps;
    const flops = 2 * D * I * T;
    const bytes = D * I * 4 + T * (D + I) * 4;
    res[T] = { ms, gflops: flops / ms / 1e6, intensity: flops / bytes };
  }
  post({ id, type: 'bench', res });
}

self.onmessage = async e => {
  const m = e.data;
  try {
    switch (m.type) {
      case 'load': await load(m); break;
      case 'prefill': prefill(m); break;
      case 'decode': decode(m); break;
      case 'lens': lens(m); break;
      case 'slice': slice(m); break;
      case 'bench': benchKernel(m); break;
      case 'rows': embedRows(m); break;
      case 'kv': kvLayer(m); break;
      case 'neighbors': neighbors(m); break;
      case 'reset': seq = []; model?.resetCache(); post({ id: m.id, type: 'reset' }); break;
      default: throw new Error(`Unknown message ${m.type}`);
    }
  } catch (err) {
    post({ id: m.id, type: 'error', error: String(err?.message || err) });
  }
};

