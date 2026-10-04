// Main thread = the "CPU". Owns the tokenizer, the sampler and the UI; sends token
// IDs to the worker (the "GPU") and gets logits back.

import { MODELS, fileUrl, chatPrompt } from './models.js';
import { BPETokenizer } from './engine/tokenizer.js';
import { fetchHeader } from './engine/safetensors.js';
import { EngineClient } from './engine/client.js';
import { $, h, fmtBytes, fmtMs, fmtInt, debounce, samplingDistribution, sampleFrom } from './util.js';
import { createMachine } from './viz/machine.js';

import { initGlossary } from './glossary.js';
import primer from './chapters/primer.js';
import artifact from './chapters/artifact.js';
import tokenizer from './chapters/tokenizer.js';
import embed from './chapters/embed.js';
import rope from './chapters/rope.js';
import loop from './chapters/loop.js';
import journey from './chapters/journey.js';
import attention from './chapters/attention.js';
import ffn from './chapters/ffn.js';
import sample from './chapters/sample.js';
import tracer from './chapters/tracer.js';
import kv from './chapters/kv.js';
import roofline from './chapters/roofline.js';
import prefix from './chapters/prefix.js';
import summary from './chapters/summary.js';

const S = {
  meta: MODELS[0],
  config: null,
  tok: null,
  header: null,
  headerError: null,
  files: {},
  loaded: false,
  loading: false,
  loadInfo: null,
  previews: null,
  // current prompt
  text: '',
  ids: [],
  enc: null,
  tokenizeMs: 0,
  // generation
  run: null,
  runs: [],
  running: false,
  stopRequested: false,
  selStep: -1,
  settings: { temperature: 0.7, topK: 40, topP: 0.95, maxNew: 48, chat: true, system: '', prefixCache: true, kvCache: true },
};

let engine = null;
const chapters = [];
const listeners = [];
const emit = (reason, data) => {
  for (const c of chapters) {
    try { c.update?.(reason, data); } catch (e) { console.error(`chapter update failed (${reason})`, e); }
  }
  for (const f of listeners) f(reason, data);
};

const ctx = {
  S,
  emit,
  on: f => listeners.push(f),
  get engine() { return engine; },
  load: () => loadWeights(),
  generate: opts => generate(opts),
  branch: (step, tokenId) => branch(step, tokenId),
  setPrompt: (text, run) => { $('#prompt').value = text; onPromptInput(); if (run) generate({}); },
  setSystem: text => { $('#system-prompt').value = text; S.settings.system = text; onPromptInput(); },
  selectStep: i => { S.selStep = i; renderOutput(); emit('select'); },
  eosIds: () => (S.meta.chat ? [2, 0] : [0]),
  dims() {
    const c = S.config;
    if (!c) return null;
    const D = c.hidden_size, H = c.num_attention_heads, KVH = c.num_key_value_heads ?? H;
    const HD = c.head_dim ?? D / H;
    const L = c.num_hidden_layers, I = c.intermediate_size, V = c.vocab_size;
    const perLayer = 2 * D * D + 2 * D * KVH * HD + 3 * D * I + 2 * D;
    const tied = c.tie_word_embeddings !== false;
    const params = V * D + L * perLayer + D + (tied ? 0 : V * D);
    // Weights touched to produce one token: every layer + the LM head (embedding row reads are negligible).
    const activeParams = L * perLayer + D + V * D;
    return {
      D, H, KVH, HD, L, I, V, perLayer, params, tied, activeParams,
      kvBytesPerToken: 2 * L * KVH * HD * 2, // K and V, every layer, bf16
      bytesPerParam: 2,
    };
  },
};

// ---------------------------------------------------------------- model setup
async function selectModel(meta) {
  S.meta = meta;
  S.config = S.tok = S.header = null;
  S.headerError = null;
  S.loaded = false;
  S.previews = null;
  S.run = null;
  S.runs = [];
  S.selStep = -1;
  if (engine) engine.worker.terminate();
  engine = new EngineClient();
  engine.on('progress', p => emit('load-progress', p));
  engine.on('layer', p => { emit('layer', p); setStatus(`Prefill: layer ${p.l + 1} of ${p.L}`, 'busy'); });
  $('#hero-model').textContent = meta.label;
  $('#hero-size').textContent = `${meta.downloadMB} MB`;
  $('#system-wrap').hidden = !meta.chat;
  $('#console-load .dim').textContent = `${meta.downloadMB} MB`;
  $('#opt-chat').closest('label').hidden = !meta.chat;
  S.settings.chat = meta.chat && $('#opt-chat').checked;
  S.settings.system = meta.systemPrompt || '';
  $('#system-prompt').value = S.settings.system;
  setStatus('Reading config and tokenizer…', 'loading');
  updateButtons();
  try {
    const [cfgText, tokText] = await Promise.all([
      fetch(fileUrl(meta, 'config.json')).then(r => r.text()),
      fetch(fileUrl(meta, 'tokenizer.json')).then(r => r.text()),
    ]);
    S.files = { config: cfgText.length, tokenizer: tokText.length };
    S.config = JSON.parse(cfgText);
    S.tok = new BPETokenizer(JSON.parse(tokText));
  } catch (e) {
    setStatus('Could not reach Hugging Face', 'idle');
    $('#console-note').textContent = `Couldn't download the config and tokenizer (${e.message}). Check your connection and reload.`;
    return;
  }
  emit('model');
  onPromptInput();
  setStatus('Weights not loaded', 'idle');
  try {
    S.header = await fetchHeader(fileUrl(meta, 'model.safetensors'));
  } catch (e) {
    S.headerError = e.message;
  }
  emit('header');
  if (!S.loaded && !S.loading && await isCached(meta)) {
    setStatus('Weights in browser cache, not loaded yet', 'idle');
  }
  updateButtons();
}

async function isCached(meta) {
  try {
    const c = await caches.open('inference-opened-weights-v1');
    return !!(await c.match(fileUrl(meta, 'model.safetensors')));
  } catch { return false; }
}

async function loadWeights() {
  if (S.loaded || S.loading || !S.config) return;
  S.loading = true;
  updateButtons();
  setStatus('Downloading weights…', 'loading');
  emit('load-start');
  try {
    const info = await engine.call('load', {
      configUrl: fileUrl(S.meta, 'config.json'),
      weightsUrl: fileUrl(S.meta, 'model.safetensors'),
    });
    S.loadInfo = info;
    S.previews = info.previews;
    S.loaded = true;
    setStatus(`Weights in memory: ${fmtBytes(info.deviceBytes)}`, 'ready');
    $('#console-note').textContent = 'Ready. Generate runs to the end; “One token at a time” lets you step through decode.';
    emit('weights');
  } catch (e) {
    setStatus('Loading failed', 'idle');
    emit('load-error', e);
    $('#console-note').textContent = `Loading failed: ${e.message}`;
  } finally {
    S.loading = false;
    updateButtons();
  }
}

// ---------------------------------------------------------------- prompt
function fullText() {
  const user = $('#prompt').value;
  return S.settings.chat ? chatPrompt(S.meta, S.settings.system, user) : user;
}

function onPromptInput() {
  if (!S.tok) return;
  S.text = fullText();
  const t = performance.now();
  S.enc = S.tok.encodeWithTrace(S.text);
  S.tokenizeMs = performance.now() - t;
  S.ids = S.enc.ids;
  emit('tokens');
  updateButtons();
}

// ---------------------------------------------------------------- generation
function sameIds(a, b) {
  if (!a || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function makeStep(res, kind) {
  const run = S.run;
  const dist = samplingDistribution(res.logits, S.settings);
  const choice = sampleFrom(dist);
  const step = {
    index: run.steps.length,
    kind,
    pos: run.ids.length + run.steps.length, // position the sampled token will occupy
    logits: res.logits,
    trace: res.trace,
    ms: res.ms,
    noCache: !!res.noCache,
    token: choice.id,
    p: choice.p,
    pRaw: choice.pRaw,
    settings: { ...S.settings },
  };
  run.steps.push(step);
  // Free old logits to bound memory; keep the recent ones for re-sampling.
  if (run.steps.length > 128) run.steps[run.steps.length - 129].logits = null;
  // Keep a full trace only for the prefill and the latest step.
  if (run.steps.length > 2) run.steps[run.steps.length - 2].trace = null;
  return step;
}

async function startRun() {
  const ids = S.ids.slice();
  S.run = {
    n: S.runs.length + 1,
    ids,
    text: S.text,
    steps: [],
    tokenizeMs: S.tokenizeMs,
    t0: performance.now(),
    done: false,
    decoder: new TextDecoder(),
  };
  S.selStep = -1;
  renderOutput();
  emit('run-start');
  setStatus('Prefill…', 'busy');
  const res = await engine.call('prefill', { ids, prefixCache: S.settings.prefixCache });
  const run = S.run;
  run.prefill = { ms: res.ms, reused: res.reused, computed: res.computed, trace: res.trace, wall: performance.now() - run.t0 };
  const step = makeStep(res, 'prefill');
  run.ttft = performance.now() - run.t0;
  S.runs.push({ n: run.n, tokens: ids.length, reused: res.reused, computed: res.computed, ms: res.ms, ttft: run.ttft, prefixCache: S.settings.prefixCache });
  emit('prefill');
  afterStep(step);
}

function afterStep(step) {
  const run = S.run;
  if (ctx.eosIds().includes(step.token)) run.done = true;
  if (run.steps.length >= S.settings.maxNew) run.done = true;
  S.selStep = run.steps.length - 1;
  renderOutput();
  emit('step', step);
}

async function decodeOne() {
  const run = S.run;
  const prev = run.steps[run.steps.length - 1];
  const res = await engine.call('decode', { token: prev.token, pos: prev.pos, noCache: !S.settings.kvCache });
  res.noCache = !S.settings.kvCache;
  const step = makeStep(res, 'decode');
  afterStep(step);
}

async function generate({ single = false } = {}) {
  if (!S.loaded || S.running) return;
  S.running = true;
  S.stopRequested = false;
  $('#console-note').textContent = single ? 'Stepping: each click runs one forward pass.' : 'Generating. Pause stops after the current token.';
  updateButtons();
  let failed = false;
  try {
    const fresh = !S.run || !sameIds(S.run.ids, S.ids) || S.run.done;
    if (fresh) {
      await startRun();
      if (single) return;
    }
    while (!S.run.done && !S.stopRequested) {
      setStatus(`Decoding token ${S.run.steps.length + 1}`, 'busy');
      await decodeOne();
      if (single) break;
      await new Promise(r => setTimeout(r, 0)); // let the page paint
    }
  } catch (e) {
    console.error(e);
    failed = true;
    $('#console-note').textContent = `Generation failed: ${e.message}`;
  } finally {
    S.running = false;
    if (S.run?.steps.length && !failed) {
      $('#console-note').textContent = S.run.done
        ? 'Done. Click any word of the answer to see how it was picked.'
        : 'Paused. Generate continues; Step adds one token.';
    }
    setStatus(S.loaded ? (S.run?.done ? 'Done' : 'Paused') : 'Weights not loaded', S.loaded ? 'ready' : 'idle');
    updateButtons();
    emit('idle');
  }
}

// Replace the token chosen at `stepIndex` and drop everything after it.
function branch(stepIndex, tokenId) {
  const run = S.run;
  if (!run || S.running) return;
  const step = run.steps[stepIndex];
  if (!step?.logits) return;
  run.steps.length = stepIndex + 1;
  const dist = samplingDistribution(step.logits, S.settings);
  const c = dist.cands.find(x => x.id === tokenId);
  step.token = tokenId;
  step.p = c ? c.p : 0;
  step.pRaw = c ? c.pRaw : 0;
  step.branched = true;
  run.done = ctx.eosIds().includes(tokenId) || run.steps.length >= S.settings.maxNew;
  S.selStep = stepIndex;
  renderOutput();
  emit('step', step);
  emit('branch', step);
  $('#console-note').textContent = `Swapped token ${stepIndex + 1}. Press Generate to continue from there.`;
}

// ---------------------------------------------------------------- output
function renderOutput() {
  const out = $('#output');
  const run = S.run;
  out.textContent = '';
  if (!run) {
    out.append(h('span', { class: 'placeholder' }, 'The model’s answer appears here, token by token.'));
    renderStats();
    return;
  }
  const eos = ctx.eosIds();
  run.steps.forEach((s, i) => {
    if (eos.includes(s.token)) return;
    const text = S.tok.tokenText(s.token);
    const span = h('span', {
      class: `tk${i === S.selStep ? ' sel' : ''}${i === run.steps.length - 1 && S.running ? ' fresh' : ''}`,
      'data-step': i,
      title: `Token ${i + 1}: id ${s.token}, p = ${(s.p * 100).toFixed(1)}%`,
    }, text);
    span.addEventListener('click', () => { ctx.selectStep(i); document.getElementById('ch-sample').scrollIntoView(); });
    out.append(span);
  });
  if (!run.steps.length) out.append(h('span', { class: 'placeholder' }, 'Prefilling the prompt…'));
  renderStats();
}

function renderStats() {
  const el = $('#run-stats');
  const run = S.run;
  el.textContent = '';
  if (!run?.prefill) return;
  const dec = run.steps.filter(s => s.kind === 'decode');
  const decMs = dec.reduce((a, s) => a + s.ms, 0);
  const kvTokens = run.ids.length + Math.max(0, run.steps.length - 1);
  const d = ctx.dims();
  const item = (label, value) => h('span', {}, `${label} `, h('b', {}, value));
  el.append(...[
    item('Prompt', `${run.ids.length} tokens`),
    run.prefill.reused ? item('Reused', `${run.prefill.reused} tokens`) : null,
    item('First token after', fmtMs(run.ttft)),
    dec.length ? item('Decode', `${(dec.length / (decMs / 1000)).toFixed(1)} tokens/s`) : null,
    item('KV cache', fmtBytes(kvTokens * d.kvBytesPerToken)),
  ].filter(Boolean));
}

// ---------------------------------------------------------------- chrome
function setStatus(text, state) {
  const el = $('#status');
  el.textContent = text;
  el.dataset.state = state;
}

function updateButtons() {
  const ready = S.loaded && !S.running;
  $('#btn-run').disabled = !ready || !S.ids.length;
  $('#btn-step').disabled = !ready || !S.ids.length;
  $('#btn-stop').disabled = !S.running;
  $('#btn-run').textContent = S.run && !S.run.done && S.run && sameIds(S.run.ids, S.ids) ? 'Continue' : 'Generate';
  for (const b of document.querySelectorAll('[data-action="load"]')) b.disabled = S.loaded || S.loading || !S.config;
  $('#hero-load').disabled = S.loaded || S.loading || !S.config;
  const cl = $('#console-load');
  cl.hidden = S.loaded;
  cl.disabled = S.loading || !S.config;
  cl.firstChild.textContent = S.loading ? 'Loading… ' : 'Load the model ';
  $('#hero-load').firstChild.textContent = S.loaded ? 'Model loaded ' : S.loading ? 'Loading… ' : 'Load the model ';
}

function initChrome() {
  const sel = $('#model-select');
  for (const m of MODELS) sel.append(h('option', { value: m.id }, m.label));
  sel.addEventListener('change', () => {
    if (S.running) { sel.value = S.meta.id; return; }
    selectModel(MODELS.find(m => m.id === sel.value));
  });
  $('#prompt').addEventListener('input', debounce(onPromptInput, 120));
  $('#system-prompt').addEventListener('input', debounce(() => { S.settings.system = $('#system-prompt').value; onPromptInput(); }, 120));
  $('#opt-chat').addEventListener('change', e => { S.settings.chat = e.target.checked; onPromptInput(); });
  $('#opt-max').addEventListener('change', e => { S.settings.maxNew = Math.max(1, Math.min(400, +e.target.value || 48)); });
  $('#btn-run').addEventListener('click', () => generate({}));
  $('#btn-step').addEventListener('click', () => generate({ single: true }));
  $('#btn-stop').addEventListener('click', () => { S.stopRequested = true; });
  $('#hero-load').addEventListener('click', () => { loadWeights(); document.getElementById('ch-artifact').scrollIntoView(); });
  $('#console-load').addEventListener('click', () => loadWeights());
  $('#prompt').addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); generate({}); }
  });
}

async function initHero() {
  try {
    const { createHero } = await import('./viz/hero3d.js');
    const hero = createHero($('#stage3d'), ctx);
    ctx.on((reason, data) => hero.update(reason, data));
    if (S.header || S.config) hero.update(S.header ? 'header' : 'model');
  } catch (e) {
    console.warn('3D view unavailable', e);
    $('#stage3d').append(h('p', { class: 'empty' }, 'The 3D view needs WebGL, which isn’t available here. Everything else on the page works without it.'));
  }
}

// The console lives in the right column on wide screens so it is always at hand,
// and in the page flow on narrow screens (where the right column is a bottom strip).
function placeConsole() {
  const wide = matchMedia('(min-width: 1181px)');
  const move = () => {
    const target = wide.matches ? $('#console-dock') : $('#console-home');
    if ($('#console').parentElement !== target) target.append($('#console'));
  };
  wide.addEventListener('change', move);
  move();
}

function init() {
  // Handy for poking around in DevTools: inference.S.run.prefill.trace, etc.
  globalThis.inference = ctx;
  initChrome();
  placeConsole();
  const machine = createMachine($('#machine'), $('#machine-caption'), ctx);
  ctx.machine = machine;
  initGlossary($('#tooltip'));
  for (const mod of [primer, artifact, tokenizer, embed, rope, loop, journey, attention, ffn, sample, tracer, kv, roofline, prefix, summary]) {
    chapters.push(mod(ctx));
  }
  chapters.push(machine);
  // Machine map follows the chapter in view.
  const io = new IntersectionObserver(entries => {
    const top = entries.filter(e => e.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (top) machine.focus(top.target.dataset.map);
  }, { rootMargin: '-35% 0px -55% 0px', threshold: [0, 0.01] });
  document.querySelectorAll('[data-map]').forEach(s => io.observe(s));
  initHero();
  selectModel(S.meta);
}

init();
