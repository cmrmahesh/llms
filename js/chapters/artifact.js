// 1. What you download: config, tokenizer, safetensors header, tensor browser, quantization.

import { $, h, fmtBytes, fmtInt, pct, drawGrid, diverging, fmtMs } from '../util.js';
import { numel } from '../engine/safetensors.js';

const CONFIG_NOTES = {
  hidden_size: 'Width of every token vector (D)',
  num_hidden_layers: 'Transformer layers stacked (L)',
  num_attention_heads: 'Query heads per layer (H)',
  num_key_value_heads: 'Key/value heads, shared by query heads',
  intermediate_size: 'Feed-forward width (I)',
  vocab_size: 'Vocabulary entries (V)',
  max_position_embeddings: 'Longest context trained for',
  rope_theta: 'RoPE base frequency',
  tie_word_embeddings: 'LM head reuses the embedding table',
  torch_dtype: 'Number format of the weights',
};

// Build a tensor list from the config when the header couldn't be range-read.
function synthHeader(c) {
  const D = c.hidden_size, H = c.num_attention_heads, KVH = c.num_key_value_heads ?? H, HD = D / H, I = c.intermediate_size;
  const t = [{ name: 'model.embed_tokens.weight', shape: [c.vocab_size, D], dtype: 'BF16' }];
  for (let l = 0; l < c.num_hidden_layers; l++) {
    const p = `model.layers.${l}.`;
    t.push(
      { name: p + 'input_layernorm.weight', shape: [D] },
      { name: p + 'self_attn.q_proj.weight', shape: [D, D] },
      { name: p + 'self_attn.k_proj.weight', shape: [KVH * HD, D] },
      { name: p + 'self_attn.v_proj.weight', shape: [KVH * HD, D] },
      { name: p + 'self_attn.o_proj.weight', shape: [D, D] },
      { name: p + 'post_attention_layernorm.weight', shape: [D] },
      { name: p + 'mlp.gate_proj.weight', shape: [I, D] },
      { name: p + 'mlp.up_proj.weight', shape: [I, D] },
      { name: p + 'mlp.down_proj.weight', shape: [D, I] },
    );
  }
  t.push({ name: 'model.norm.weight', shape: [D] });
  let off = 0;
  for (const x of t) { x.dtype = 'BF16'; x.begin = off; off += numel(x) * 2; x.end = off; }
  return { tensors: t, headerBytes: null, synthetic: true };
}

function group(name) {
  if (name.includes('embed_tokens') || name.includes('lm_head')) return 'emb';
  if (name.includes('self_attn')) return 'attn';
  if (name.includes('mlp')) return 'mlp';
  return 'norm';
}

export default function artifact(ctx) {
  const { S } = ctx;
  let selected = 'model.layers.0.self_attn.q_proj.weight';

  function header() { return S.header || (S.config ? synthHeader(S.config) : null); }

  function renderFiles() {
    const el = $('#files');
    el.textContent = '';
    const hd = header();
    const total = hd ? hd.tensors.reduce((a, t) => Math.max(a, t.end), 0) + (hd.headerBytes ?? 0) + 8 : 0;
    el.append(
      h('div', { class: 'file' },
        h('div', { class: 'fname' }, 'config.json'),
        h('div', { class: 'fsize' }, S.files.config ? fmtBytes(S.files.config) : '–'),
        h('p', {}, 'Architecture name and the handful of sizes below.')),
      h('div', { class: 'file' },
        h('div', { class: 'fname' }, 'tokenizer.json'),
        h('div', { class: 'fsize' }, S.files.tokenizer ? fmtBytes(S.files.tokenizer) : '–'),
        h('p', {}, S.tok ? `${fmtInt(S.tok.vocabSize)} vocabulary entries and ${fmtInt(S.tok.numMerges)} merge rules.` : 'Vocabulary and merge rules.')),
      h('div', { class: 'file big' },
        h('div', { class: 'fname' }, 'model.safetensors'),
        h('div', { class: 'fsize' }, total ? fmtBytes(total) : '–'),
        h('p', {}, hd ? `${hd.tensors.length} tensors, ${fmtInt(hd.tensors.reduce((a, t) => a + numel(t), 0))} numbers. That's ${S.files.config ? pct(total / (S.files.config + S.files.tokenizer + total)) : 'almost all'} of the download.` : 'The weights.')),
    );
  }

  function renderConfig() {
    const el = $('#config-view');
    el.textContent = '';
    const c = S.config;
    if (!c) return;
    for (const [k, note] of Object.entries(CONFIG_NOTES)) {
      if (!(k in c)) continue;
      const v = typeof c[k] === 'number' ? fmtInt(c[k]) : String(c[k]);
      el.append(h('div', { class: 'cfg' }, h('div', { class: 'k' }, k), h('div', { class: 'v' }, v), h('div', { class: 'm' }, note)));
    }
  }

  function renderSplit() {
    const el = $('#param-split');
    el.textContent = '';
    const hd = header();
    if (!hd) return;
    const sums = { emb: 0, attn: 0, mlp: 0, norm: 0 };
    for (const t of hd.tensors) sums[group(t.name)] += numel(t);
    const total = sums.emb + sums.attn + sums.mlp + sums.norm;
    const parts = [
      ['emb', 'Embedding table', '#c4cfde', 'var(--ink)'],
      ['attn', 'Attention (Q, K, V, O)', '#7f95b8', '#fff'],
      ['mlp', 'Feed-forward (gate, up, down)', '#2f4570', '#fff'],
      ['norm', 'Norms', '#14213a', '#fff'],
    ];
    const bar = h('div', { class: 'split-bar', role: 'img', 'aria-label': parts.map(([k, l]) => `${l} ${pct(sums[k] / total)}`).join(', ') });
    for (const [k, label, bg, fg] of parts) {
      const f = sums[k] / total;
      bar.append(h('div', { style: { width: `${f * 100}%`, background: bg, color: fg } }, f > 0.08 ? pct(f, 0) : ''));
    }
    const legend = h('div', { class: 'split-legend' }, parts.map(([k, label, bg]) =>
      h('span', {}, h('i', { style: { background: bg } }), `${label}: ${fmtInt(sums[k])} (${pct(sums[k] / total)})`)));
    const note = S.config?.tie_word_embeddings
      ? 'In a model this small, the vocabulary table is a fifth of all weights. In an 8B model it is closer to 7%.'
      : '';
    el.append(bar, legend);
    if (note) el.append(h('p', { class: 'caption' }, note));
    if (hd.synthetic) el.append(h('p', { class: 'caption' }, `Couldn't range-read the real header (${S.headerError}). These shapes are computed from config.json instead.`));
    else el.append(h('p', { class: 'caption' }, `Real header: ${fmtInt(hd.headerBytes)} bytes of JSON after the 8-byte length prefix.`));
  }

  function renderBrowser() {
    const el = $('#tensor-browser');
    el.textContent = '';
    const hd = header();
    if (!hd) return;
    const list = h('div', { class: 'tensor-list', role: 'list' });
    let lastGroup = '';
    for (const t of hd.tensors) {
      const m = t.name.match(/layers\.(\d+)\./);
      const g = m ? `Layer ${m[1]}` : t.name.includes('embed') ? 'Embedding' : 'Output';
      if (g !== lastGroup) { list.append(h('div', { class: 'grp' }, g)); lastGroup = g; }
      const b = h('button', { 'aria-pressed': String(t.name === selected), 'data-name': t.name },
        h('span', { class: 'tn' }, t.name.replace(/^model\./, '').replace(/^layers\.\d+\./, '')),
        h('span', { class: 'ts' }, t.shape.join(' × ')));
      b.addEventListener('click', () => { selected = t.name; renderBrowser(); });
      list.append(b);
    }
    el.append(list, detail(hd.tensors.find(t => t.name === selected) || hd.tensors[0]));
    const btn = list.querySelector('[aria-pressed="true"]');
    if (btn) list.scrollTop = btn.offsetTop - 60;
  }

  function role(name) {
    const tied = S.config?.tie_word_embeddings;
    if (name.includes('embed_tokens')) return `The vocabulary table. Row n is the vector for token ID n (chapter 3).${tied ? ' This model reuses it at the very end to score every possible next token (chapter 7).' : ''}`;
    if (name.includes('lm_head')) return 'Scores every vocabulary entry as the possible next token (chapter 7).';
    if (name.includes('q_proj')) return 'Turns each token’s vector into its query: “what am I looking for?” (chapter 5).';
    if (name.includes('k_proj')) return 'Turns each token’s vector into its key: “what do I contain?” It is smaller than the query matrix because several heads share each key (chapter 5).';
    if (name.includes('v_proj')) return 'Turns each token’s vector into its value: the information it passes on when another token pays attention to it (chapter 5).';
    if (name.includes('o_proj')) return 'Combines the results of all attention heads into one vector, which is added to the token’s running total (chapter 6).';
    if (name.includes('gate_proj')) return 'Feed-forward, step 1 of 3: stretches 576 numbers to 1,536. This one decides how much of each passes through (chapter 6).';
    if (name.includes('up_proj')) return 'Feed-forward, step 2 of 3: also stretches 576 numbers to 1,536. This one carries the content that the gate lets through (chapter 6).';
    if (name.includes('down_proj')) return 'Feed-forward, step 3 of 3: squeezes the 1,536 numbers back to 576 and adds them to the token’s running total (chapter 6).';
    return 'A list of scale factors, one per position in the vector. Before each block, the token’s vector is rescaled to a standard size and then multiplied by these.';
  }

  function detail(t) {
    const n = numel(t);
    const box = h('div', { class: 'tensor-detail' },
      h('h4', {}, t.name),
      h('p', { class: 'tensor-role' }, role(t.name)),
      h('p', { class: 'tensor-io' }, t.shape.length === 2
        ? `${fmtInt(t.shape[0])} rows × ${fmtInt(t.shape[1])} columns: takes in a vector of ${fmtInt(t.shape[1])} numbers and gives out ${fmtInt(t.shape[0])}. That's ${fmtInt(n)} numbers, ${fmtBytes(t.end - t.begin)} in the file.`
        : `${fmtInt(n)} numbers, ${fmtBytes(t.end - t.begin)} in the file.`));
    const p = S.previews?.find(x => x.name === t.name);
    if (!p) {
      box.append(h('p', { class: 'caption' }, S.loaded ? '' : 'Load the weights to see the actual numbers.'));
      return box;
    }
    // A few raw numbers, as they are
    const raw = h('div', {});
    box.append(raw);
    if (t.shape.length === 2) {
      ctx.engine.call('slice', { name: t.name, rows: 4, cols: 7 }).then(r => {
        const tbl = h('table', { class: 'raw-grid', 'aria-label': 'The first 4 rows and 7 columns of this matrix' },
          h('tbody', {}, [0, 1, 2, 3].map(i => h('tr', {}, Array.from(r.data.subarray(i * 7, i * 7 + 7), v => h('td', {}, v.toFixed(3)))))));
        raw.append(h('p', { class: 'caption', style: { margin: '6px 0 0' } }, 'The top-left corner, exactly as stored:'), tbl);
      }).catch(() => {});
    }
    if (p.grid) {
      const cv = h('canvas', { class: 'hm', 'aria-label': `Sampled values of ${t.name}` });
      // Canvas keeps its intrinsic aspect ratio: fix the height, let the width follow.
      cv.style.height = `${Math.min(220, Math.max(120, 220 * p.gr / p.gc))}px`;
      cv.style.width = 'auto';
      cv.style.maxWidth = '100%';
      drawGrid(cv, p.grid, p.gr, p.gc, diverging, 3 * p.std || 1);
      box.append(h('p', { class: 'caption', style: { margin: '10px 0 0' } }, `The whole matrix at a glance: ${p.gr} × ${p.gc} numbers picked evenly from it, one coloured square each.`), cv,
        h('div', { class: 'scale' }, h('span', {}, `−${(3 * p.std).toFixed(2)}`), h('span', { class: 'grad' }), h('span', {}, `+${(3 * p.std).toFixed(2)}`)));
    }
    box.append(h('ul', { class: 'notice-list' },
      t.shape.length === 2
        ? [
          h('li', {}, `The numbers are small: about two-thirds lie between −${p.std.toFixed(2)} and +${p.std.toFixed(2)}, and the largest is ${p.absmax.toFixed(2)}.`),
          h('li', {}, `They average out to about zero (${p.mean.toFixed(4)}), so positive and negative values balance.`),
          h('li', {}, 'Apart from faint stripes (some input positions carry bigger numbers throughout), there’s no picture to see. What the model learned is spread across all the numbers working together, not stored in any one place.'),
        ]
        : [h('li', {}, `Average ${p.mean.toFixed(3)}, largest ${p.absmax.toFixed(2)}. Each factor stretches or shrinks one of the ${fmtInt(n)} positions.`)]));
    return box;
  }

  function renderQuant() {
    const el = $('#quant-table');
    el.textContent = '';
    const d = ctx.dims();
    if (!d) return;
    const rows = [['FP32', 4], ['BF16 / FP16', 2], ['INT8', 1], ['INT4', 0.5]];
    const table = h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, h('th', {}, 'Precision'), h('th', {}, 'Bytes per weight'), h('th', {}, 'This model'), h('th', {}, '8B model'), h('th', {}, '70B model'), h('th', {}, '8B ceiling on one H100'))),
      h('tbody', {}, rows.map(([name, b]) => h('tr', { class: b === 2 ? 'hl' : '' },
        h('td', {}, name), h('td', {}, String(b)),
        h('td', {}, fmtBytes(d.params * b)),
        h('td', {}, fmtBytes(8e9 * b)),
        h('td', {}, fmtBytes(70e9 * b) + (70e9 * b > 80e9 ? ' (2+ GPUs)' : '')),
        h('td', {}, `${Math.round(3.35e12 / (8e9 * b))} tokens/s`)))));
    el.append(table, h('p', { class: 'caption' }, 'The ceiling is HBM bandwidth ÷ bytes of weights: the most decode steps per second one H100 can do for a single request if reading weights were the only cost. INT8 and INT4 formats also store small scale factors, a few percent extra.'));
  }

  function renderLoad(p) {
    const box = $('#load-progress');
    const btn = $('#btn-load');
    if (S.loaded) {
      btn.hidden = true;
      box.hidden = false;
      const i = S.loadInfo;
      box.querySelector('.bar span').style.width = '100%';
      box.querySelector('.progress-label').textContent =
        `${fmtBytes(i.fileBytes)} ${i.fromCache ? 'read from your browser cache' : 'downloaded'} in ${fmtMs(i.tDownload)}, converted to 32-bit floats in WebAssembly memory (${fmtBytes(i.deviceBytes)}) in ${fmtMs(i.tConvert)}, then one warm-up pass in ${fmtMs(i.tWarm)}. Real engines warm up too, compiling kernels and capturing CUDA graphs before taking traffic. Kernel: ${i.runtime === 'wasm-simd' ? 'WebAssembly SIMD' : 'plain JavaScript'}.`;
      return;
    }
    if (!p) return;
    box.hidden = false;
    const f = p.total ? p.loaded / p.total : 0;
    const label = p.phase === 'download'
      ? `Downloading ${fmtBytes(p.loaded)} of ${fmtBytes(p.total || S.meta.downloadMB * 1e6)}`
      : p.phase === 'cache' ? 'Reading from browser cache…'
        : p.phase === 'warmup' ? 'Warm-up pass through all layers…'
          : `Copying tensors into device memory: ${p.loaded} of ${p.total}`;
    box.querySelector('.bar span').style.width = `${f * 100}%`;
    box.querySelector('.progress-label').textContent = label;
  }

  $('#btn-load').addEventListener('click', () => ctx.load());
  $('#btn-load').dataset.action = 'load';

  return {
    update(reason, data) {
      if (reason === 'model') {
        $('#btn-load').textContent = `Load the weights (${S.meta.downloadMB} MB)`;
        $('#btn-load').hidden = false;
        $('#load-progress').hidden = true;
        renderFiles(); renderConfig(); renderSplit(); renderBrowser(); renderQuant();
      }
      if (reason === 'header') { renderFiles(); renderSplit(); renderBrowser(); }
      if (reason === 'weights') { renderBrowser(); renderLoad(); }
      if (reason === 'load-progress') renderLoad(data);
      if (reason === 'load-start') { $('#btn-load').disabled = true; renderLoad({ phase: 'download', loaded: 0, total: 0 }); }
      if (reason === 'load-error') { $('#btn-load').disabled = false; $('#load-progress .progress-label').textContent = `Loading failed: ${data.message}`; }
    },
  };
}
