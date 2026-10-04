// "Follow the shape": one token's vector through embedding → one layer → ×30 →
// unembedding → softmax, drawn as matrix × vector = vector, with a ledger of
// every weight matrix (in the spirit of 3Blue1Brown's GPT-3 tally).

import { $, h, fmtInt, vis, drawGrid, diverging, rms } from '../util.js';

const PX = 0.26;        // pixels per row/column when drawing shapes to scale
const CAP = 260;        // tallest/widest drawing; longer shapes get a "break" mark

export default function journey(ctx) {
  const { S } = ctx;
  let step = 0, timer = null, ui = null;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------------- data: real if a run exists, otherwise stand-ins
  function data() {
    const d = ctx.dims();
    const tr = S.run?.prefill?.trace;
    const tk = tr?.tok;
    const fake = (n, seed) => {
      let s = seed;
      return Float32Array.from({ length: n }, () => { s = (s * 16807) % 2147483647; return (s / 2147483647) * 2 - 1; });
    };
    const { D, I, KVH, HD, V, L } = d;
    const kv = KVH * HD;
    if (!tk) {
      return {
        real: false, tokenId: 3575, chosen: null, N: 12,
        x: fake(D, 3), h: fake(D, 5), q: fake(D, 7), k: fake(kv, 9), v: fake(kv, 11), mix: fake(D, 13), o: fake(D, 15),
        xmid: fake(D, 17), h2: fake(D, 19), gate: fake(I, 21), up: fake(I, 23), act: fake(I, 25), down: fake(D, 27),
        x1: fake(D, 29), xL: fake(D, 31), final: fake(D, 33), scores: fake(V, 35), probs: null,
      };
    }
    const at = (a, l, n) => a.subarray(l * n, l * n + n);
    const step0 = S.run.steps[0];
    return {
      real: true, tokenId: S.run.ids[S.run.ids.length - 1], chosen: step0?.token ?? null, N: tr.N,
      x: at(tk.x, 0, D), h: at(tk.h, 0, D), q: at(tk.q, 0, D), k: at(tk.k, 0, kv), v: at(tk.v, 0, kv),
      mix: at(tk.att, 0, D), o: at(tk.o, 0, D), xmid: at(tk.xmid, 0, D), h2: at(tk.h2, 0, D),
      gate: at(tk.gate, 0, I), up: at(tk.up, 0, I), act: at(tr.ffnAct, 0, I), down: at(tk.down, 0, D),
      x1: L > 1 ? at(tk.x, 1, D) : at(tr.hidden, 0, D), xL: at(tr.hidden, L - 1, D), final: tk.final,
      scores: step0?.logits ?? fake(V, 35), probs: null,
    };
  }

  // ---------------- the steps
  function steps() {
    const d = ctx.dims();
    const { D, I, H, KVH, HD, V, L } = d;
    const kv = KVH * HD;
    const n = fmtInt;
    const vec = (key, len, label, segs) => ({ t: 'vec', key, len, label, segs });
    const mat = (rows, cols, label, preview) => ({ t: 'mat', rows, cols, label, preview });
    const op = text => ({ t: 'op', text });
    const p0 = 'model.layers.0.';
    return [
      { title: 'Look up the embedding', ledger: 'emb', shape: `${n(D)} × 1`,
        text: `The token’s ID selects one row of the ${n(V)} × ${n(D)} embedding table. That row, ${n(D)} numbers, is the token’s vector. From now on we draw it standing up, as a column: ${n(D)} × 1.`,
        items: [{ t: 'tok' }, op('selects a row of'), mat(V, D, 'embedding table', 'model.embed_tokens.weight'), op('→'), vec('x', D, 'x')] },
      { title: 'Layer 0 · normalize', ledger: 'norm', shape: `${n(D)} × 1`,
        text: `Rescale the vector to a standard size and multiply each number by a learned factor (${n(D)} weights). The shape doesn’t change.`,
        items: [vec('x', D, 'x'), op('normalize →'), vec('h', D, 'h')] },
      { title: 'Layer 0 · query', ledger: 'q', shape: `${n(D)} × 1 = ${H} heads × ${HD}`,
        text: `Multiply by the query matrix: ${n(H * HD)} × ${n(D)} times ${n(D)} × 1 gives ${n(H * HD)} × 1. Those ${n(H * HD)} numbers are ${H} heads of ${HD}, one “search” per head.`,
        items: [mat(H * HD, D, 'Wq', p0 + 'self_attn.q_proj.weight'), op('×'), vec('h', D, 'h'), op('='), vec('q', H * HD, 'q', H)] },
      { title: 'Layer 0 · key', ledger: 'k', shape: `${n(kv)} × 1 = ${KVH} heads × ${HD}`,
        text: `The key matrix is shorter: ${n(kv)} × ${n(D)}, so the key comes out as ${n(kv)} × 1, just ${KVH} heads, each shared by ${H / KVH} query heads. It is saved into the KV cache.`,
        items: [mat(kv, D, 'Wk', p0 + 'self_attn.k_proj.weight'), op('×'), vec('h', D, 'h'), op('='), vec('k', kv, 'k', KVH)] },
      { title: 'Layer 0 · value', ledger: 'v', shape: `${n(kv)} × 1 = ${KVH} heads × ${HD}`,
        text: `Same shape for the value: ${n(kv)} × 1, also saved into the KV cache. That’s all the weights attention needs for this token.`,
        items: [mat(kv, D, 'Wv', p0 + 'self_attn.v_proj.weight'), op('×'), vec('h', D, 'h'), op('='), vec('v', kv, 'v', KVH)] },
      { title: 'Layer 0 · attention', ledger: 'attn', shape: `${n(D)} × 1 = ${H} heads × ${HD}`,
        text: `No weight matrix here. Each of the ${H} query heads is compared with the keys of all __N__ tokens so far (from the KV cache). The scores become percentages, and the head takes that mix of their values: ${HD} numbers per head, ${n(H * HD)} in all.`,
        items: [vec('q', H * HD, 'q', H), op('compare with the cached keys of __N__ tokens, mix their values →'), vec('mix', H * HD, 'mixed', H)] },
      { title: 'Layer 0 · output', ledger: 'o', shape: `${n(D)} × 1`,
        text: `The output matrix blends the heads’ results back into one ${n(D)}-number correction: ${n(D)} × ${n(H * HD)} times ${n(H * HD)} × 1.`,
        items: [mat(D, H * HD, 'Wo', p0 + 'self_attn.o_proj.weight'), op('×'), vec('mix', H * HD, 'mixed', H), op('='), vec('o', D, 'correction')] },
      { title: 'Layer 0 · add', ledger: null, shape: `${n(D)} × 1`,
        text: 'The correction is added to the running vector, number by number. Same shape in, same shape out: this is the residual stream.',
        items: [vec('x', D, 'x'), op('+'), vec('o', D, 'correction'), op('='), vec('xmid', D, 'x')] },
      { title: 'Layer 0 · normalize again', ledger: 'norm', shape: `${n(D)} × 1`,
        text: `Rescale once more before the feed-forward half (another ${n(D)} weights).`,
        items: [vec('xmid', D, 'x'), op('normalize →'), vec('h2', D, 'h')] },
      { title: 'Layer 0 · gate (up-projection)', ledger: 'gate', shape: `${n(I)} × 1`,
        text: `The feed-forward half widens the vector: ${n(I)} × ${n(D)} times ${n(D)} × 1 gives ${n(I)} × 1. The drawing is cut off because it is too tall to show to scale.`,
        items: [mat(I, D, 'Wgate', p0 + 'mlp.gate_proj.weight'), op('×'), vec('h2', D, 'h'), op('='), vec('gate', I, 'gate')] },
      { title: 'Layer 0 · up-projection', ledger: 'up', shape: `${n(I)} × 1`,
        text: `A second ${n(I)} × ${n(D)} matrix gives another ${n(I)} numbers.`,
        items: [mat(I, D, 'Wup', p0 + 'mlp.up_proj.weight'), op('×'), vec('h2', D, 'h'), op('='), vec('up', I, 'up')] },
      { title: 'Layer 0 · combine', ledger: null, shape: `${n(I)} × 1`,
        text: 'No weights: silu(gate) × up, number by number. The gate decides how much of each “up” number gets through. Most end up near zero.',
        items: [vec('gate', I, 'gate'), op('silu( ) ×'), vec('up', I, 'up'), op('='), vec('act', I, 'combined')] },
      { title: 'Layer 0 · down-projection', ledger: 'down', shape: `${n(D)} × 1`,
        text: `Back down: ${n(D)} × ${n(I)} times ${n(I)} × 1 gives ${n(D)} × 1, the feed-forward’s correction.`,
        items: [mat(D, I, 'Wdown', p0 + 'mlp.down_proj.weight'), op('×'), vec('act', I, 'combined'), op('='), vec('down', D, 'correction')] },
      { title: 'Layer 0 · add', ledger: null, shape: `${n(D)} × 1`,
        text: `Add it to the running vector. That’s one whole layer: the vector went in as ${n(D)} × 1 and comes out as ${n(D)} × 1, now carrying information from the other tokens.`,
        items: [vec('xmid', D, 'x'), op('+'), vec('down', D, 'correction'), op('='), vec('x1', D, 'x')] },
      { title: `Layers 1 to ${L - 1} · the same again`, ledger: 'layers', shape: `${n(D)} × 1`,
        text: `Every other layer repeats exactly those steps, each with its own weights, and the vector stays ${n(D)} × 1 throughout. In the table, every per-layer row now counts × ${L}.`,
        items: [vec('x1', D, 'after layer 0'), op(`${L - 1} more layers →`), vec('xL', D, `after layer ${L - 1}`)] },
      { title: 'Unembedding: a score for every token', ledger: 'unemb', shape: `${n(V)} × 1`,
        text: `After a final normalize, multiply by the vocabulary table: ${n(V)} × ${n(D)} times ${n(D)} × 1 gives ${n(V)} × 1, one score per vocabulary entry. ${d.tied ? 'This model reuses the embedding table here instead of storing a second one.' : ''}`,
        items: [mat(V, D, d.tied ? 'embedding table again' : 'LM head', 'model.embed_tokens.weight'), op('×'), vec('final', D, 'x'), op('='), vec('scores', V, 'scores')] },
      { title: 'Softmax and pick', ledger: 'all', shape: '1 token',
        text: `Softmax turns the ${n(V)} scores into percentages that add up to 100%, and the sampler picks one. Compare the two columns: the scores vary everywhere, but after softmax nearly all the chance sits in a few spots. The ${n(D)}-number vector has become a single token ID, and the whole trip starts again for that token.`,
        items: [vec('scores', V, 'scores'), op('softmax →'), vec('probs', V, 'chances'), op('pick →'), { t: 'tok', chosen: true }] },
    ];
  }

  // ---------------- the ledger (which matrices, how many weights)
  function ledgerRows() {
    const { D, I, H, KVH, HD, V, L, tied } = ctx.dims();
    const kv = KVH * HD;
    return [
      { key: 'emb', name: 'Embedding', dims: [['n_vocab', V], ['d_embed', D]], per: V * D, times: 1 },
      { key: 'q', name: 'Query', dims: [['n_heads·d_head', H * HD], ['d_embed', D]], per: H * HD * D, times: L, layer: true },
      { key: 'k', name: 'Key', dims: [['n_kv·d_head', kv], ['d_embed', D]], per: kv * D, times: L, layer: true },
      { key: 'v', name: 'Value', dims: [['n_kv·d_head', kv], ['d_embed', D]], per: kv * D, times: L, layer: true },
      { key: 'attn', name: 'Attention scores', text: 'no weights: uses the KV cache', per: 0, times: L, layer: true },
      { key: 'o', name: 'Output', dims: [['d_embed', D], ['n_heads·d_head', H * HD]], per: D * H * HD, times: L, layer: true },
      { key: 'gate', name: 'Gate', dims: [['d_ff', I], ['d_embed', D]], per: I * D, times: L, layer: true },
      { key: 'up', name: 'Up-projection', dims: [['d_ff', I], ['d_embed', D]], per: I * D, times: L, layer: true },
      { key: 'down', name: 'Down-projection', dims: [['d_embed', D], ['d_ff', I]], per: D * I, times: L, layer: true },
      { key: 'norm', name: 'Norm scales', dims: [['d_embed', D]], per: D, times: 2 * L + 1, layer: true },
      { key: 'unemb', name: 'Unembedding', dims: [['n_vocab', V], ['d_embed', D]], per: tied ? 0 : V * D, shown: V * D, times: 1, note: tied ? '0 extra (shared)' : '' },
    ];
  }

  // ---------------- drawing helpers
  const sz = n => Math.min(CAP, Math.max(6, n * PX));
  const broken = n => n * PX > CAP;

  function drawVec(arr, len, segs, peak = false) {
    const cells = Math.min(len, len > 2000 ? 96 : 64);
    const per = len / cells;
    const col = new Float32Array(cells);
    for (let c = 0; c < cells; c++) {
      let best = 0;
      for (let i = Math.floor(c * per); i < Math.floor((c + 1) * per); i++) if (Math.abs(arr[i]) > Math.abs(best)) best = arr[i];
      col[c] = best;
    }
    const cv = h('canvas', { class: 'jv-canvas', 'aria-hidden': 'true' });
    let scale = 2.5 * rms(arr) || 1;
    if (peak) { scale = 0; for (const v of col) scale = Math.max(scale, Math.abs(v)); scale = scale || 1; }
    drawGrid(cv, col, cells, 1, diverging, scale);
    cv.style.height = `${sz(len)}px`;
    const wrap = h('div', { class: `jv-vec${broken(len) ? ' broken' : ''}` }, cv);
    if (segs) for (let s = 1; s < segs; s++) wrap.append(h('i', { class: 'jv-seg', style: { top: `${(s / segs) * 100}%` } }));
    return wrap;
  }

  function drawMat(rows, cols, preview) {
    const p = S.previews?.find(x => x.name === preview);
    const box = h('div', { class: `jv-mat${broken(rows) ? ' broken-v' : ''}${broken(cols) ? ' broken-h' : ''}`, style: { width: `${sz(cols)}px`, height: `${sz(rows)}px` } });
    if (p?.grid) {
      const cv = h('canvas', { class: 'jv-canvas', 'aria-hidden': 'true' });
      drawGrid(cv, p.grid, p.gr, p.gc, diverging, 3 * p.std || 1);
      box.append(cv);
    }
    return box;
  }

  // ---------------- build + render
  function build() {
    const el = $('#journey');
    el.textContent = '';
    if (!ctx.dims()) return;
    const back = h('button', { class: 'btn' }, '◀ Back');
    const next = h('button', { class: 'btn primary' }, 'Next ▶');
    const play = h('button', { class: 'btn' }, 'Autoplay');
    back.addEventListener('click', () => { stopPlay(); go(step - 1); });
    next.addEventListener('click', () => { stopPlay(); go(step + 1); });
    play.addEventListener('click', () => (timer ? stopPlay() : startPlay()));
    const dots = h('div', { class: 'jv-dots', role: 'group', 'aria-label': 'Jump to a step' });
    steps().forEach((st, i) => {
      const b = h('button', { class: 'jv-dot', title: st.title, 'aria-label': `Step ${i + 1}: ${st.title}` });
      b.addEventListener('click', () => { stopPlay(); go(i); });
      dots.append(b);
    });
    const shapeNow = h('div', { class: 'jv-now' });
    const title = h('h4', { class: 'tv-title' });
    const text = h('p', { class: 'tv-plain' });
    const stage = h('div', { class: 'jv-stage', 'aria-live': 'polite' });
    const note = h('p', { class: 'caption jv-note' });
    const ledger = h('div', { class: 'jv-ledger' });
    el.tabIndex = 0;
    el.setAttribute('aria-label', 'Shape walkthrough. Left and right arrow keys step back and forward.');
    el.addEventListener('keydown', e => {
      if (e.target !== el) return;
      if (e.key === 'ArrowRight') { e.preventDefault(); stopPlay(); go(step + 1); }
      if (e.key === 'ArrowLeft') { e.preventDefault(); stopPlay(); go(step - 1); }
    });
    el.append(
      h('div', { class: 'tv-top' }, h('div', { class: 'tv-nav' }, back, next), play, dots),
      h('div', { class: 'jv-head' }, h('div', {}, title, text), shapeNow),
      stage, note, ledger);
    ui = { back, next, play, dots, shapeNow, title, text, stage, note, ledger };
    render();
  }

  function go(i) {
    const n = steps().length;
    step = Math.max(0, Math.min(n - 1, i));
    render();
  }
  function startPlay() {
    if (step >= steps().length - 1) step = -1;
    ui.play.textContent = 'Pause';
    const tick = () => {
      go(step + 1);
      if (step >= steps().length - 1) { stopPlay(); return; }
      timer = setTimeout(tick, reduced ? 4000 : 3200);
    };
    tick();
  }
  function stopPlay() { clearTimeout(timer); timer = null; if (ui) ui.play.textContent = 'Autoplay'; }

  function render() {
    if (!ui) return;
    const all = steps();
    const st = all[step];
    const dat = data();
    const fill = s => s.replace(/__N__/g, fmtInt(dat.N));
    ui.back.disabled = step === 0;
    ui.next.disabled = step === all.length - 1;
    [...ui.dots.children].forEach((b, i) => { b.classList.toggle('on', i === step); b.classList.toggle('done', i < step); });
    ui.title.textContent = `Step ${step + 1} of ${all.length}: ${st.title}`;
    ui.text.textContent = fill(st.text);
    ui.shapeNow.textContent = '';
    ui.shapeNow.append(h('small', {}, 'the token right now'), h('b', {}, st.shape));

    // the picture
    ui.stage.textContent = '';
    const row = h('div', { class: `jv-row${reduced ? '' : ' enter'}` });
    let prevOp = '';
    for (const it of st.items) {
      if (it.t === 'op') { prevOp = it.text; row.append(h('span', { class: 'jv-op' }, fill(it.text))); continue; }
      if (it.t === 'tok') {
        const id = it.chosen ? dat.chosen : dat.tokenId;
        row.append(h('div', { class: 'jv-item' },
          h('div', { class: 'jv-tok' }, h('small', {}, 'token ID'), h('b', {}, id === null ? '?' : String(id))),
          h('span', { class: 'jv-label' }, h('code', {}, id === null ? '…' : vis(S.tok.tokenText(id)))),
          h('span', { class: 'jv-dim' }, '1 number')));
        continue;
      }
      if (it.t === 'mat') {
        row.append(h('div', { class: 'jv-item' }, drawMat(it.rows, it.cols, it.preview),
          h('span', { class: 'jv-label' }, h('code', {}, it.label)),
          h('span', { class: 'jv-dim' }, h('span', { class: 'dm' }, fmtInt(it.rows)), ' × ', h('span', { class: 'dn' }, fmtInt(it.cols)))));
        continue;
      }
      let arr = dat[it.key];
      if (it.key === 'scores') {
        // Show how far each score stands above the typical score.
        const sorted = Float32Array.from(arr).sort();
        const med = sorted[sorted.length >> 1];
        arr = Float32Array.from(arr, v => Math.max(0, v - med));
      }
      if (it.key === 'probs') {
        const sc = dat.scores;
        let m = -Infinity; for (const v of sc) if (v > m) m = v;
        arr = Float32Array.from(sc, v => Math.exp(v - m));
      }
      row.append(h('div', { class: 'jv-item' }, drawVec(arr, it.len, it.segs, it.key === 'scores' || it.key === 'probs'),
        h('span', { class: 'jv-label' }, h('code', {}, it.label)),
        h('span', { class: 'jv-dim' }, h('span', { class: prevOp === '×' ? 'dn' : 'dm' }, fmtInt(it.len)), ' × 1')));
      prevOp = '';
    }
    ui.stage.append(row);
    ui.note.textContent = dat.real
      ? 'Real numbers: your prompt’s last token in layer 0, and the model’s real weights (a sample of each matrix). Very tall or wide shapes are cut off, shown by the jagged edge.'
      : `Stand-in numbers until you generate${S.previews ? '' : ' (and grey matrices until the weights are loaded)'}. The shapes are already exact. Very tall or wide shapes are cut off, shown by the jagged edge.`;
    renderLedger(st.ledger);
  }

  function renderLedger(current) {
    const rows = ledgerRows();
    const { L } = ctx.dims();
    const stepsSoFar = steps().slice(0, step + 1).map(s => s.ledger).filter(Boolean);
    const seen = new Set(stepsSoFar);
    const layersDone = seen.has('layers') || seen.has('unemb') || seen.has('all');
    const total = rows.reduce((a, r) => a + r.per * r.times, 0);
    const matrices = rows.filter(r => r.dims && r.dims.length === 2 && r.per > 0).reduce((a, r) => a + r.times, 0);
    // Norm vectors met so far: 1 or 2 inside layer 0, then all 2L, plus the final one.
    const normsSeen = seen.has('unemb') || seen.has('all') ? 2 * L + 1 : layersDone ? 2 * L : stepsSoFar.filter(k => k === 'norm').length;
    const counted = rows.reduce((a, r) => {
      if (r.key === 'norm') return a + r.per * normsSeen;
      if (!(seen.has(r.key) || (r.layer && layersDone) || seen.has('all'))) return a;
      return a + r.per * (r.layer && !layersDone ? 1 : r.times);
    }, 0);
    const t = ui.ledger;
    t.textContent = '';
    t.append(h('div', { class: 'jl-total' },
      h('div', {}, h('span', {}, 'Weights passed so far: '), h('b', {}, fmtInt(counted))),
      h('div', {}, h('span', {}, 'Whole model: '), h('b', {}, fmtInt(total)), h('span', { class: 'dim' }, ` in ${fmtInt(matrices)} matrices plus ${2 * L + 1} small norm vectors`))));
    const table = h('div', { class: 'jl-table' });
    for (const r of rows) {
      const isCur = current === r.key || (current === 'layers' && r.layer);
      const visited = seen.has(r.key) || (r.layer && layersDone) || seen.has('all');
      const formula = r.dims
        ? r.dims.map(([name, v], i) => [i ? h('span', { class: 'jl-x' }, '×') : null, h('span', { class: 'jl-dim' }, h('sup', {}, fmtInt(v)), name)]).flat()
        : [h('span', { class: 'dim' }, r.text)];
      const times = r.times === 1 ? 'once' : `× ${r.times}${r.key === 'norm' ? '' : ' layers'}`;
      table.append(h('div', { class: `jl-row${isCur ? ' cur' : ''}${visited ? ' seen' : ''}` },
        h('span', { class: 'jl-name' }, r.name),
        h('span', { class: 'jl-formula' }, ...formula, r.dims ? h('span', { class: 'jl-x' }, '=') : null, r.dims ? h('b', {}, fmtInt(r.shown ?? r.per)) : null),
        h('span', { class: `jl-times${current === 'layers' && r.layer ? ' hot' : ''}` }, r.per || r.shown ? times : ''),
        h('span', { class: 'jl-sum' }, r.per ? fmtInt(r.per * r.times) : (r.note || '—'))));
    }
    t.append(table);
  }

  return {
    update(reason) {
      if (reason === 'model') { stopPlay(); step = 0; build(); }
      if (reason === 'prefill' || reason === 'weights') render();
    },
  };
}
