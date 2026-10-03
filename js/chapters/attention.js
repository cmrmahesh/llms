// 5. Attention: layer diagram, real attention maps, per-op timings.

import { $, h, fmtInt, fmtMs, vis, sequential, rgb, tip, esc } from '../util.js';

const NS = 'http://www.w3.org/2000/svg';

export function tokenAt(S, p) {
  const run = S.run;
  if (!run) return S.ids[p];
  if (p < run.ids.length) return run.ids[p];
  return run.steps[p - run.ids.length]?.token;
}

export default function attention(ctx) {
  const { S } = ctx;
  let layer = 0, head = 0, query = -1, source = 'prompt';
  let view = null;

  // ---------------- layer diagram
  function diagram() {
    const el = $('#layer-diagram');
    el.textContent = '';
    const d = ctx.dims();
    if (!d) return;
    const T = S.ids.length || 'T';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 960 250');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'One transformer layer: normalise, project to Q K V, rotate, attend, project out and add; then normalise, gate and up projections, SiLU, down projection and add.');
    const add = (tag, a, txt) => {
      const e = document.createElementNS(NS, tag);
      for (const [k, v] of Object.entries(a)) e.setAttribute(k, v);
      if (txt !== undefined) e.textContent = txt;
      svg.append(e);
      return e;
    };
    const box = (x, y, w, label, sub, kind) => {
      const stroke = kind === 'w' ? 'var(--hbm)' : kind === 'op' ? 'var(--sm)' : 'var(--ink-3)';
      const fill = kind === 'w' ? 'var(--hbm-fill)' : kind === 'op' ? 'var(--sm-fill)' : 'var(--panel-2)';
      add('rect', { x, y, width: w, height: 46, rx: 5, fill, stroke, 'stroke-width': 1.2 });
      add('text', { x: x + w / 2, y: y + 19, 'text-anchor': 'middle', 'font-size': 12.5, 'font-weight': 650, fill: 'var(--ink)' }, label);
      if (sub) add('text', { x: x + w / 2, y: y + 35, 'text-anchor': 'middle', 'font-size': 10.5, fill: 'var(--ink-2)', 'font-family': 'IBM Plex Mono, monospace' }, sub);
    };
    const arrow = (x1, y1, x2, y2) => add('path', { d: `M${x1} ${y1} L${x2} ${y2}`, stroke: 'var(--ink-3)', 'stroke-width': 1.3, fill: 'none', 'marker-end': 'url(#ah)' });
    const defs = document.createElementNS(NS, 'defs');
    defs.innerHTML = '<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="var(--ink-3)"/></marker>';
    svg.append(defs);
    const kvw = d.KVH * d.HD;
    // Row 1: attention
    add('text', { x: 4, y: 16, 'font-size': 12, 'font-weight': 700, fill: 'var(--ink-2)' }, 'Attention block');
    box(4, 26, 92, 'x', `${T}×${d.D}`, 'act');
    arrow(96, 49, 112, 49);
    box(112, 26, 84, 'RMSNorm', '', 'op');
    arrow(196, 49, 212, 49);
    box(212, 26, 150, 'Q = x·Wqᵀ', `Wq ${d.D}×${d.D}`, 'w');
    box(372, 26, 150, 'K, V = x·Wk,vᵀ', `${kvw}×${d.D} each`, 'w');
    arrow(522, 49, 538, 49);
    box(538, 26, 110, 'RoPE on Q, K', `${d.H} + ${d.KVH} heads`, 'op');
    arrow(648, 49, 664, 49);
    box(664, 26, 150, 'softmax(QKᵀ/√d)·V', `per head, d = ${d.HD}`, 'op');
    arrow(814, 49, 830, 49);
    box(830, 26, 126, 'out = ·Woᵀ, x +=', `Wo ${d.D}×${d.D}`, 'w');
    // KV cache write
    add('path', { d: 'M447 72 V96 H600', stroke: 'var(--hbm)', 'stroke-width': 1.3, fill: 'none', 'stroke-dasharray': '4 3', 'marker-end': 'url(#ah)' });
    add('text', { x: 606, y: 100, 'font-size': 11, fill: 'var(--hbm)', 'font-weight': 600 }, 'K, V appended to the KV cache in HBM');
    // Row 2: FFN
    add('text', { x: 4, y: 146, 'font-size': 12, 'font-weight': 700, fill: 'var(--ink-2)' }, 'Feed-forward block');
    box(4, 156, 92, 'x', `${T}×${d.D}`, 'act');
    arrow(96, 179, 112, 179);
    box(112, 156, 84, 'RMSNorm', '', 'op');
    arrow(196, 179, 212, 179);
    box(212, 156, 170, 'gate, up = x·Wᵀ', `${d.I}×${d.D} each`, 'w');
    arrow(382, 179, 398, 179);
    box(398, 156, 150, 'silu(gate) ⊙ up', `${T}×${d.I}`, 'op');
    arrow(548, 179, 564, 179);
    box(564, 156, 170, 'down, x +=', `Wdown ${d.D}×${d.I}`, 'w');
    arrow(734, 179, 750, 179);
    box(750, 156, 206, `→ next layer (${d.L} total)`, `${fmtInt(d.perLayer)} weights per layer`, 'act');
    // legend
    add('rect', { x: 4, y: 222, width: 12, height: 12, rx: 2, fill: 'var(--hbm-fill)', stroke: 'var(--hbm)' });
    add('text', { x: 22, y: 232, 'font-size': 11, fill: 'var(--ink-2)' }, 'reads weights from HBM, matmul on tensor cores');
    add('rect', { x: 320, y: 222, width: 12, height: 12, rx: 2, fill: 'var(--sm-fill)', stroke: 'var(--sm)' });
    add('text', { x: 338, y: 232, 'font-size': 11, fill: 'var(--ink-2)' }, 'element-wise or attention math, no weights');
    el.append(svg);
  }

  // ---------------- controls
  function controls() {
    const el = $('#attn-controls');
    el.textContent = '';
    const d = ctx.dims();
    if (!d) return;
    const mk = (label, max, get, set) => {
      const out = h('output', {}, String(get()));
      const inp = h('input', { type: 'range', min: 0, max, value: get(), 'aria-label': label });
      inp.addEventListener('input', () => { set(+inp.value); out.textContent = inp.value; draw(); });
      return h('label', {}, label, inp, out);
    };
    const src = h('select', { class: 'inline', 'aria-label': 'Which pass to show' },
      h('option', { value: 'prompt' }, 'Prefill (the prompt)'),
      h('option', { value: 'latest' }, 'Latest decode step'));
    src.value = source;
    src.addEventListener('change', () => { source = src.value; query = -1; draw(); });
    el.append(mk('Layer', d.L - 1, () => layer, v => { layer = v; }), mk('Head', d.H - 1, () => head, v => { head = v; }), h('label', {}, 'Show ', src));
    view = true;
  }

  function currentTrace() {
    const run = S.run;
    if (!run?.prefill) return null;
    if (source === 'latest') {
      const last = [...run.steps].reverse().find(s => s.trace && s.kind === 'decode');
      return last ? last.trace : null;
    }
    return run.prefill.trace;
  }

  function draw() {
    const mapEl = $('#attn-map'), rowEl = $('#attn-row');
    mapEl.textContent = ''; rowEl.textContent = '';
    const d = ctx.dims();
    const tr = currentTrace();
    if (!tr) {
      mapEl.append(h('p', { class: 'empty' }, S.loaded
        ? (source === 'latest' ? 'Generate at least two tokens to see a decode step.' : 'Press Generate to fill this in with your prompt’s attention.')
        : 'Load the weights and generate to see real attention patterns.'));
      return;
    }
    const { T, N, startPos, fullAttn } = tr;
    const H = d.H;
    const rowsHere = fullAttn ? T : 1;
    const rowOf = t => tr.attn.subarray(((layer * H + head) * rowsHere + t) * N, ((layer * H + head) * rowsHere + t) * N + N);
    if (query < 0 || query >= rowsHere) query = rowsHere - 1;

    // --- heatmap
    if (fullAttn && T > 1) {
      const labelW = T <= 64 ? 92 : 0;
      const cell = Math.max(2, Math.min(16, Math.floor(560 / N)));
      const W = labelW + cell * N, Hh = cell * T + (T <= 64 ? 4 : 0);
      const dpr = devicePixelRatio || 1;
      const cv = h('canvas', { 'aria-label': `Attention weights for layer ${layer}, head ${head}: ${T} query rows by ${N} key columns` });
      cv.width = W * dpr; cv.height = Hh * dpr;
      cv.style.width = `${W}px`; cv.style.maxWidth = '100%';
      const g = cv.getContext('2d');
      g.scale(dpr, dpr);
      for (let t = 0; t < T; t++) {
        const row = rowOf(t);
        const pos = startPos + t;
        for (let j = 0; j < N; j++) {
          if (j > pos) { g.fillStyle = '#fafbfc'; }
          else g.fillStyle = rgb(sequential(Math.sqrt(row[j])));
          g.fillRect(labelW + j * cell, t * cell, cell, cell);
        }
        if (labelW) {
          g.fillStyle = t === query ? getComputedStyle(document.body).getPropertyValue('--sm') : '#47566e';
          g.font = `${Math.min(11, cell)}px "IBM Plex Mono", monospace`;
          g.textBaseline = 'middle';
          g.textAlign = 'right';
          g.fillText(vis(S.tok.tokenText(tokenAt(S, pos))).slice(0, 12), labelW - 6, t * cell + cell / 2);
        }
      }
      // selected row outline
      g.strokeStyle = getComputedStyle(document.body).getPropertyValue('--sm');
      g.lineWidth = 2;
      g.strokeRect(labelW + 1, query * cell + 1, N * cell - 2, cell - 2);
      const pick = e => {
        const r = cv.getBoundingClientRect();
        const sx = W / r.width;
        const x = (e.clientX - r.left) * sx - labelW, y = (e.clientY - r.top) * sx;
        const t = Math.floor(y / cell), j = Math.floor(x / cell);
        return { t, j, ok: t >= 0 && t < T && j >= 0 && j < N };
      };
      cv.addEventListener('mousemove', e => {
        const { t, j, ok } = pick(e);
        if (!ok || j > startPos + t) return tip(e, null);
        const p = rowOf(t)[j];
        tip(e, `<code>${esc(vis(S.tok.tokenText(tokenAt(S, startPos + t))))}</code> → <code>${esc(vis(S.tok.tokenText(tokenAt(S, j))))}</code><br>${(p * 100).toFixed(1)}% of its attention`);
      });
      cv.addEventListener('mouseleave', e => tip(e, null));
      cv.addEventListener('click', e => { const { t, ok } = pick(e); if (ok) { query = t; draw(); } });
      mapEl.append(h('h4', {}, `Layer ${layer}, head ${head} (shares K/V head ${Math.floor(head / (d.H / d.KVH))})`), cv,
        h('div', { class: 'scale' }, h('span', {}, '0'), h('span', { class: 'grad seq' }), h('span', {}, '1 (square-root scale)')),
        h('p', { class: 'caption' }, `Each row is a query token, each column a key. The empty upper-right triangle is the causal mask: no token sees the future.${startPos ? ` The first ${startPos} positions came from the prefix cache, so only ${T} rows were computed.` : ''} Click a row.`));
    } else {
      mapEl.append(h('h4', {}, `Layer ${layer}, head ${head}`),
        h('p', { class: 'caption' }, source === 'latest'
          ? `In a decode step there is exactly one query: the new token. It attends to all ${N} positions in the KV cache, which is the row on the right.`
          : `The full ${T} × ${N} map is too large to keep, so only the last query row is shown.`));
    }

    // --- one query row as bars
    const row = rowOf(query);
    const qpos = startPos + (fullAttn ? query : T - 1);
    const list = h('div', { class: 'arow-list' });
    let maxJ = 0;
    for (let j = 0; j <= qpos; j++) if (row[j] > row[maxJ]) maxJ = j;
    for (let j = 0; j <= qpos; j++) {
      list.append(h('div', { class: 'arow' },
        h('span', { class: 's' }, vis(S.tok.tokenText(tokenAt(S, j)))),
        h('span', { class: 'b' }, h('span', { style: { width: `${row[j] * 100}%` } })),
        h('span', { class: 'p' }, `${(row[j] * 100).toFixed(1)}%`)));
    }
    rowEl.append(h('h4', {}, 'Query ', h('code', {}, vis(S.tok.tokenText(tokenAt(S, qpos)))), ` at position ${qpos}`),
      h('p', { class: 'caption', style: { marginTop: 0 } }, `Strongest pull: `, h('code', {}, vis(S.tok.tokenText(tokenAt(S, maxJ)))), ` (${(row[maxJ] * 100).toFixed(0)}%). ${maxJ === 0 && row[0] > 0.5 ? 'Many heads park most of their weight on the first token when nothing else is relevant. That pattern is called an attention sink.' : ''}`),
      list);
  }

  function opTimes() {
    const el = $('#op-times');
    el.textContent = '';
    const tr = S.run?.prefill?.trace;
    if (!tr) { el.append(h('p', { class: 'empty' }, 'Timings appear after a run.')); return; }
    const op = tr.opMs;
    const parts = [
      ['qkv', 'Q, K, V projections', '#c76ba7'],
      ['attn', 'Attention scores and mixing', '#f0b7d6'],
      ['oproj', 'Output projection', '#d68fbd'],
      ['ffn', 'Feed-forward', '#a8126f'],
      ['head', 'LM head', '#7a0d50'],
      ['norm', 'Norms', '#cfd6df'],
      ['rope', 'RoPE + KV write', '#e2e7ed'],
      ['embed', 'Embedding lookup', '#f0f2f5'],
    ];
    const total = parts.reduce((a, [k]) => a + op[k], 0);
    const bar = h('div', { class: 'stack', role: 'img', 'aria-label': parts.map(([k, l]) => `${l} ${fmtMs(op[k])}`).join(', ') });
    for (const [k, , c] of parts) {
      const f = op[k] / total;
      const dark = ['#a8126f', '#7a0d50', '#c76ba7'].includes(c);
      bar.append(h('div', { style: { width: `${f * 100}%`, background: c, color: dark ? '#fff' : 'var(--ink)' }, title: `${k}: ${fmtMs(op[k])}` }, f > 0.09 ? `${Math.round(f * 100)}%` : ''));
    }
    el.append(bar, h('div', { class: 'split-legend' }, parts.map(([k, l, c]) => h('span', {}, h('i', { style: { background: c } }), `${l} ${fmtMs(op[k])}`))),
      h('p', { class: 'caption' }, `${tr.T} tokens through ${ctx.dims().L} layers in ${fmtMs(tr.totalMs)}. With a prompt this short the weight matmuls dominate. Attention's share grows with the square of the context length.`));
  }

  return {
    update(reason) {
      if (reason === 'model') {
        const d = ctx.dims();
        for (const [k, v] of Object.entries({ L: d.L, H: d.H, KVH: d.KVH, I: d.I, V: d.V })) {
          document.querySelectorAll(`[data-bind="${k}"]`).forEach(e => { e.textContent = fmtInt(v); });
        }
        layer = Math.min(layer, d.L - 1); head = Math.min(head, d.H - 1);
        controls(); diagram(); draw(); opTimes();
      }
      if (reason === 'tokens') diagram();
      if (reason === 'weights') draw();
      if (reason === 'prefill') { query = -1; draw(); opTimes(); }
      if (reason === 'step' && source === 'latest' && view) draw();
    },
  };
}
