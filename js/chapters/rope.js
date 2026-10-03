// 4. Position via RoPE: rotate query/key pairs by position × frequency.

import { $, h } from '../util.js';

const NS = 'http://www.w3.org/2000/svg';

export default function rope(ctx) {
  const { S } = ctx;
  let pair = 0, pos = 7, dist = 3;
  let q = null, k = null, fromModel = false;

  function dims() {
    const c = S.config;
    const HD = c.head_dim ?? c.hidden_size / c.num_attention_heads;
    return { HD, half: HD / 2, base: c.rope_theta ?? 10000 };
  }

  function exampleVectors(HD) {
    // Deterministic pseudo-random vectors when the model hasn't run yet.
    let s = 7;
    const r = () => ((s = (s * 16807) % 2147483647) / 2147483647) * 2 - 1;
    return [Float32Array.from({ length: HD }, r), Float32Array.from({ length: HD }, r)];
  }

  function rotate(v, p, base, HD) {
    const half = HD / 2, out = new Float32Array(HD);
    for (let i = 0; i < half; i++) {
      const a = p * Math.pow(base, -2 * i / HD);
      const c = Math.cos(a), s = Math.sin(a);
      out[i] = v[i] * c - v[i + half] * s;
      out[i + half] = v[i + half] * c + v[i] * s;
    }
    return out;
  }
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

  function svgEl(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k2, v] of Object.entries(attrs)) e.setAttribute(k2, v);
    parent?.append(e);
    return e;
  }

  let dial = null, readout = null;

  function build() {
    const el = $('#rope');
    el.textContent = '';
    if (!S.config) return;
    const { half } = dims();
    dial = h('div', {});
    readout = h('div', { class: 'readout' });
    const ctl = h('div', { class: 'ctl' });
    const slider = (label, min, max, val, on) => {
      const out = h('output', {}, String(val));
      const inp = h('input', { type: 'range', min, max, value: val, 'aria-label': label });
      inp.addEventListener('input', () => { out.textContent = inp.value; on(+inp.value); draw(); });
      ctl.append(h('label', {}, label, ' ', out), inp);
    };
    slider('Pair', 0, half - 1, pair, v => { pair = v; });
    slider('Position m', 0, 512, pos, v => { pos = v; });
    slider('Distance m − n', 0, 64, dist, v => { dist = v; });
    ctl.append(readout);
    el.append(dial, ctl);
    draw();
  }

  function draw() {
    if (!dial) return;
    const { HD, half, base } = dims();
    if (!q) [q, k] = exampleVectors(HD);
    const theta = Math.pow(base, -2 * pair / HD);
    const ang = pos * theta;

    // --- dial ---
    const svg = svgEl('svg', { viewBox: '-150 -150 300 300', role: 'img', 'aria-label': `Pair ${pair} of the query vector rotated by ${ang.toFixed(2)} radians` });
    svgEl('circle', { r: 120, fill: 'none', stroke: 'var(--rule)' }, svg);
    svgEl('line', { x1: -140, x2: 140, y1: 0, y2: 0, stroke: 'var(--rule-2)' }, svg);
    svgEl('line', { y1: -140, y2: 140, x1: 0, x2: 0, stroke: 'var(--rule-2)' }, svg);
    const x0 = q[pair], y0 = q[pair + half];
    const r0 = Math.hypot(x0, y0) || 1;
    const sc = 110 / r0;
    const a0 = Math.atan2(y0, x0);
    const x1 = r0 * Math.cos(a0 + ang), y1 = r0 * Math.sin(a0 + ang);
    // arc from original to rotated (mod 2π, drawn up to one turn)
    const sweep = ang % (2 * Math.PI);
    const steps = 48;
    let d = '';
    for (let i = 0; i <= steps; i++) {
      const t = a0 + sweep * i / steps;
      d += `${i ? 'L' : 'M'}${(70 * Math.cos(t)).toFixed(1)} ${(-70 * Math.sin(t)).toFixed(1)}`;
    }
    svgEl('path', { d, fill: 'none', stroke: 'var(--sm)', 'stroke-width': 1.5, 'stroke-dasharray': '3 3' }, svg);
    svgEl('line', { x1: 0, y1: 0, x2: x0 * sc, y2: -y0 * sc, stroke: 'var(--ink-3)', 'stroke-width': 2 }, svg);
    svgEl('circle', { cx: x0 * sc, cy: -y0 * sc, r: 6, fill: 'var(--panel)', stroke: 'var(--ink-3)', 'stroke-width': 2 }, svg);
    svgEl('line', { x1: 0, y1: 0, x2: x1 * sc, y2: -y1 * sc, stroke: 'var(--sm)', 'stroke-width': 2.5 }, svg);
    svgEl('circle', { cx: x1 * sc, cy: -y1 * sc, r: 6.5, fill: 'var(--sm)' }, svg);
    const t1 = svgEl('text', { x: -144, y: -132, 'font-size': 11, fill: 'var(--ink-3)' }, svg);
    t1.textContent = `dims ${pair} and ${pair + half}`;
    const t2 = svgEl('text', { x: -144, y: 144, 'font-size': 11, fill: 'var(--ink-3)' }, svg);
    t2.textContent = 'hollow: before · filled: after';

    // --- readout ---
    const turn = 2 * Math.PI / theta;
    const m = pos, n = Math.max(0, pos - dist);
    const qa = rotate(q, m, base, HD), ka = rotate(k, n, base, HD);
    const qb = rotate(q, m + 1000, base, HD), kb = rotate(k, n + 1000, base, HD);
    const qc = rotate(q, m, base, HD), kc = rotate(k, Math.max(0, m - dist - 20), base, HD);
    readout.textContent = '';
    readout.append(
      h('p', {}, `Pair ${pair} turns `, h('b', {}, `${theta.toPrecision(3)} rad`), ' per position, one full turn every ', h('b', {}, `${turn < 1e5 ? turn.toFixed(turn < 10 ? 1 : 0) : turn.toExponential(1)} positions`), '.'),
      h('p', {}, `At position ${m} it has rotated `, h('b', {}, `${ang.toFixed(2)} rad`), '.'),
      h('p', { style: { marginTop: '14px' } }, `Query at ${m} · key at ${n}: `, h('b', {}, dot(qa, ka).toFixed(4))),
      h('p', {}, `Query at ${m + 1000} · key at ${n + 1000}: `, h('b', {}, dot(qb, kb).toFixed(4)), h('span', { class: 'dim' }, '  same distance, same score')),
      h('p', {}, `Query at ${m} · key at ${Math.max(0, m - dist - 20)}: `, h('b', {}, dot(qc, kc).toFixed(4)), h('span', { class: 'dim' }, '  further away, different score')),
      h('p', { class: 'caption' }, fromModel
        ? `Real vectors: query and key of head 0, layer 0, for the last token of your prompt, before rotation.`
        : 'Example vectors. Run the model to use the real query and key from layer 0.'));
    dial.textContent = '';
    dial.append(svg);
  }

  return {
    update(reason) {
      if (reason === 'model') { q = k = null; fromModel = false; build(); }
      if (reason === 'prefill') {
        const tr = S.run?.prefill?.trace;
        if (tr?.qkv) { q = tr.qkv.qPre; k = tr.qkv.kPre; fromModel = true; pos = Math.min(512, tr.qkv.pos); build(); }
      }
    },
  };
}
