// 4. Word order via RoPE, shown as clock hands: each position turns the hand,
// and the angle between two hands depends only on how far apart they are.

import { $, h } from '../util.js';

const NS = 'http://www.w3.org/2000/svg';

export default function rope(ctx) {
  const { S } = ctx;
  let pair = 0, qPos = 12, kPos = 9, shift = 0;
  let q = null, k = null, fromModel = false;
  let dial = null, readout = null, shiftBtn = null, kInput = null, kOut = null;

  function dims() {
    const c = S.config;
    const HD = c.head_dim ?? c.hidden_size / c.num_attention_heads;
    return { HD, half: HD / 2, base: c.rope_theta ?? 10000 };
  }

  function exampleVectors(HD) {
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
  const deg = r => r * 180 / Math.PI;
  const wrap = a => { a = ((a % 360) + 360) % 360; return a > 180 ? a - 360 : a; };

  function svgEl(tag, attrs, parent, text) {
    const e = document.createElementNS(NS, tag);
    for (const [k2, v] of Object.entries(attrs)) e.setAttribute(k2, v);
    if (text !== undefined) e.textContent = text;
    parent?.append(e);
    return e;
  }

  function build() {
    const el = $('#rope');
    el.textContent = '';
    if (!S.config) return;
    const { half } = dims();
    dial = h('div', {});
    readout = h('div', { class: 'readout', 'aria-live': 'polite' });
    const ctl = h('div', { class: 'ctl' });
    const slider = (label, min, max, get, set, hint) => {
      const out = h('output', {}, String(get()));
      const inp = h('input', { type: 'range', min, max, value: get(), 'aria-label': label });
      inp.addEventListener('input', () => { set(+inp.value); out.textContent = String(get()); inp.value = get(); draw(); });
      ctl.append(h('label', {}, label, ' ', out, hint ? h('span', { class: 'dim' }, ` ${hint}`) : null), inp);
      return [inp, out];
    };
    slider('Position of the word asking (query)', 0, 300, () => qPos, v => { qPos = v; if (kPos > qPos) { kPos = qPos; syncK(); } });
    [kInput, kOut] = slider('Position of the earlier word (key)', 0, 300, () => kPos, v => { kPos = Math.min(v, qPos); });
    slider('Which pair of numbers', 0, half - 1, () => pair, v => { pair = v; }, '(0 turns fastest)');
    shiftBtn = h('button', { class: 'btn small', 'aria-pressed': 'false' }, 'Move both words 1,000 positions later');
    shiftBtn.addEventListener('click', () => { shift = shift ? 0 : 1000; shiftBtn.setAttribute('aria-pressed', String(!!shift)); draw(); });
    ctl.append(h('div', { style: { margin: '14px 0 4px' } }, shiftBtn), readout);
    el.append(dial, ctl);
    draw();
  }
  function syncK() { if (kInput) { kInput.value = kPos; kOut.textContent = String(kPos); } }

  function draw() {
    if (!dial) return;
    const { HD, half, base } = dims();
    if (!q) [q, k] = exampleVectors(HD);
    const omega = Math.pow(base, -2 * pair / HD); // radians per position for this pair
    const m = qPos + shift, n = kPos + shift;
    const aq0 = Math.atan2(q[pair + half], q[pair]), ak0 = Math.atan2(k[pair + half], k[pair]);
    const aq = aq0 + m * omega, ak = ak0 + n * omega;
    const between = wrap(deg(aq - ak));

    const svg = svgEl('svg', { viewBox: '-150 -150 300 320', role: 'img', 'aria-label': `Two clock hands; the angle between them is ${Math.round(Math.abs(between))} degrees` });
    svgEl('circle', { r: 120, fill: 'var(--panel-2)', stroke: 'var(--rule)' }, svg);
    for (let i = 0; i < 12; i++) {
      const a = i * Math.PI / 6;
      svgEl('line', { x1: 112 * Math.cos(a), y1: 112 * Math.sin(a), x2: 120 * Math.cos(a), y2: 120 * Math.sin(a), stroke: 'var(--rule)', 'stroke-width': 2 }, svg);
    }
    // arc between the hands
    const steps = 40;
    let d = '';
    const from = -ak, span = -(aq - ak);
    const spanW = ((span % (2 * Math.PI)) + 3 * Math.PI) % (2 * Math.PI) - Math.PI;
    for (let i = 0; i <= steps; i++) {
      const t = from + spanW * i / steps;
      d += `${i ? 'L' : 'M'}${(52 * Math.cos(t)).toFixed(1)} ${(52 * Math.sin(t)).toFixed(1)}`;
    }
    svgEl('path', { d, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2 }, svg);
    const hand = (a, color, len) => {
      const x = len * Math.cos(a), y = -len * Math.sin(a);
      svgEl('line', { x1: 0, y1: 0, x2: x, y2: y, stroke: color, 'stroke-width': 5, 'stroke-linecap': 'round' }, svg);
    };
    hand(ak, 'var(--host)', 92);
    hand(aq, 'var(--sm)', 104);
    svgEl('circle', { r: 6, fill: 'var(--ink)' }, svg);
    svgEl('text', { x: 0, y: 142, 'text-anchor': 'middle', 'font-size': 13, fill: 'var(--ink)', 'font-weight': 650 }, svg, `Angle between: ${Math.abs(between).toFixed(0)}°`);
    svgEl('text', { x: -60, y: 162, 'text-anchor': 'middle', 'font-size': 11.5, fill: 'var(--sm)', 'font-weight': 650 }, svg, `query, position ${m}`);
    svgEl('text', { x: 60, y: 162, 'text-anchor': 'middle', 'font-size': 11.5, fill: 'var(--host)', 'font-weight': 650 }, svg, `key, position ${n}`);
    dial.textContent = '';
    dial.append(svg);

    // scores using all pairs
    const qa = rotate(q, qPos, base, HD), ka = rotate(k, kPos, base, HD);
    const qb = rotate(q, qPos + 1000, base, HD), kb = rotate(k, kPos + 1000, base, HD);
    const turn = 360 * omega / (2 * Math.PI);
    readout.textContent = '';
    readout.append(
      h('p', {}, `This pair turns `, h('b', {}, `${turn < 0.1 ? turn.toFixed(4) : turn.toFixed(1)}° per position`),
        turn > 1 ? `, a full circle every ${(360 / turn).toFixed(1)} positions.` : ', so slowly that it only matters for words far apart.'),
      h('p', {}, `The two words are `, h('b', {}, `${qPos - kPos} position${qPos - kPos === 1 ? '' : 's'} apart`), `. Press the button: both hands spin to new places, but the angle between them stays the same.`),
      h('p', { class: 'score' }, 'How strongly the query matches the key, using all 32 pairs together: ',
        h('b', {}, dot(qa, ka).toFixed(3)), shift ? '' : h('span', { class: 'dim' }, ` (1,000 positions later: ${dot(qb, kb).toFixed(3)}, the same)`)),
      h('p', { class: 'caption' }, fromModel
        ? 'Real vectors: the query and key from the model’s first layer, first attention head, for the last token of your prompt.'
        : 'Example vectors for now. After you generate, these become the model’s real query and key.'));
  }

  return {
    update(reason) {
      if (reason === 'model') { q = k = null; fromModel = false; build(); }
      if (reason === 'prefill') {
        const tr = S.run?.prefill?.trace;
        if (tr?.qkv) { q = tr.qkv.qPre; k = tr.qkv.kPre; fromModel = true; draw(); }
      }
    },
  };
}
