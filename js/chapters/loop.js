// 5. The transformer loop: attention (tokens talk) → add → feed-forward (each
// token alone) → add, repeated for every layer. A sketch, not real data: the
// tracer (chapter 9) replays the real numbers.

import { $, h, diverging, rgb } from '../util.js';

const NS = 'http://www.w3.org/2000/svg';
const WORDS = ['The', ' cat', ' sat', ' on', ' the'];
const PHASES = [
  { title: 'step 1, attention: tokens share information', text: 'Each token looks back at the tokens before it and pulls in what it needs. These are the only arrows between tokens in the whole layer. In this sketch, the arcs show who pulls from whom; chapter 6 shows the real pattern.' },
  { title: 'step 2, add: each running vector gets a correction', text: 'What attention gathered is added to each token’s running vector. Nothing is replaced: the vector keeps everything it had and gains context from its neighbours.' },
  { title: 'step 3, feed-forward: each token thinks on its own', text: 'Now every token goes through the same small network, separately. There are no arrows between tokens here: each one works only with what it already holds, applying patterns stored in this layer’s weights.' },
  { title: 'step 4, add again, then on to the next layer', text: 'The feed-forward result is added too. The updated vectors go into the next layer, which repeats exactly the same two halves with its own weights.' },
];

// Deterministic pseudo-random numbers so the sketch is the same every time.
function rng(seed) { let s = seed; return () => ((s = (s * 16807) % 2147483647) / 2147483647) * 2 - 1; }

export default function loop(ctx) {
  const L = () => ctx.dims()?.L ?? 30;
  const n = WORDS.length, dims = 8;
  let step = 0;          // phase index across all layers: layer = floor(step/4), phase = step % 4; final = 4L
  let timer = null;
  let vec = null;
  let els = null;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  function reset() {
    const r = rng(11);
    vec = Array.from({ length: n }, () => Array.from({ length: dims }, () => r() * 0.6));
    step = -1;
  }

  // Schematic attention weights for a layer: who pulls from whom.
  function weights(l) {
    const r = rng(101 + l * 7);
    const w = [];
    for (let j = 0; j < n; j++) {
      const row = Array(n).fill(0);
      for (let i = 0; i < j; i++) row[i] = Math.max(0, r()) * 0.5;
      if (j > 0) {
        if (l % 3 === 0) row[j - 1] += 1.2;           // previous-word heads
        if (l % 3 === 1) row[1] += j >= 2 ? 1.2 : 0;  // focus on the noun
        if (l % 3 === 2 && j === n - 1) { row[1] += 0.9; row[2] += 0.9; }
      }
      const s = row.reduce((a, b) => a + b, 0) || 1;
      w.push(row.map(x => x / s));
    }
    return w;
  }

  function applyAttention(l) {
    const w = weights(l), r = rng(500 + l);
    const next = vec.map(v => v.slice());
    for (let j = 1; j < n; j++) for (let k = 0; k < dims; k++) {
      let mix = 0;
      for (let i = 0; i < j; i++) mix += w[j][i] * vec[i][(k + l) % dims];
      next[j][k] += 0.35 * mix + 0.05 * r();
    }
    vec = next;
    recentre();
  }
  function applyFfn(l) {
    const r = rng(900 + l);
    for (let j = 0; j < n; j++) for (let k = 0; k < dims; k++) {
      const x = vec[j][k];
      vec[j][k] += 0.25 * Math.tanh(1.5 * x + r()) + 0.2 * r();
    }
    recentre();
  }
  // Keep each sketch vector centred with a steady spread, so the colours stay
  // readable and tokens stay distinct (real vectors do too).
  function recentre() {
    for (const v of vec) {
      const m = v.reduce((a, b) => a + b, 0) / v.length;
      const sd = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length) || 1;
      for (let k = 0; k < v.length; k++) v[k] = (v[k] - m) / sd * 0.5;
    }
  }

  const cx = i => 190 + i * 142;

  function build() {
    const el = $('#loop-viz');
    el.textContent = '';
    reset();
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 860 400');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Five tokens passing through one transformer layer after another: attention connects the tokens, the feed-forward network processes each token separately');
    const add = (tag, a, parent = svg, txt) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); if (txt !== undefined) e.textContent = txt; parent.append(e); return e; };
    const defs = add('defs', {});
    defs.innerHTML = '<marker id="loop-ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--sm)"/></marker>';
    // tower of layers
    const tower = [];
    const Lc = L();
    const barH = 300 / Lc;
    for (let l = 0; l < Lc; l++) tower.push(add('rect', { x: 26, y: 362 - (l + 1) * barH, width: 40, height: Math.max(2, barH - 2), rx: 1.5, fill: 'var(--rule-2)' }));
    add('text', { x: 46, y: 382, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--ink-3)' }, svg, 'layer 0');
    add('text', { x: 46, y: 52, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--ink-3)' }, svg, `layer ${Lc - 1}`);
    // phase title
    const title = add('text', { x: 110, y: 30, 'font-size': 16, 'font-weight': 700, fill: 'var(--ink)' }, svg, 'Press Next or Autoplay');
    // groups
    const arcs = add('g', {});
    const ffn = add('g', {});
    const chips = [];
    for (let i = 0; i < n; i++) {
      const g = add('g', {});
      const cells = [];
      for (let k = 0; k < dims; k++) cells.push(add('rect', { x: cx(i) - 52 + k * 13, y: 312, width: 12, height: 18, rx: 2, fill: '#eee' }, g));
      add('text', { x: cx(i), y: 356, 'text-anchor': 'middle', 'font-size': 15, 'font-weight': 650, fill: 'var(--ink)', 'font-family': 'IBM Plex Mono, monospace' }, g, WORDS[i].replace(' ', '␣'));
      const plus = add('text', { x: cx(i) + 60, y: 327, 'font-size': 18, 'font-weight': 800, fill: 'var(--sm)', opacity: 0 }, g, '+');
      chips.push({ cells, plus });
    }
    add('text', { x: 110, y: 392, 'font-size': 11, fill: 'var(--ink-3)' }, svg, 'each token’s running vector (a few of its 576 numbers)');
    // final box
    const final = add('g', { opacity: 0 });
    add('path', { d: `M${cx(n - 1)} 308 C ${cx(n - 1)} 200, ${cx(n - 1) - 40} 150, ${cx(n - 1) - 90} 128`, fill: 'none', stroke: 'var(--sm)', 'stroke-width': 2, 'marker-end': 'url(#loop-ah)' }, final);
    add('rect', { x: cx(n - 1) - 330, y: 92, width: 240, height: 50, rx: 6, fill: 'var(--sm-fill)', stroke: 'var(--sm)' }, final);
    add('text', { x: cx(n - 1) - 210, y: 114, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 700, fill: 'var(--ink)' }, final, 'Last token’s vector → scores');
    add('text', { x: cx(n - 1) - 210, y: 132, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--ink-2)' }, final, 'for all 49,152 tokens → next word');

    const back = h('button', { class: 'btn' }, '◀ Back');
    const next = h('button', { class: 'btn primary' }, 'Next ▶');
    const play = h('button', { class: 'btn' }, 'Autoplay');
    const speed = h('select', { class: 'inline', 'aria-label': 'Autoplay speed' },
      h('option', { value: 'slow' }, 'Slow'), h('option', { value: 'normal', selected: true }, 'Normal'), h('option', { value: 'fast' }, 'Fast'));
    const toEnd = h('button', { class: 'btn ghost' }, 'Skip to the end');
    const resetBtn = h('button', { class: 'btn ghost' }, 'Reset');
    const status = h('span', { class: 'loop-status' });
    const text = h('p', { class: 'loop-text', 'aria-live': 'polite' }, 'Five tokens, their running vectors drawn as coloured cells, and the tower of 30 layers on the left. Step through with Next, or press Autoplay. You can also click a layer in the tower, or use the ← and → keys.');
    back.addEventListener('click', () => { stop(); setStep(step - 1); });
    next.addEventListener('click', () => { stop(); advance(); });
    play.addEventListener('click', () => (timer ? stop() : start()));
    toEnd.addEventListener('click', () => { stop(); setStep(4 * L()); });
    resetBtn.addEventListener('click', () => { stop(); reset(); draw(); text.textContent = 'Back to the start. Press Next or Autoplay.'; });
    tower.forEach((r, l) => {
      r.style.cursor = 'pointer';
      r.addEventListener('click', () => { stop(); setStep(4 * l); });
    });
    el.tabIndex = 0;
    el.setAttribute('aria-label', 'Transformer loop animation. Left and right arrow keys step back and forward.');
    el.addEventListener('keydown', e => {
      if (e.target !== el && e.target.tagName !== 'svg') return;
      if (e.key === 'ArrowRight') { e.preventDefault(); stop(); advance(); }
      if (e.key === 'ArrowLeft') { e.preventDefault(); stop(); setStep(step - 1); }
    });
    el.append(svg, h('div', { class: 'loop-ctl' }, back, next, h('span', { class: 'loop-sep' }), play, h('label', { class: 'num' }, 'Speed ', speed), toEnd, resetBtn), h('p', { class: 'loop-status' }, status), text);
    els = { svg, tower, title, arcs, ffn, chips, final, play, back, next, speed, status, text, add };
    draw();
  }

  // Jump to any step by replaying the (deterministic) sketch from the start.
  function setStep(target) {
    target = Math.max(-1, Math.min(4 * L(), target));
    reset();
    while (step < target) advance(true);
    draw();
  }

  function start() {
    if (step >= 4 * L()) reset();
    els.play.textContent = 'Pause';
    tick();
  }
  function stop() {
    clearTimeout(timer); timer = null;
    if (els) els.play.textContent = 'Autoplay';
  }
  function tick() {
    advance();
    if (step >= 4 * L()) { stop(); return; }
    const l = Math.floor(step / 4);
    // Slow for the first two layers and the last one, fast in between.
    const slow = l < 2 || l >= L() - 1;
    const f = { slow: 1.8, normal: 1, fast: 0.45 }[els.speed.value];
    timer = setTimeout(tick, (reduced ? 1400 : slow ? 1300 : 160) * f);
  }

  function advance(quiet = false) {
    if (step >= 4 * L()) return;
    step++;
    if (step < 4 * L()) {
      const l = Math.floor(step / 4), p = step % 4;
      if (p === 1) applyAttention(l);
      if (p === 3) applyFfn(l);
    }
    if (!quiet) draw();
  }

  function draw() {
    if (!els) return;
    const { tower, title, arcs, ffn, chips, final, status, text, add, back, next } = els;
    back.disabled = step < 0;
    next.disabled = step >= 4 * L();
    const Lc = L();
    const done = step >= 4 * Lc;
    const l = done ? Lc - 1 : Math.max(0, Math.floor(step / 4));
    const p = step < 0 ? -1 : step % 4;
    tower.forEach((r, i) => r.setAttribute('fill', step < 0 ? 'var(--rule-2)' : i < l || done ? '#7f95b8' : i === l ? 'var(--sm)' : 'var(--rule-2)'));
    // chips
    chips.forEach((c, i) => {
      // Each vector on its own colour scale, like RMSNorm rescales it before every block.
      let max = 0.3;
      for (const x of vec[i]) max = Math.max(max, Math.abs(x));
      c.cells.forEach((cell, k) => cell.setAttribute('fill', rgb(diverging(vec[i][k] / max))));
      c.plus.setAttribute('opacity', (p === 1 || p === 3) ? 1 : 0);
    });
    // attention arcs
    arcs.textContent = '';
    if (p === 0 && !done) {
      const w = weights(l);
      for (let j = 1; j < chips.length; j++) for (let i = 0; i < j; i++) {
        if (w[j][i] < 0.12) continue;
        const x1 = cx(i), x2 = cx(j), hgt = 50 + 26 * (j - i);
        add('path', { d: `M${x1} 306 Q ${(x1 + x2) / 2} ${306 - hgt}, ${x2 - 6} 307`, fill: 'none', stroke: 'var(--sm)', 'stroke-width': 1 + 4 * w[j][i], opacity: 0.25 + 0.75 * w[j][i], 'marker-end': 'url(#loop-ah)' }, arcs);
      }
    }
    // feed-forward boxes
    ffn.textContent = '';
    if (p === 2) {
      chips.forEach((_, i) => {
        add('rect', { x: cx(i) - 60, y: 214, width: 120, height: 50, rx: 6, fill: 'var(--panel)', stroke: '#2f4570', 'stroke-width': 1.5 }, ffn);
        add('text', { x: cx(i), y: 236, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 700, fill: '#2f4570' }, ffn, 'feed-forward');
        add('text', { x: cx(i), y: 252, 'text-anchor': 'middle', 'font-size': 10.5, fill: 'var(--ink-3)' }, ffn, '576 → 1,536 → 576');
        add('path', { d: `M${cx(i) - 14} 306 V 266`, stroke: '#2f4570', 'stroke-width': 1.5, fill: 'none' }, ffn);
        add('path', { d: `M${cx(i) + 14} 266 V 306`, stroke: '#2f4570', 'stroke-width': 1.5, fill: 'none' }, ffn);
      });
    }
    final.setAttribute('opacity', done ? 1 : 0);
    if (step < 0) { title.textContent = 'Press Next or Autoplay'; status.textContent = 'Not started'; return; }
    if (done) {
      title.textContent = `After all ${Lc} layers`;
      status.textContent = `${Lc} of ${Lc} layers done`;
      text.textContent = `Every token went through ${Lc} rounds of talk-then-think. Only the last token’s vector is used to predict the next word: it is turned into a score for every vocabulary entry (chapter 8). Then the chosen word is added to the end and the whole stack runs again for it, one token per pass (chapter 10).`;
      return;
    }
    title.textContent = `Layer ${l}: ${PHASES[p].title}`;
    status.textContent = `Layer ${l} of ${Lc - 1}, step ${p + 1} of 4`;
    text.textContent = l >= 2 && l < Lc - 1 && timer && els.speed.value !== 'slow' ? `Layer ${l}… the same two halves, again and again, each layer with its own weights.` : PHASES[p].text;
  }

  build();

  return {
    update(reason) {
      if (reason === 'model') { stop(); build(); }
    },
  };
}
