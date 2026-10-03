// 6. Feed-forward activations, residual stream growth, logit lens.

import { $, h, fmtInt, vis, diverging, rgb, pct } from '../util.js';
import { tokenAt } from './attention.js';

const NS = 'http://www.w3.org/2000/svg';

export default function ffn(ctx) {
  const { S } = ctx;
  let layer = 0;
  let lens = null;
  let ffnView = null;

  function buildFfn() {
    const el = $('#ffn');
    el.textContent = '';
    ffnView = null;
    const d = ctx.dims();
    const tr = S.run?.prefill?.trace;
    if (!tr) { el.append(h('p', { class: 'empty', style: { margin: 0 } }, S.loaded ? 'Generate to see the activations for your prompt.' : 'Load the weights and generate to see real activations.')); return; }
    layer = Math.min(layer, d.L - 1);
    const out = h('output', {}, String(layer));
    const inp = h('input', { type: 'range', min: 0, max: d.L - 1, value: layer, 'aria-label': 'Layer' });
    inp.addEventListener('input', () => { layer = +inp.value; out.textContent = inp.value; drawFfn(); });
    const cv = h('canvas', { 'aria-label': 'Feed-forward activations' });
    const stats = h('div', { class: 'facts' });
    el.append(h('div', { class: 'attn-controls', style: { marginTop: 0 } }, h('label', {}, 'Layer', inp, out)), cv, stats,
      h('p', { class: 'caption' }, `All ${fmtInt(d.I)} values of silu(gate) ⊙ up for the last prompt token. A few units fire hard and most stay near zero. Each unit is one row of W_down, so a large value writes that row into the residual stream.`));
    ffnView = { cv, stats };
    drawFfn();
  }

  function drawFfn() {
    if (!ffnView) return;
    const d = ctx.dims();
    const tr = S.run.prefill.trace;
    const v = tr.ffnAct.subarray(layer * d.I, (layer + 1) * d.I);
    const { cv, stats } = ffnView;
    const dpr = devicePixelRatio || 1;
    const W = cv.clientWidth || 800, H = 120;
    cv.width = W * dpr; cv.height = H * dpr;
    const g = cv.getContext('2d');
    g.scale(dpr, dpr);
    let max = 0;
    for (const x of v) max = Math.max(max, Math.abs(x));
    g.fillStyle = '#e2e7ed';
    g.fillRect(0, H / 2, W, 1);
    const bw = W / v.length;
    for (let i = 0; i < v.length; i++) {
      const f = v[i] / max;
      g.fillStyle = rgb(diverging(f));
      const hgt = Math.abs(f) * (H / 2 - 2);
      g.fillRect(i * bw, f > 0 ? H / 2 - hgt : H / 2, Math.max(1, bw), Math.max(0.5, hgt));
    }
    const sorted = Array.from(v, (x, i) => [Math.abs(x), i, x]).sort((a, b) => b[0] - a[0]);
    const small = v.reduce((a, x) => a + (Math.abs(x) < 0.1 * max ? 1 : 0), 0);
    stats.textContent = '';
    stats.append(
      h('span', {}, 'Largest |value| ', h('b', {}, max.toFixed(2))),
      h('span', {}, 'Below a tenth of that ', h('b', {}, `${fmtInt(small)} of ${fmtInt(v.length)}`)),
      h('span', {}, 'Strongest units ', h('b', {}, sorted.slice(0, 4).map(s => `#${s[1]} (${s[2].toFixed(1)})`).join(', '))));
  }

  function resid() {
    const el = $('#resid');
    el.textContent = '';
    const d = ctx.dims();
    const tr = S.run?.prefill?.trace;
    if (!tr) { el.append(h('p', { class: 'empty', style: { margin: 0 } }, 'Generate to see the residual stream grow.')); return; }
    const { T, startPos } = tr;
    const L = d.L;
    const W = 860, H = 260, pl = 50, pr = 120, pt = 14, pb = 34;
    let max = 0;
    for (const x of tr.resNorm) max = Math.max(max, x);
    const X = l => pl + (W - pl - pr) * l / L;
    const Y = v => pt + (H - pt - pb) * (1 - v / max);
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', `Length of each token's residual vector after each of the ${L} layers`);
    const add = (tag, a, txt) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); if (txt !== undefined) e.textContent = txt; svg.append(e); return e; };
    for (const f of [0, 0.25, 0.5, 0.75, 1]) {
      add('line', { x1: pl, x2: W - pr, y1: Y(max * f), y2: Y(max * f), stroke: 'var(--rule-2)' });
      add('text', { x: pl - 8, y: Y(max * f) + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--ink-3)' }, (max * f).toFixed(0));
    }
    for (let l = 0; l <= L; l += Math.ceil(L / 10)) add('text', { x: X(l), y: H - 12, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--ink-3)' }, String(l));
    add('text', { x: (pl + W - pr) / 2, y: H, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--ink-2)' }, 'after layer');
    // Find tokens with the biggest final norms to label (often the first token).
    for (let t = 0; t < T; t++) {
      let dd = '';
      for (let l = 0; l <= L; l++) dd += `${l ? 'L' : 'M'}${X(l).toFixed(1)} ${Y(tr.resNorm[l * T + t]).toFixed(1)}`;
      const last = t === T - 1;
      const p = add('path', { d: dd, fill: 'none', stroke: last ? 'var(--sm)' : 'var(--ink-3)', 'stroke-width': last ? 2.5 : 1, opacity: last ? 1 : 0.35 });
      const title = document.createElementNS(NS, 'title');
      title.textContent = `${vis(S.tok.tokenText(tokenAt(S, startPos + t)))} (position ${startPos + t})`;
      p.append(title);
      const end = tr.resNorm[L * T + t];
      if (last || end === Math.max(...Array.from({ length: T }, (_, i) => tr.resNorm[L * T + i]))) {
        add('text', { x: X(L) + 6, y: Y(end) + 4, 'font-size': 11, fill: last ? 'var(--sm)' : 'var(--ink-2)', 'font-family': 'IBM Plex Mono, monospace' }, vis(S.tok.tokenText(tokenAt(S, startPos + t))).slice(0, 14));
      }
    }
    el.append(svg, h('p', { class: 'caption' }, `One line per prompt token: the length (L2 norm) of its ${fmtInt(d.D)}-number vector after each layer. The highlighted line is the last token, the one that predicts what comes next.`));
  }

  function renderLens() {
    const el = $('#lens');
    el.textContent = '';
    if (!S.run?.prefill) { el.append(h('p', { class: 'empty', style: { margin: 0 } }, 'Generate first, then run the lens on your prompt.')); return; }
    const btn = h('button', { class: 'btn' }, lens ? 'Run again' : 'Run the logit lens');
    btn.addEventListener('click', async () => {
      btn.disabled = true; btn.textContent = 'Projecting 30 layers…';
      const r = await ctx.engine.call('lens');
      lens = r.layers;
      renderLens();
    });
    el.append(btn);
    if (!lens) {
      el.append(h('p', { class: 'caption' }, `Costs ${ctx.dims().L} extra LM-head multiplies (${fmtInt(ctx.dims().V)} × ${fmtInt(ctx.dims().D)} each).`));
      return;
    }
    const final = lens[lens.length - 1][0].id;
    const table = h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'After layer'), h('th', {}, 'Top guesses if the model stopped here'))));
    const tb = h('tbody');
    lens.forEach((top, l) => {
      tb.append(h('tr', {}, h('td', { class: 'l' }, String(l)),
        h('td', {}, top.slice(0, 4).map(c => h('span', {
          class: 'chip',
          style: { background: c.id === final ? 'var(--sm-fill)' : 'transparent', marginRight: '8px' },
        }, `${vis(S.tok.tokenText(c.id))} ${(c.p * 100).toFixed(0)}%`)))));
    });
    table.append(tb);
    el.append(h('div', { class: 'wrap' }, table), h('p', { class: 'caption' }, 'Highlighted: the token the full model actually ranks first.'));
  }

  return {
    update(reason) {
      if (reason === 'model') {
        const d = ctx.dims();
        const ffnW = 3 * d.D * d.I;
        const share = ffnW / d.perLayer;
        document.querySelectorAll('[data-bind="ffnShare"]').forEach(e => { e.textContent = pct(share, 0); });
        lens = null;
        buildFfn(); resid(); renderLens();
      }
      if (reason === 'weights' || reason === 'prefill') { lens = null; buildFfn(); resid(); renderLens(); }
    },
  };
}
