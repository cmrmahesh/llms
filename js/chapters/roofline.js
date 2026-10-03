// 9. Arithmetic intensity and the roofline: why decode is memory-bound.

import { $, h, fmtInt, fmtBytes, fmtMs, fmtFlops } from '../util.js';

const NS = 'http://www.w3.org/2000/svg';
const GPU = { name: 'H100 SXM', bw: 3.35e12, flops: 989e12 }; // HBM3 bytes/s, dense BF16 FLOP/s

export default function roofline(ctx) {
  const { S } = ctx;
  let B = 1, T = 512, Nctx = 1024;
  let chart, readout, benchOut;

  // FLOPs and bytes for one forward pass of `tokens` new tokens per sequence, `seqs` sequences, context `n`.
  function cost(tokens, seqs, n) {
    const d = ctx.dims();
    const P = d.activeParams;
    const attnFlops = 4 * d.L * d.H * d.HD * tokens * n * seqs; // QK^T and PV
    const flops = 2 * P * tokens * seqs + attnFlops;
    const bytes = 2 * P + seqs * (n * d.kvBytesPerToken + tokens * d.kvBytesPerToken);
    const tCompute = flops / GPU.flops, tMem = bytes / GPU.bw;
    return { flops, bytes, I: flops / bytes, t: Math.max(tCompute, tMem), bound: tCompute > tMem ? 'compute' : 'memory', tCompute, tMem };
  }

  function build() {
    const el = $('#roofline');
    el.textContent = '';
    if (!S.config) return;
    chart = h('div', {});
    readout = h('div', {});
    const ctl = h('div', { class: 'ctl' });
    const logSlider = (label, min, max, get, set) => {
      const out = h('output', {}, fmtInt(get()));
      const inp = h('input', { type: 'range', min: Math.log2(min), max: Math.log2(max), step: 0.01, value: Math.log2(get()), 'aria-label': label });
      inp.addEventListener('input', () => { set(Math.round(2 ** +inp.value)); out.textContent = fmtInt(get()); draw(); });
      ctl.append(h('label', {}, label, ' ', out), inp);
    };
    logSlider('Prompt length for prefill (tokens)', 1, 8192, () => T, v => { T = v; });
    logSlider('Decode batch (requests decoded together)', 1, 512, () => B, v => { B = v; });
    logSlider('Context already in each KV cache', 16, 8192, () => Nctx, v => { Nctx = v; });
    ctl.append(readout);
    el.append(chart, ctl);
    draw();
  }

  function draw() {
    const d = ctx.dims();
    if (!d || !chart) return;
    const pre = cost(T, 1, T / 2);
    const dec = cost(1, B, Nctx);
    const W = 520, H = 360, pl = 56, pr = 26, pt = 14, pb = 44;
    const x0 = -1, x1 = 4, y0 = -1, y1 = 3.3; // log10 ranges
    const X = I => pl + (W - pl - pr) * (Math.log10(I) - x0) / (x1 - x0);
    const Y = tf => pt + (H - pt - pb) * (1 - (Math.log10(tf) - y0) / (y1 - y0));
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', `Roofline for ${GPU.name}: decode at intensity ${dec.I.toFixed(1)} is ${dec.bound}-bound, prefill at ${pre.I.toFixed(0)} is ${pre.bound}-bound`);
    const add = (tag, a, txt) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); if (txt !== undefined) e.textContent = txt; svg.append(e); return e; };
    for (let e = x0; e <= x1; e++) {
      add('line', { x1: X(10 ** e), x2: X(10 ** e), y1: pt, y2: H - pb, stroke: 'var(--rule-2)' });
      add('text', { x: X(10 ** e), y: H - pb + 16, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--ink-3)' }, String(10 ** e));
    }
    for (let e = y0; e <= 3; e++) {
      add('line', { x1: pl, x2: W - pr, y1: Y(10 ** e), y2: Y(10 ** e), stroke: 'var(--rule-2)' });
      add('text', { x: pl - 6, y: Y(10 ** e) + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--ink-3)' }, String(10 ** e));
    }
    add('text', { x: (pl + W - pr) / 2, y: H - 6, 'text-anchor': 'middle', 'font-size': 11.5, fill: 'var(--ink-2)' }, 'Arithmetic intensity (FLOPs per byte read from HBM)');
    add('text', { x: 14, y: (pt + H - pb) / 2, 'text-anchor': 'middle', 'font-size': 11.5, fill: 'var(--ink-2)', transform: `rotate(-90 14 ${(pt + H - pb) / 2})` }, 'Achievable TFLOP/s');
    // roof
    const ridge = GPU.flops / GPU.bw;
    const roof = I => Math.min(GPU.flops, GPU.bw * I) / 1e12;
    add('path', { d: `M${X(10 ** x0)} ${Y(roof(10 ** x0))} L${X(ridge)} ${Y(roof(ridge))} L${X(10 ** x1)} ${Y(roof(10 ** x1))}`, stroke: 'var(--ink)', 'stroke-width': 2.5, fill: 'none' });
    add('text', { x: X(1.2), y: Y(roof(1.2)) - 10, 'font-size': 11, fill: 'var(--hbm)', 'font-weight': 600, transform: `rotate(-33 ${X(1.2)} ${Y(roof(1.2)) - 10})` }, 'memory roof: 3.35 TB/s');
    add('text', { x: X(10 ** x1) - 4, y: Y(989) + 16, 'font-size': 11, fill: 'var(--sm)', 'font-weight': 600, 'text-anchor': 'end' }, 'compute roof: 989 TFLOP/s');
    add('line', { x1: X(ridge), x2: X(ridge), y1: Y(roof(ridge)), y2: H - pb, stroke: 'var(--ink-3)', 'stroke-dasharray': '3 3' });
    add('text', { x: X(ridge) + 4, y: H - pb - 6, 'font-size': 10.5, fill: 'var(--ink-3)' }, `ridge ≈ ${Math.round(ridge)}`);
    const pt2 = (I, label, color, dy) => {
      const xx = X(Math.max(10 ** x0, Math.min(10 ** x1, I))), yy = Y(roof(I));
      add('circle', { cx: xx, cy: yy, r: 7, fill: color, stroke: 'var(--panel)', 'stroke-width': 2 });
      const right = xx > W * 0.6;
      add('text', { x: right ? xx - 10 : xx + 10, y: yy + dy, 'font-size': 12, 'font-weight': 650, fill: color, 'text-anchor': right ? 'end' : 'start' }, label);
    };
    pt2(dec.I, `decode, batch ${B}`, 'var(--hbm)', 18);
    pt2(pre.I, `prefill, ${fmtInt(T)} tokens`, 'var(--sm)', pre.I > 200 ? 34 : -10);
    chart.textContent = '';
    chart.append(svg);

    const line = (c, what) => h('p', {}, h('b', {}, what), `: ${fmtFlops(c.flops)} over ${fmtBytes(c.bytes)} → `, h('b', {}, `${c.I < 10 ? c.I.toFixed(1) : fmtInt(c.I)} FLOP/B`), `, ${c.bound}-bound, ≥ ${fmtMs(c.t * 1000)} on one ${GPU.name}.`);
    readout.textContent = '';
    readout.append(
      line(pre, 'Prefill'),
      line(dec, 'Decode step'),
      h('p', { class: 'caption' }, `At batch ${B}, a decode step yields ${fmtInt(B)} token${B > 1 ? 's' : ''} in about ${fmtMs(dec.t * 1000)}: ${fmtInt(B / dec.t)} tokens/s in total. Raising the batch is nearly free until the point reaches the ridge, or the KV caches run out of memory.`));
  }

  function buildBench() {
    const el = $('#bench');
    el.textContent = '';
    const btn = h('button', { class: 'btn', disabled: !S.loaded }, 'Benchmark the kernel');
    benchOut = h('div', { class: 'bench-out' });
    btn.addEventListener('click', async () => {
      btn.disabled = true; btn.textContent = 'Measuring…';
      const r = await ctx.engine.call('bench');
      btn.disabled = false; btn.textContent = 'Measure again';
      benchOut.textContent = '';
      for (const [t, v] of Object.entries(r.res)) {
        benchOut.append(h('div', { class: 'bench-cell' },
          h('p', {}, `${t} token${t > 1 ? 's' : ''} per pass`),
          h('div', { class: 'v' }, `${v.gflops.toFixed(1)} GFLOP/s`),
          h('p', {}, `${fmtMs(v.ms / t)} per token per matrix`)));
      }
      const s1 = r.res[1].gflops, s32 = r.res[32].gflops;
      benchOut.append(h('p', { class: 'caption', style: { gridColumn: '1 / -1' } }, `Same matrix (${fmtInt(ctx.dims().I)} × ${fmtInt(ctx.dims().D)}), same code. Reusing each loaded weight across tokens made it ${(s32 / s1).toFixed(1)}× faster per token. That's prefill versus decode in miniature.`));
    });
    el.append(btn, benchOut, runCompare());
  }

  function runCompare() {
    const run = S.run;
    const dec = run?.steps.filter(s => s.kind === 'decode' && !s.noCache) || [];
    if (!run?.prefill || !dec.length) return h('p', { class: 'caption' }, 'After a run, this compares your prefill and decode speed per token.');
    const preTok = run.prefill.ms / Math.max(1, run.prefill.computed);
    const decTok = dec.reduce((a, s) => a + s.ms, 0) / dec.length;
    return h('p', { class: 'facts' },
      h('span', {}, 'Your last prefill ', h('b', {}, `${fmtMs(preTok)} per token`)),
      h('span', {}, 'Your decode ', h('b', {}, `${fmtMs(decTok)} per token`)),
      h('span', {}, 'Ratio ', h('b', {}, `${(decTok / preTok).toFixed(1)}×`)));
  }

  return {
    update(reason) {
      if (reason === 'model') {
        const d = ctx.dims();
        document.querySelectorAll('[data-bind="weightMB"]').forEach(e => { e.textContent = fmtBytes(d.activeParams * 2); });
        document.querySelectorAll('[data-bind="flopsPerTok"]').forEach(e => { e.textContent = fmtFlops(2 * d.activeParams); });
        build(); buildBench();
      }
      if (reason === 'weights') buildBench();
      if (reason === 'idle') { const el = $('#bench'); if (el.lastChild && benchOut) el.replaceChild(runCompare(), el.lastChild); }
    },
  };
}
