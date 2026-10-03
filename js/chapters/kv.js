// 8. Decode loop and KV cache: a live grid of cached K/V, sizes, and step timings.

import { $, h, fmtInt, fmtBytes, fmtMs, vis, drawVector, tip, esc } from '../util.js';
import { tokenAt } from './attention.js';

const NS = 'http://www.w3.org/2000/svg';
const KNOWN = [
  // name, layers, kv heads, head dim (public configs)
  ['Llama 3.1 8B', 32, 8, 128],
  ['Llama 3.1 70B', 80, 8, 128],
];

export default function kv(ctx) {
  const { S } = ctx;
  let built = false;
  let gridWrap, facts, vecs, steps, toggle;

  function build() {
    const el = $('#kv');
    el.textContent = '';
    gridWrap = h('div', { class: 'kv-scroll' });
    facts = h('div', {});
    vecs = h('div', {});
    el.append(
      gridWrap,
      h('div', { class: 'kv-legend' },
        h('span', {}, h('i', { style: { background: '#b9822a' } }), 'reused from the prefix cache'),
        h('span', {}, h('i', { style: { background: 'var(--hbm-fill)' } }), 'written during prefill'),
        h('span', {}, h('i', { style: { background: 'var(--sm-fill)' } }), 'appended by decode steps'),
        h('span', {}, h('i', { style: { background: 'var(--panel)', outline: '2px solid var(--sm)' } }), 'latest')),
      facts, vecs);
    const st = $('#steps');
    st.textContent = '';
    toggle = h('input', { type: 'checkbox', checked: S.settings.kvCache });
    toggle.addEventListener('change', () => { S.settings.kvCache = toggle.checked; });
    steps = h('div', {});
    st.append(h('div', { class: 'steps-ctl' },
      h('label', { class: 'toggle' }, toggle, ' Use the KV cache'),
      h('span', { class: 'dim' }, 'Switch it off, then press Generate or Continue: new steps recompute every K and V from scratch.')), steps);
    built = true;
  }

  function drawGrid() {
    gridWrap.textContent = '';
    const d = ctx.dims();
    const run = S.run;
    if (!run?.prefill) {
      gridWrap.append(h('p', { class: 'empty', style: { margin: 0 } }, S.loaded ? 'Generate to watch the cache fill.' : 'Load the weights and generate to watch the cache fill.'));
      facts.textContent = ''; vecs.textContent = '';
      return;
    }
    const P = run.ids.length;
    const N = P + Math.max(0, run.steps.length - 1); // the last sampled token isn't fed yet
    const cw = Math.max(3, Math.min(14, Math.floor(820 / Math.max(N, 1))));
    const ch = 6, top = 4;
    const W = cw * N + 40, H = ch * d.L + top + 16;
    const dpr = devicePixelRatio || 1;
    const cv = h('canvas', { 'aria-label': `KV cache: ${d.L} layers by ${N} positions` });
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = `${W}px`; cv.style.height = `${H}px`;
    const g = cv.getContext('2d');
    g.scale(dpr, dpr);
    const reused = run.prefill.reused;
    for (let p = 0; p < N; p++) {
      g.fillStyle = p < reused ? '#b9822a' : p < P ? '#f7d68f' : '#f6c9e3';
      for (let l = 0; l < d.L; l++) g.fillRect(40 + p * cw, top + l * ch, cw - (cw > 4 ? 1 : 0), ch - 1);
    }
    g.strokeStyle = getComputedStyle(document.body).getPropertyValue('--sm');
    g.lineWidth = 2;
    if (N) g.strokeRect(40 + (N - 1) * cw, top - 1, cw, d.L * ch + 1);
    g.fillStyle = '#6f7b8f';
    g.font = '10px Archivo, sans-serif';
    g.textAlign = 'right';
    g.fillText('layer 0', 36, top + 6);
    g.fillText(`layer ${d.L - 1}`, 36, top + d.L * ch);
    g.textAlign = 'left';
    g.fillText(`position 0 … ${N - 1}`, 40, H - 3);
    cv.addEventListener('mousemove', e => {
      const r = cv.getBoundingClientRect();
      const p = Math.floor((e.clientX - r.left - 40) / cw), l = Math.floor((e.clientY - r.top - top) / ch);
      if (p < 0 || p >= N || l < 0 || l >= d.L) return tip(e, null);
      tip(e, `Position ${p}: <code>${esc(vis(S.tok.tokenText(tokenAt(S, p))))}</code><br>Layer ${l}: K and V, ${d.KVH} heads × ${d.HD} numbers each<br>${fmtBytes(d.kvBytesPerToken / d.L)} in bf16`);
    });
    cv.addEventListener('mouseleave', e => tip(e, null));
    gridWrap.append(cv);
    requestAnimationFrame(() => { gridWrap.scrollLeft = gridWrap.scrollWidth; });

    // Facts
    const per = d.kvBytesPerToken;
    const ctxLen = S.config.max_position_embeddings || 8192;
    const free = 80e9 - d.params * 2;
    facts.textContent = '';
    facts.append(
      h('div', { class: 'facts' },
        h('span', {}, 'Per token ', h('b', {}, `2 × ${d.L} layers × ${d.KVH} heads × ${d.HD} × 2 B = ${fmtBytes(per)}`)),
        h('span', {}, 'This request ', h('b', {}, `${fmtInt(N)} tokens = ${fmtBytes(N * per)}`)),
        h('span', {}, 'In this tab (32-bit floats) ', h('b', {}, fmtBytes(N * per * 2)))),
      h('p', { class: 'caption' }, `At the full ${fmtInt(ctxLen)}-token context, one conversation needs ${fmtBytes(ctxLen * per)}. After the weights, an 80 GB H100 has room for about ${fmtInt(free / (ctxLen * per))} of them at once. The KV cache, not compute, decides how many users one GPU can serve.`),
      h('table', { class: 'data' },
        h('thead', {}, h('tr', {}, h('th', {}, 'Model'), h('th', {}, 'Layers × KV heads × head dim'), h('th', {}, 'KV per token'), h('th', {}, 'At 8,192 tokens'))),
        h('tbody', {},
          h('tr', { class: 'hl' }, h('td', {}, S.meta.label), h('td', {}, `${d.L} × ${d.KVH} × ${d.HD}`), h('td', {}, fmtBytes(per)), h('td', {}, fmtBytes(per * 8192))),
          KNOWN.map(([n, L, K, HD]) => h('tr', {}, h('td', {}, n), h('td', {}, `${L} × ${K} × ${HD}`), h('td', {}, fmtBytes(2 * L * K * HD * 2)), h('td', {}, fmtBytes(2 * L * K * HD * 2 * 8192))))),
      ),
    );
    // Latest K and V vectors
    vecs.textContent = '';
    const latest = [...run.steps].reverse().find(s => s.trace?.qkv);
    const tr = latest?.trace || run.prefill.trace;
    if (tr?.qkv) {
      const kc = h('canvas', { class: 'hm', 'aria-hidden': 'true' });
      const vc = h('canvas', { class: 'hm', 'aria-hidden': 'true' });
      drawVector(kc, tr.qkv.k); drawVector(vc, tr.qkv.v);
      vecs.append(h('p', { class: 'caption' }, `What one cache entry holds: the key and value that `, h('code', {}, vis(S.tok.tokenText(tokenAt(S, tr.qkv.pos)))), ` (position ${tr.qkv.pos}) wrote in layer 0, KV head 0. ${d.HD} numbers each. The key is already rotated by RoPE.`),
        h('div', { class: 'kv-vecs' }, h('span', {}, 'K'), kc, h('span', {}, 'V'), vc));
    }
  }

  function drawSteps() {
    steps.textContent = '';
    const run = S.run;
    const dec = run?.steps.filter(s => s.kind === 'decode') || [];
    if (!dec.length) { steps.append(h('p', { class: 'caption' }, 'Per-step times appear once decoding starts.')); return; }
    const W = 860, H = 200, pl = 50, pb = 26, pt = 10;
    const max = Math.max(...dec.map(s => s.ms)) * 1.1;
    const bw = (W - pl - 10) / Math.max(dec.length, 24);
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Time per decode step in milliseconds');
    const add = (tag, a, txt) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); if (txt !== undefined) e.textContent = txt; svg.append(e); return e; };
    const Y = v => pt + (H - pt - pb) * (1 - v / max);
    for (const f of [0, 0.5, 1]) {
      add('line', { x1: pl, x2: W, y1: Y(max * f), y2: Y(max * f), stroke: 'var(--rule-2)' });
      add('text', { x: pl - 6, y: Y(max * f) + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--ink-3)' }, fmtMs(max * f));
    }
    dec.forEach((s, i) => {
      const r = add('rect', { x: pl + i * bw + 1, y: Y(s.ms), width: Math.max(1, bw - 2), height: Math.max(1, Y(0) - Y(s.ms)), fill: s.noCache ? 'var(--sm)' : 'var(--ink)', rx: 1 });
      const t = document.createElementNS(NS, 'title');
      t.textContent = `Step ${i + 1}: ${fmtMs(s.ms)}${s.noCache ? ' (no KV cache)' : ''}`;
      r.append(t);
    });
    add('text', { x: pl, y: H - 6, 'font-size': 11, fill: 'var(--ink-3)' }, 'decode step →');
    const withC = dec.filter(s => !s.noCache), without = dec.filter(s => s.noCache);
    const avg = a => a.reduce((x, s) => x + s.ms, 0) / a.length;
    steps.append(svg, h('div', { class: 'facts' },
      withC.length ? h('span', {}, h('i', { style: { display: 'inline-block', width: '10px', height: '10px', background: 'var(--ink)', marginRight: '6px' } }), 'With cache, average ', h('b', {}, fmtMs(avg(withC)))) : null,
      without.length ? h('span', {}, h('i', { style: { display: 'inline-block', width: '10px', height: '10px', background: 'var(--sm)', marginRight: '6px' } }), 'Without cache, average ', h('b', {}, fmtMs(avg(without)))) : null,
      h('span', {}, 'Prefill of the prompt ', h('b', {}, fmtMs(run.prefill.ms)))));
  }

  return {
    update(reason) {
      if (reason === 'model') build();
      if (!built) return;
      if (['model', 'weights', 'prefill', 'step', 'branch'].includes(reason)) { drawGrid(); drawSteps(); }
    },
  };
}
