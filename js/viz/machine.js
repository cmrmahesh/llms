// The machine map: a server with a CPU, RAM, a PCIe link and one H100-class GPU.
// Each chapter lights up the parts it uses and animates the data it moves.

import { fmtBytes, fmtInt } from '../util.js';

const NS = 'http://www.w3.org/2000/svg';
const COL = {
  host: ['var(--host)', 'var(--host-fill)'],
  link: ['var(--link)', 'var(--link-fill)'],
  hbm: ['var(--hbm)', 'var(--hbm-fill)'],
  sm: ['var(--sm)', 'var(--sm-fill)'],
};

// Node geometry (viewBox 300 × 450)
const NODES = {
  disk: { x: 20, y: 8, w: 260, h: 34, kind: 'link', label: 'Disk or network', sub: '' },
  cpu: { x: 20, y: 70, w: 122, h: 58, kind: 'host', label: 'CPU', sub: 'tokenize · sample · text' },
  ram: { x: 158, y: 70, w: 122, h: 58, kind: 'host', label: 'CPU RAM', sub: 'staging copy' },
  hbm: { x: 26, y: 200, w: 248, h: 92, kind: 'hbm', label: 'HBM · 80 GB · 3.35 TB/s', sub: '' },
  l2: { x: 26, y: 308, w: 248, h: 32, kind: 'hbm', label: 'L2 cache · 50 MB', sub: '' },
  sm: { x: 26, y: 356, w: 248, h: 78, kind: 'sm', label: '132 SMs', sub: '256 KB SRAM each + tensor cores' },
};
// Link paths
const LINKS = {
  'disk-ram': 'M219 42 V70',
  'ram-cpu': 'M158 99 H142',
  'cpu-ram': 'M142 99 H158',
  'ram-hbm': 'M219 128 V200',
  'cpu-hbm': 'M81 128 V200',
  'hbm-cpu': 'M81 200 V128',
  'hbm-l2': 'M150 292 V308',
  'l2-sm': 'M150 340 V356',
  'hbm-sm': 'M150 292 V356',
  'sm-hbm': 'M110 356 V292',
};

// Where each flow's label sits: [x, y, text-anchor]
const LABEL_AT = {
  'disk-ram': [227, 60, 'start'],
  'ram-hbm': [227, 150, 'start'],
  'cpu-hbm': [73, 150, 'end'],
  'hbm-cpu': [73, 150, 'end'],
  'hbm-sm': [158, 304, 'start'],
  'sm-hbm': [102, 304, 'end'],
};

function chapters(d, S) {
  const w = d ? fmtBytes(d.activeParams * 2) : 'all weights';
  const kvTok = d ? fmtBytes(d.kvBytesPerToken) : '';
  const file = d ? fmtBytes(d.params * 2) : '';
  return {
    hero: { on: [], flows: [], cap: 'This map follows you down the page. It shows which part of a GPU server each step uses, and what crosses between them.' },
    console: { on: ['cpu'], flows: [], cap: 'Your prompt is a string in CPU memory. Nothing has touched the GPU yet.' },
    artifact: { on: ['disk', 'ram', 'hbm'], flows: [['disk-ram', 'link', file], ['ram-hbm', 'hbm', 'copied once']], cap: `At startup the engine reads ${file} of weights from disk into RAM and copies them over PCIe into HBM. They stay there for every request after that.` },
    tokenizer: { on: ['cpu'], flows: [], cap: 'Tokenizing is string work on one CPU core: microseconds for a short prompt. Its output is one 4-byte integer per token.' },
    embed: { on: ['cpu', 'hbm', 'sm'], flows: [['cpu-hbm', 'link', 'token IDs'], ['hbm-sm', 'hbm', 'T rows']], cap: `The IDs cross PCIe, a few bytes each. The GPU gathers one ${d ? fmtInt(d.D) : ''}-wide row per token from the embedding table in HBM.` },
    position: { on: ['sm'], flows: [], cap: 'RoPE is a handful of multiply-adds per number. It runs in registers inside the attention kernel and stores nothing.' },
    attention: { on: ['hbm', 'l2', 'sm'], flows: [['hbm-sm', 'hbm', 'Wq Wk Wv Wo'], ['sm-hbm', 'sm', 'new K, V']], cap: 'Each layer streams its attention weights from HBM through L2 into the SMs, multiplies them on tensor cores, and writes the new keys and values back to HBM.' },
    ffn: { on: ['hbm', 'l2', 'sm'], flows: [['hbm-sm', 'hbm', 'W gate, up, down']], cap: 'The feed-forward weights are the biggest part of every layer, so they are the biggest stream out of HBM.' },
    sample: { on: ['sm', 'cpu'], flows: [['hbm-sm', 'hbm', 'LM head'], ['hbm-cpu', 'link', '1 token ID']], cap: 'The LM head turns one vector into a score per vocabulary entry. Sampling can run on the GPU. Only the chosen ID, 4 bytes, goes back to the CPU to become text.' },
    kv: { on: ['hbm', 'l2', 'sm'], flows: [['hbm-sm', 'hbm', `${w} + KV`], ['sm-hbm', 'sm', `+${kvTok}`]], cap: `Every decode step reads ${w} of weights plus the whole KV cache, and writes ${kvTok} of new K and V for one token.` },
    roofline: { on: ['hbm', 'sm'], flows: [['hbm-sm', 'hbm', 'bytes']], cap: 'Decode keeps the 3.35 TB/s memory bus busy while the tensor cores mostly wait. Prefill is the other way round.' },
    prefix: { on: ['hbm'], flows: [], cap: 'Prefix caching keeps finished KV blocks in HBM after a request ends, so the next request with the same start can skip computing them.' },
    summary: { on: ['disk', 'cpu', 'ram', 'hbm', 'l2', 'sm'], flows: [['cpu-hbm', 'link', 'IDs ↕'], ['hbm-sm', 'hbm', 'weights + KV']], cap: 'Disk and PCIe matter once, at load. Per token, the traffic that counts is HBM to SM.' },
  };
}

export function createMachine(root, captionEl, ctx) {
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 300 450');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Diagram of a GPU server: disk, CPU, RAM, PCIe link, and a GPU with HBM, L2 cache and streaming multiprocessors');
  const el = (tag, attrs, parent = svg) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    parent.append(e);
    return e;
  };

  // GPU outline
  el('rect', { x: 10, y: 172, width: 280, height: 272, rx: 10, fill: 'none', stroke: 'var(--rule)', 'stroke-width': 1.5 });
  el('text', { x: 22, y: 190, 'font-size': 11, fill: 'var(--ink-3)', 'font-weight': 600 }).textContent = 'GPU (H100 SXM)';
  el('text', { x: 150, y: 160, 'font-size': 10, fill: 'var(--ink-3)', 'text-anchor': 'middle' }).textContent = 'PCIe 5 ×16 · 64 GB/s';

  // static links
  const base = {};
  for (const [id, d] of Object.entries(LINKS)) {
    if (['ram-cpu', 'hbm-cpu', 'sm-hbm', 'hbm-sm'].includes(id)) continue;
    base[id] = el('path', { d, stroke: 'var(--rule)', 'stroke-width': 2, fill: 'none', class: 'm-link' });
  }

  const nodes = {};
  for (const [id, n] of Object.entries(NODES)) {
    const g = el('g', { class: 'm-node' });
    const r = el('rect', { x: n.x, y: n.y, width: n.w, height: n.h, rx: 6, fill: 'var(--panel)', stroke: 'var(--rule)', 'stroke-width': 1.5 }, g);
    const t = el('text', { x: n.x + 10, y: n.y + 20, 'font-size': 12.5, 'font-weight': 650, fill: 'var(--ink-3)' }, g);
    t.textContent = n.label;
    let s = null;
    if (n.sub) {
      s = el('text', { x: n.x + 10, y: n.y + 37, 'font-size': 10.5, fill: 'var(--ink-3)' }, g);
      s.textContent = n.sub;
    }
    nodes[id] = { g, r, t, s, n };
  }
  // HBM contents: weights and KV cache rows
  const hb = NODES.hbm;
  const wLabel = el('text', { x: hb.x + 10, y: hb.y + 44, 'font-size': 10.5, fill: 'var(--ink-2)' });
  const wBar = el('rect', { x: hb.x + 130, y: hb.y + 36, width: 2, height: 10, fill: 'var(--hbm)', rx: 1 });
  const kvLabel = el('text', { x: hb.x + 10, y: hb.y + 64, 'font-size': 10.5, fill: 'var(--ink-2)' });
  const kvBar = el('rect', { x: hb.x + 130, y: hb.y + 56, width: 2, height: 10, fill: 'var(--sm)', rx: 1 });
  const freeLabel = el('text', { x: hb.x + 10, y: hb.y + 83, 'font-size': 10, fill: 'var(--ink-3)' });

  const flowLayer = el('g', {});
  root.append(svg);

  // Compact strip for small screens
  const compact = document.createElementNS(NS, 'svg');
  compact.setAttribute('viewBox', '0 0 360 40');
  compact.classList.add('m-compact');
  compact.setAttribute('aria-hidden', 'true');
  const cNodes = {};
  [['cpu', 'CPU', 0], ['link', 'PCIe', 92], ['hbm', 'HBM', 184], ['sm', 'SMs', 276]].forEach(([id, label, x]) => {
    const r = document.createElementNS(NS, 'rect');
    Object.entries({ x: x + 2, y: 4, width: 80, height: 32, rx: 6, fill: 'var(--panel)', stroke: 'var(--rule)', 'stroke-width': 1.5 }).forEach(([k, v]) => r.setAttribute(k, v));
    const t = document.createElementNS(NS, 'text');
    Object.entries({ x: x + 42, y: 25, 'font-size': 13, 'font-weight': 650, 'text-anchor': 'middle', fill: 'var(--ink-3)' }).forEach(([k, v]) => t.setAttribute(k, v));
    t.textContent = label;
    compact.append(r, t);
    cNodes[id] = { r, t };
  });
  root.append(compact);
  const style = document.createElement('style');
  style.textContent = '#machine .m-compact{display:none}@media (max-width:1180px){#machine>svg:first-child{display:none}#machine .m-compact{display:block;width:100%;max-width:360px;height:auto}}';
  root.append(style);

  let current = 'hero';
  let live = false;

  function render() {
    const d = ctx.dims();
    const cfg = chapters(d, ctx.S)[current] || chapters(d, ctx.S).hero;
    for (const [id, nd] of Object.entries(nodes)) {
      const on = cfg.on.includes(id);
      const [stroke, fill] = COL[nd.n.kind];
      nd.r.setAttribute('fill', on ? fill : 'var(--panel)');
      nd.r.setAttribute('stroke', on ? stroke : 'var(--rule)');
      nd.t.setAttribute('fill', on ? stroke : 'var(--ink-3)');
      if (nd.s) nd.s.setAttribute('fill', on ? 'var(--ink-2)' : 'var(--ink-3)');
    }
    const compactOn = { cpu: cfg.on.includes('cpu') || cfg.on.includes('ram'), link: cfg.flows.some(f => f[1] === 'link'), hbm: cfg.on.includes('hbm'), sm: cfg.on.includes('sm') };
    for (const [id, c] of Object.entries(cNodes)) {
      const kind = id === 'cpu' ? 'host' : id;
      const [stroke, fill] = COL[kind];
      c.r.setAttribute('fill', compactOn[id] ? fill : 'var(--panel)');
      c.r.setAttribute('stroke', compactOn[id] ? stroke : 'var(--rule)');
      c.t.setAttribute('fill', compactOn[id] ? stroke : 'var(--ink-3)');
    }
    flowLayer.textContent = '';
    for (const [id, kind, label] of cfg.flows) {
      const [stroke] = COL[kind];
      const p = document.createElementNS(NS, 'path');
      p.setAttribute('d', LINKS[id]);
      p.setAttribute('stroke', stroke);
      p.setAttribute('stroke-width', 3);
      p.setAttribute('fill', 'none');
      p.setAttribute('class', 'm-flow');
      flowLayer.append(p);
      if (label) {
        const [x, y, anchor] = LABEL_AT[id];
        const t = document.createElementNS(NS, 'text');
        t.setAttribute('font-size', 10);
        t.setAttribute('font-weight', 600);
        t.setAttribute('fill', stroke);
        t.setAttribute('x', x);
        t.setAttribute('y', y);
        t.setAttribute('text-anchor', anchor);
        t.textContent = label;
        flowLayer.append(t);
      }
    }
    // HBM occupancy
    if (d) {
      const wBytes = d.params * 2;
      const run = ctx.S.run;
      const kvTokens = run ? run.ids.length + Math.max(0, run.steps.length - 1) : 0;
      const kvBytes = kvTokens * d.kvBytesPerToken;
      const cap = 80e9;
      wLabel.textContent = `weights ${fmtBytes(wBytes)}`;
      kvLabel.textContent = `KV cache ${fmtBytes(kvBytes)}`;
      wBar.setAttribute('width', Math.max(2, 100 * wBytes / cap));
      kvBar.setAttribute('width', Math.max(kvBytes ? 2 : 0, 100 * kvBytes / cap));
      freeLabel.textContent = `${((wBytes + kvBytes) / cap * 100).toFixed(2)}% of 80 GB used by this request`;
    }
    captionEl.textContent = cfg.cap;
    svg.classList.toggle('live', live);
  }

  return {
    focus(id) { if (id && id !== current) { current = id; render(); } },
    update(reason) {
      if (reason === 'run-start') live = true;
      if (reason === 'idle') live = false;
      if (['model', 'header', 'weights', 'step', 'prefill', 'run-start', 'idle'].includes(reason)) render();
    },
    render,
  };
}
