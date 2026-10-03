// 0. The ideas you need first: vectors, tensors, matrix × vector, hardware.

import { $, h, fmtInt } from '../util.js';

const NS = 'http://www.w3.org/2000/svg';

export default function primer(ctx) {
  const { S } = ctx;

  // ---------------- a colour is a vector
  function colour() {
    const el = $('#primer-color');
    el.textContent = '';
    const v = [212, 120, 40];
    const swatch = h('div', { class: 'swatch', 'aria-hidden': 'true' });
    const readout = h('code', { class: 'vec-readout' });
    const sliders = h('div', { class: 'rgb' });
    const names = ['Red', 'Green', 'Blue'];
    const draw = () => {
      swatch.style.background = `rgb(${v[0]},${v[1]},${v[2]})`;
      readout.textContent = `[${v.join(', ')}]`;
    };
    names.forEach((n, i) => {
      const out = h('output', {}, String(v[i]));
      const inp = h('input', { type: 'range', min: 0, max: 255, value: v[i], 'aria-label': n });
      inp.addEventListener('input', () => { v[i] = +inp.value; out.textContent = inp.value; draw(); });
      sliders.append(h('label', {}, h('span', {}, n), inp, out));
    });
    el.append(swatch, h('div', {}, sliders, h('p', { class: 'vec-line' }, 'This colour as a vector: ', readout)));
    draw();
  }

  // ---------------- shapes of tensors
  function shapes() {
    const el = $('#primer-shapes');
    el.textContent = '';
    const d = ctx.dims();
    const D = d ? d.D : 576, L = d ? d.L : 30, KVH = d ? d.KVH : 3, HD = d ? d.HD : 64;
    const T = S.ids.length || 'T';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 720 150');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'A single number, a vector drawn as a row of cells, a matrix drawn as a grid, and a tensor drawn as a stack of grids');
    const add = (tag, a, txt) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); if (txt !== undefined) e.textContent = txt; svg.append(e); return e; };
    const cell = (x, y, s, fill) => add('rect', { x, y, width: s, height: s, fill, stroke: 'var(--panel)', 'stroke-width': 1.5, rx: 2 });
    const tone = i => ['#c4cfde', '#7f95b8', '#2f4570', '#e2e7ed', '#4c6488'][i % 5];
    // number
    cell(40, 50, 26, tone(2));
    // vector
    for (let i = 0; i < 6; i++) cell(150 + i * 26, 50, 26, tone(i));
    // matrix
    for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) cell(370 + c * 22, 26 + r * 22, 22, tone(r + c));
    // tensor (stack)
    for (let k = 3; k >= 0; k--) for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) cell(560 + c * 20 + k * 14, 22 + r * 20 - k * 10 + 10, 20, tone(r + c + k));
    const label = (x, t1, t2) => {
      add('text', { x, y: 136, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 650, fill: 'var(--ink)' }, t1);
      add('text', { x, y: 150, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--ink-3)' }, t2);
    };
    label(53, 'a number', 'no directions');
    label(228, 'a vector', '1 direction: a list');
    label(425, 'a matrix', '2 directions: rows × columns');
    label(620, 'a 3-D tensor', '3 directions: a stack of grids');
    el.append(svg,
      h('table', { class: 'data shapes-table' },
        h('thead', {}, h('tr', {}, h('th', {}, 'In this model'), h('th', {}, 'Kind'), h('th', {}, 'Shape'))),
        h('tbody', {},
          h('tr', {}, h('td', {}, 'One token’s meaning'), h('td', {}, 'vector'), h('td', {}, `${fmtInt(D)} numbers`)),
          h('tr', {}, h('td', {}, 'Your whole prompt'), h('td', {}, 'matrix'), h('td', {}, `${T} tokens × ${fmtInt(D)}`)),
          h('tr', {}, h('td', {}, 'One weight matrix, e.g. the “query” maker'), h('td', {}, 'matrix'), h('td', {}, `${fmtInt(D)} × ${fmtInt(D)}`)),
          h('tr', {}, h('td', {}, 'The notes kept while writing (KV cache), keys only'), h('td', {}, '4-D tensor'), h('td', {}, `${L} layers × ${T} tokens × ${KVH} heads × ${HD}`)))));
  }

  // ---------------- matrix × vector, by hand
  function matvec() {
    const el = $('#primer-matvec');
    el.textContent = '';
    const M = [[1, 2, 0, -1], [0.5, -1, 3, 0], [-2, 0, 1, 1]];
    const x = [2, 1, 0.5, 3];
    let sel = 0;
    const grid = h('div', { class: 'mv' });
    const work = h('p', { class: 'mv-work', 'aria-live': 'polite' });
    const fmt = v => (Math.round(v * 100) / 100).toString().replace('-', '−');
    const out = () => M.map(row => row.reduce((s, w, j) => s + w * x[j], 0));
    function draw() {
      grid.textContent = '';
      const mat = h('div', { class: 'mv-mat', role: 'table', 'aria-label': 'Weight matrix' },
        M.map((row, i) => h('div', { class: `mv-row${i === sel ? ' on' : ''}`, role: 'row' }, row.map(w => h('span', { role: 'cell' }, fmt(w))))));
      const vec = h('div', { class: 'mv-vec' }, x.map((v, j) => {
        const inp = h('input', { type: 'number', step: 0.5, value: v, 'aria-label': `Input number ${j + 1}` });
        inp.addEventListener('input', () => { x[j] = Number(inp.value) || 0; drawOut(); });
        return inp;
      }));
      const outs = h('div', { class: 'mv-out' });
      function drawOut() {
        const yy = out();
        outs.textContent = '';
        yy.forEach((v, i) => {
          const b = h('button', { class: i === sel ? 'on' : '', 'aria-pressed': String(i === sel) }, fmt(v));
          b.addEventListener('click', () => { sel = i; draw(); });
          outs.append(b);
        });
        work.textContent = '';
        work.append(`Output ${sel + 1} = row ${sel + 1} · input = `,
          ...M[sel].flatMap((w, j) => [j ? ' + ' : '', h('span', { class: 'w' }, `(${fmt(w)} × ${fmt(x[j])})`)]),
          ' = ', h('b', {}, fmt(yy[sel])));
      }
      grid.append(
        h('div', { class: 'mv-col' }, h('small', {}, 'weights (3 × 4 matrix)'), mat),
        h('div', { class: 'mv-op', 'aria-hidden': 'true' }, '×'),
        h('div', { class: 'mv-col' }, h('small', {}, 'input vector'), vec),
        h('div', { class: 'mv-op', 'aria-hidden': 'true' }, '='),
        h('div', { class: 'mv-col' }, h('small', {}, 'output vector'), outs));
      drawOut();
    }
    draw();
    el.append(grid, work, h('p', { class: 'caption' }, 'Four numbers in, three out. Every arrow in a neural network diagram is one of these. The weights decide how much each input counts toward each output.'));
  }

  // ---------------- hardware
  function hardware() {
    const el = $('#primer-hw');
    el.textContent = '';
    const d = ctx.dims();
    const bytes = d ? d.params * 2 : 269e6;
    const place = (cls, name, term, desc) => h('div', { class: `hw-place ${cls}` }, h('b', {}, h('dfn', { 'data-term': term }, name)), h('p', {}, desc));
    const ms = (b, bw) => { const t = b / bw * 1000; return t < 1 ? `${(t * 1000).toFixed(0)} µs` : `${t.toFixed(1)} ms`; };
    el.append(
      h('div', { class: 'hw-row' },
        place('host', 'CPU', 'cpu', 'Handles text: splitting it into tokens and turning answers back into words.'),
        place('host', 'RAM', 'ram', 'The computer’s main memory, next to the CPU.'),
        place('link', 'PCIe', 'pcie', 'The cable-like link between the computer and the GPU card.'),
        place('hbm', 'HBM', 'hbm', 'The GPU’s own memory. The weights live here.'),
        place('sm', 'Compute cores', 'sm', 'Thousands of small units on the GPU chip that do the multiplications.')),
      h('div', { class: 'hw-speeds' },
        h('p', {}, 'How long it takes to move the whole model once (', h('b', {}, `${Math.round(bytes / 1e6)} MB`), '):'),
        h('div', { class: 'speed' }, h('span', {}, 'Over PCIe, 64 GB/s'), h('span', { class: 'bar link' }, h('i', { style: { width: '100%' } })), h('b', {}, ms(bytes, 64e9))),
        h('div', { class: 'speed' }, h('span', {}, 'From HBM, 3,350 GB/s'), h('span', { class: 'bar hbm' }, h('i', { style: { width: `${64 / 3350 * 100}%` } })), h('b', {}, ms(bytes, 3.35e12))),
        h('p', { class: 'caption' }, 'Figures for an NVIDIA H100, a common data-centre GPU. Shorter bar = less time. That’s why the weights are copied to the GPU once, at startup, and kept there.')));
  }

  // These work before anything is downloaded.
  colour(); matvec(); shapes(); hardware();

  return {
    update(reason) {
      if (reason === 'model') {
        const d = ctx.dims();
        document.querySelectorAll('[data-bind="macsPerTok"]').forEach(e => { e.textContent = `${Math.round(d.activeParams / 1e6)} million`; });
        shapes(); hardware();
      }
      if (reason === 'tokens') shapes();
    },
  };
}
