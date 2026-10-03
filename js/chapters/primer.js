// 0. The ideas you need first: vectors, tensors, matrix × vector, hardware.

import { $, h, fmtInt, vis, drawVector } from '../util.js';

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
          h('tr', {}, h('td', {}, 'The notes kept while writing (KV cache), keys only. Drawn below.'), h('td', {}, '4-D tensor'), h('td', {}, `${L} layers × ${T} tokens × ${KVH} heads × ${HD}`)))));
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

  // ---------------- hardware: two journeys, with explicit from → to
  function hardware() {
    const el = $('#primer-hw');
    el.textContent = '';
    const d = ctx.dims();
    const bytes = d ? d.params * 2 : 269e6;
    const place = (cls, name, term, desc) => h('div', { class: `hw-place ${cls}` }, h('b', {}, h('dfn', { 'data-term': term }, name)), h('p', {}, desc));
    const ms = (b, bw) => { const t = b / bw * 1000; return t < 1 ? `${(t * 1000).toFixed(0)} µs` : `${t.toFixed(1)} ms`; };
    const stop = (cls, label) => h('span', { class: `stop ${cls}` }, label);
    const journey = (title, when, from, via, to, bw, width, barCls) => h('div', { class: 'journey' },
      h('div', { class: 'j-head' }, h('b', {}, title), h('span', { class: 'dim' }, when)),
      h('div', { class: 'j-path' }, from, h('span', { class: 'j-arrow' }, via), to),
      h('div', { class: 'speed' }, h('span', { class: `bar ${barCls}` }, h('i', { style: { width } })), h('b', {}, ms(bytes, bw))));
    el.append(
      h('div', { class: 'hw-row' },
        place('host', 'CPU', 'cpu', 'Handles text: splitting it into tokens and turning answers back into words.'),
        place('host', 'RAM', 'ram', 'The computer’s main memory, next to the CPU.'),
        place('link', 'PCIe', 'pcie', 'The link between the computer and the GPU card.'),
        place('hbm', 'HBM', 'hbm', 'The GPU’s own memory. The weights live here.'),
        place('sm', 'Compute cores', 'sm', 'Thousands of small units on the GPU chip that do the multiplications.')),
      h('div', { class: 'hw-speeds' },
        h('p', {}, `The model’s weights (${Math.round(bytes / 1e6)} MB) make two different trips:`),
        journey('1. Loading the model', 'once, when the server starts',
          stop('host', 'CPU RAM'), 'over PCIe, 64 GB/s', stop('hbm', 'GPU memory (HBM)'), 64e9, '100%', 'link'),
        journey('2. Generating each word', 'again for every single token',
          stop('hbm', 'GPU memory (HBM)'), 'on the chip, 3,350 GB/s', stop('sm', 'compute cores'), 3.35e12, `${64 / 3350 * 100}%`, 'hbm'),
        h('p', { class: 'caption' }, `Figures for an NVIDIA H100, a common data-centre GPU. Trip 1 is slow but happens once. Trip 2 is fast, but it repeats for every word the model writes, so it sets the speed limit. ${d ? `Reading ${Math.round(bytes / 1e6)} MB in ${ms(bytes, 3.35e12)} caps this model at about ${fmtInt(3.35e12 / bytes)} words per second per GPU. An 8-billion-parameter model is 16 GB, which caps it near ${fmtInt(3.35e12 / 16e9)}.` : ''}`)));
  }

  // ---------------- one ID, two tables
  async function twoTables() {
    const el = $('#primer-two-tables');
    if (!el || !S.tok) return;
    const user = document.getElementById('prompt').value;
    let id = S.ids.find(x => S.tok.tokenText(x) === ' capital');
    if (id === undefined) {
      const words = S.ids.filter(x => !S.tok.specialById.has(x) && /\p{L}{3,}/u.test(S.tok.tokenText(x)) && user.includes(S.tok.tokenText(x).trim()));
      id = words.sort((a, b) => S.tok.tokenText(b).length - S.tok.tokenText(a).length)[0] ?? 3575;
    }
    const D = S.config?.hidden_size ?? 576;
    const bar = h('canvas', { class: 'hm', 'aria-hidden': 'true' });
    const nums = h('code', {}, S.loaded ? '…' : '[ … load the model to see them … ]');
    el.textContent = '';
    el.append(
      h('div', { class: 'tt-id' }, h('small', {}, 'token ID'), h('b', {}, String(id))),
      h('div', { class: 'tt-rows' },
        h('div', { class: 'tt-row' }, h('span', { class: 'tt-name' }, 'Vocabulary', h('small', {}, 'tokenizer.json')), h('span', { class: 'tt-arrow' }, '→'), h('code', { class: 'tt-text' }, `“${vis(S.tok.tokenText(id))}”`)),
        h('div', { class: 'tt-row' }, h('span', { class: 'tt-name' }, 'Embedding table', h('small', {}, `model.safetensors, row ${id}`)), h('span', { class: 'tt-arrow' }, '→'),
          h('div', { class: 'tt-vec' }, nums, bar, h('small', {}, `${fmtInt(D)} numbers, drawn as coloured bars`)))));
    if (!S.loaded) return;
    try {
      const r = await ctx.engine.call('rows', { ids: [id] });
      const v = r.data;
      nums.textContent = `[${Array.from(v.slice(0, 6), x => x.toFixed(3)).join(', ')}, … ${fmtInt(D - 6)} more]`;
      drawVector(bar, v);
    } catch { /* engine busy or reset */ }
  }

  // ---------------- the KV cache's 4-D shape, schematically
  function kvSchematic() {
    const el = $('#primer-kv');
    if (!el) return;
    el.textContent = '';
    const d = ctx.dims();
    const L = d ? d.L : 30, KVH = d ? d.KVH : 3, HD = d ? d.HD : 64;
    const T = S.ids.length || 37;
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 640 300');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', `The keys in the KV cache: ${L} stacked sheets, one per layer; on each sheet ${T} token columns of ${KVH} heads times ${HD} numbers`);
    const add = (tag, a, txt) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); if (txt !== undefined) e.textContent = txt; svg.append(e); return e; };
    const cols = 14, x0 = 90, y0 = 70, cw = 22, bandH = 48, gap = 6;
    const sheetW = cols * cw, sheetH = KVH * bandH + (KVH - 1) * gap;
    // sheets behind (other layers)
    for (let k = 4; k >= 1; k--) {
      add('rect', { x: x0 + k * 16, y: y0 - k * 12, width: sheetW, height: sheetH, rx: 3, fill: 'var(--panel)', stroke: 'var(--rule)', 'stroke-width': 1.2 });
    }
    add('text', { x: x0 + 4 * 16 + sheetW + 8, y: y0 - 4 * 12 + 12, 'font-size': 12, fill: 'var(--ink-2)', 'font-weight': 650 }, `× ${L} layers (one sheet each)`);
    // front sheet: token columns × head bands
    const tones = ['#f7d68f', '#f0c46b', '#e8b24c'];
    for (let hh = 0; hh < KVH; hh++) {
      const y = y0 + hh * (bandH + gap);
      for (let c = 0; c < cols; c++) {
        add('rect', { x: x0 + c * cw + 1, y, width: cw - 2, height: bandH, rx: 2, fill: tones[hh % 3], stroke: c === cols - 1 ? 'var(--sm)' : 'none', 'stroke-width': 2 });
        for (let r = 1; r < 4; r++) add('line', { x1: x0 + c * cw + 4, x2: x0 + c * cw + cw - 4, y1: y + r * bandH / 4, y2: y + r * bandH / 4, stroke: 'rgba(150,88,0,0.25)' });
      }
      add('text', { x: x0 - 10, y: y + bandH / 2 + 4, 'text-anchor': 'end', 'font-size': 12, fill: 'var(--ink)', 'font-weight': 650 }, `head ${hh}`);
      add('text', { x: x0 - 10, y: y + bandH / 2 + 18, 'text-anchor': 'end', 'font-size': 10.5, fill: 'var(--ink-3)' }, `${HD} numbers`);
    }
    add('text', { x: x0 + sheetW / 2, y: y0 + sheetH + 22, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--ink)', 'font-weight': 650 }, `${T} tokens → (one column each; a new column is added for every word written)`);
    add('text', { x: x0 + sheetW / 2, y: y0 + sheetH + 40, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--ink-3)' }, `One sheet = one layer’s keys: ${T} × ${KVH} × ${HD} = ${fmtInt(T * KVH * HD)} numbers`);
    add('text', { x: x0 + sheetW + 10, y: y0 + sheetH - 4, 'font-size': 11, fill: 'var(--sm)', 'font-weight': 600 }, '← newest token');
    el.append(svg, h('p', { class: 'caption' }, `Total keys: ${L} × ${T} × ${KVH} × ${HD} = ${fmtInt(L * T * KVH * HD)} numbers, and the same again for values. At 2 bytes per number (bf16), that’s ${fmtInt(L * T * KVH * HD * 2 * 2 / 1000)} KB for this prompt.`));
  }

  // ---------------- (m × n) × (n × 1) = (m × 1)
  function shapeRule() {
    const el = $('#primer-rule');
    if (!el) return;
    el.textContent = '';
    const d = ctx.dims();
    const D = d ? d.D : 576, I = d ? d.I : 1536;
    const T = S.ids.length || 37;
    const m = v => h('span', { class: 'dm' }, v), n = v => h('span', { class: 'dn' }, v), t = v => h('span', { class: 'dt' }, v);
    const row = (label, a, b, c, note) => h('div', { class: 'sr-row' },
      h('span', { class: 'sr-label' }, label),
      h('span', { class: 'sr-shape' }, ...a), h('span', { class: 'sr-op' }, '×'),
      h('span', { class: 'sr-shape' }, ...b), h('span', { class: 'sr-op' }, '='),
      h('span', { class: 'sr-shape' }, ...c), h('span', { class: 'sr-note' }, note));
    el.append(
      row('The rule', [m('m'), ' × ', n('n')], [n('n'), ' × 1'], [m('m'), ' × 1'], 'the two inner sizes must match'),
      row('The example above', [m('3'), ' × ', n('4')], [n('4'), ' × 1'], [m('3'), ' × 1'], '3 rows, so 3 outputs'),
      row('One token, one model matrix', [m(fmtInt(I)), ' × ', n(fmtInt(D))], [n(fmtInt(D)), ' × 1'], [m(fmtInt(I)), ' × 1'], 'feed-forward “up”, chapter 6'),
      row('Your whole prompt at once', [m(fmtInt(I)), ' × ', n(fmtInt(D))], [n(fmtInt(D)), ' × ', t(String(T))], [m(fmtInt(I)), ' × ', t(String(T))], `${T} tokens side by side: this is prefill`),
      h('p', { class: 'caption' }, h('span', { class: 'dm' }, 'm'), ' = rows of the matrix = how many numbers come out. ',
        h('span', { class: 'dn' }, 'n'), ' = columns = how many numbers go in. ',
        h('span', { class: 'dt' }, 'T'), ' = how many tokens are processed together.'));
  }

  let ttTimer = null;
  const twoTablesSoon = () => { clearTimeout(ttTimer); ttTimer = setTimeout(twoTables, 300); };

  // These work before anything is downloaded.
  colour(); matvec(); shapes(); hardware(); kvSchematic(); shapeRule();

  return {
    update(reason) {
      if (reason === 'model') {
        const d = ctx.dims();
        document.querySelectorAll('[data-bind="macsPerTok"]').forEach(e => { e.textContent = `${Math.round(d.activeParams / 1e6)} million`; });
        shapes(); hardware(); kvSchematic(); shapeRule();
      }
      if (reason === 'tokens') { shapes(); kvSchematic(); shapeRule(); twoTablesSoon(); }
      if (reason === 'weights') twoTables();
    },
  };
}
