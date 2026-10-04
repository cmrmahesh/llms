// 9. The whole prompt through the model, as matrices. Prefill sends every prompt
// token through together: a T × D matrix, one row per token. Each layer takes it in
// and hands on a matrix of the same shape. This chapter shows that matrix after every
// layer, every intermediate matrix of one layer, and the final scores of every row.
// The worker recomputes these on a side pass (model.inspectPrompt / inspectLayer).

import { $, h, fmtInt, vis, drawGrid, diverging, sequential, softmaxTop, tip, esc } from '../util.js';

const STAGES = ['Normalize', 'Q, K, V', 'Attention', 'Combine + add', 'Normalize', 'Feed-forward + add'];
const KEEP_LAYERS = 6;
const MASKED = [226, 230, 236];

export default function tracer(ctx) {
  const { S } = ctx;
  let pos = 0;             // 0 = embedding, 1..6L = layer stages, 6L+1 = final scores
  let head = 0, zoomRaw = false;
  let base = null;         // { run, T, resid, guesses } for the run being shown
  let layers = new Map();  // layer -> every intermediate matrix of that layer
  let busy = false, failed = null, visible = false;
  let lens = null, lensFor = null, lensBusy = false;
  let ui = null;

  const d = () => ctx.dims();
  const last = () => 6 * d().L + 1;
  const where = p => (p === 0 ? { kind: 'embed' } : p === last() ? { kind: 'final' } : { kind: 'layer', l: Math.floor((p - 1) / 6), s: (p - 1) % 6 });
  const ids = () => S.run?.ids;
  const tokText = t => vis(S.tok.tokenText(ids()[t]));
  const fmt = v => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3));
  const dt = v => h('span', { class: 'dt' }, v), dm = v => h('span', { class: 'dm' }, v), dn = v => h('span', { class: 'dn' }, v);
  const shp = (r, c) => h('span', { class: 'mx-shape' }, r, ' × ', c);
  const rowLen = (a, t, n) => { let s = 0; for (let i = t * n; i < t * n + n; i++) s += a[i] * a[i]; return Math.sqrt(s); };

  // Colour scale that ignores a handful of huge outliers (the residual stream has some).
  function robust(a, q = 0.985) {
    const step = Math.max(1, Math.floor(a.length / 20000));
    const s = [];
    for (let i = 0; i < a.length; i += step) s.push(Math.abs(a[i]));
    s.sort((x, y) => x - y);
    return s[Math.floor((s.length - 1) * q)] || 1;
  }

  function hover(el, rows, cols, info) {
    el.addEventListener('mousemove', e => {
      const b = el.getBoundingClientRect();
      const r = Math.min(rows - 1, Math.max(0, Math.floor(((e.clientY - b.top) / b.height) * rows)));
      const c = Math.min(cols - 1, Math.max(0, Math.floor(((e.clientX - b.left) / b.width) * cols)));
      tip(e, info(r, c));
    });
    el.addEventListener('mouseleave', () => tip(null));
  }

  const rowH = T => Math.max(2, Math.min(12, Math.floor(420 / T)));

  // ---------------- building blocks
  // A matrix, drawn one pixel per number and stretched to fit.
  function mx(arr, rows, cols, o) {
    const cv = h('canvas', { class: 'mx-cv', 'aria-hidden': 'true' });
    drawGrid(cv, arr, rows, cols, o.color ?? diverging, o.scale ?? robust(arr));
    cv.style.height = `${o.height ?? rows * o.rh}px`;
    const box = h('div', { class: `mx-box${o.heads ? ' mx-split' : ''}` }, cv);
    if (o.heads) box.style.setProperty('--heads', o.heads);
    if (o.hi !== undefined) box.append(h('i', { class: 'mx-hi', style: { left: `${(o.hi / o.heads) * 100}%`, width: `${100 / o.heads}%` } }));
    hover(cv, rows, cols, o.info ?? ((r, c) => `<b>${esc(tokText(r))}</b> (row ${r}), column ${c}<br>${o.name}: ${fmt(arr[r * cols + c])}`));
    return h('figure', { class: 'mx', style: { flex: `${o.flex ?? cols} 1 0` } },
      h('figcaption', {}, h('b', {}, o.name), ' ', o.shape), box);
  }

  // A weight matrix, shown small: the same one is used for every row.
  function wt(name, preview, inDim, outDim) {
    const p = S.previews?.find(x => x.name === preview);
    const box = h('div', { class: 'mx-w' });
    if (p?.grid) {
      // Stored as [out × in]; X × Wᵀ uses it flipped, [in × out].
      const g = new Float32Array(p.gr * p.gc);
      for (let i = 0; i < p.gr; i++) for (let j = 0; j < p.gc; j++) g[j * p.gr + i] = p.grid[i * p.gc + j];
      const cv = h('canvas', { 'aria-hidden': 'true' });
      drawGrid(cv, g, p.gc, p.gr, diverging, 3 * p.std || 1);
      box.append(cv);
    }
    return h('figure', { class: 'mx mx-wfig' },
      h('figcaption', {}, h('b', {}, `${name}ᵀ`), ' ', shp(dn(fmtInt(inDim)), dm(fmtInt(outDim)))), box,
      h('small', { class: 'mx-wnote' }, 'same for every row'));
  }

  const op = t => h('div', { class: 'mx-op', 'aria-hidden': 'true' }, t);

  function tokCol(T, rh) {
    const col = h('div', { class: `mx-toks${rh < 8 ? ' tiny' : ''}` });
    col.style.setProperty('--rh', `${rh}px`);
    for (let t = 0; t < T; t++) col.append(h('span', { class: t === T - 1 ? 'last' : '' }, tokText(t)));
    return h('figure', { class: 'mx mx-tokfig' }, h('figcaption', {}, h('b', {}, 'token'), ' ', h('span', { class: 'mx-shape' }, 'one per row')), col);
  }

  // Each row's length, as a bar.
  function lenCol(arr, T, n, rh) {
    const ls = Array.from({ length: T }, (_, t) => rowLen(arr, t, n));
    const max = Math.max(...ls) || 1;
    const col = h('div', { class: 'mx-lens' });
    col.style.setProperty('--rh', `${rh}px`);
    ls.forEach((v, t) => {
      const b = h('span', {}, h('i', { style: { width: `${(v / max) * 100}%` } }));
      b.addEventListener('mousemove', e => tip(e, `<b>${esc(tokText(t))}</b>: row length ${v.toFixed(1)}`));
      b.addEventListener('mouseleave', () => tip(null));
      col.append(b);
    });
    return h('figure', { class: 'mx mx-lenfig' }, h('figcaption', {}, h('b', {}, 'length')), col);
  }

  const mrow = (...items) => h('div', { class: 'mx-row' }, ...items);
  const say = t => h('p', { class: 'mx-say' }, ...[].concat(t));

  // ---------------- build
  function build() {
    const el = $('#tracer');
    el.textContent = '';
    ui = null;
    if (!d()) return;
    if (!S.run?.ids?.length) {
      el.append(h('p', { class: 'empty', style: { margin: 0 } }, S.loaded ? 'Generate something, then come back: this chapter replays your prompt.' : 'Load the model and generate to see your prompt go through the layers.'));
      return;
    }
    const L = d().L;
    const back = h('button', { class: 'btn' }, '◀ Back');
    const next = h('button', { class: 'btn primary' }, 'Next ▶');
    back.addEventListener('click', () => go(pos - 1));
    next.addEventListener('click', () => go(pos + 1));
    const slider = h('input', { type: 'range', min: 0, max: L - 1, value: 0, 'aria-label': 'Layer' });
    const sliderOut = h('output', {}, '0');
    slider.addEventListener('input', () => {
      const w = where(pos);
      go(1 + 6 * +slider.value + (w.kind === 'layer' ? w.s : 0));
    });
    // The flow: the prompt matrix after each layer, left to right.
    const strip = h('div', { class: 'tv-strip mx-strip', role: 'group', 'aria-label': 'The prompt matrix after each layer. Choose one to open it.' });
    const mk = (label, target, cls) => {
      const b = h('button', { class: `tv-cell ${cls}`, title: label, 'aria-label': label }, cls === 'end' ? label : h('canvas', { 'aria-hidden': 'true' }));
      b.addEventListener('click', () => go(target));
      return b;
    };
    strip.append(mk('Embed', 0, 'end'));
    for (let l = 0; l < L; l++) strip.append(mk(`Layer ${l}`, 1 + 6 * l, 'layer'));
    strip.append(mk('Scores', last(), 'end'));
    const pills = h('div', { class: 'tv-pills' });
    STAGES.forEach((st, i) => {
      const b = h('button', { class: 'tv-pill' }, `${i + 1}. ${st}`);
      b.addEventListener('click', () => { const w = where(pos); const l = w.kind === 'layer' ? w.l : (w.kind === 'final' ? L - 1 : 0); go(1 + 6 * l + i); });
      pills.append(b);
    });
    const card = h('div', { class: 'tv-card', 'aria-live': 'polite' });
    const T = S.run.ids.length;
    el.append(
      h('div', { class: 'tv-top' },
        h('div', { class: 'tv-nav' }, back, next),
        h('label', { class: 'tv-layer' }, 'Layer ', slider, sliderOut),
        h('span', { class: 'tv-follow' }, 'Your prompt is ', dt(fmtInt(T)), ' tokens, so the matrix is ', dt(fmtInt(T)), ' × ', dn(fmtInt(d().D)))),
      h('p', { class: 'mx-striplabel' }, 'The prompt matrix as each layer hands it on. Same shape every time; only the numbers change. Click one to open that layer.'),
      strip, pills, card);
    ui = { back, next, slider, sliderOut, strip, pills, card, stripDrawn: null };
    render();
  }

  function go(p) {
    pos = Math.max(0, Math.min(last(), p));
    render();
  }

  // ---------------- data
  async function ensureBase() {
    if (base?.run === S.run || busy || S.running || !S.loaded || !visible) return;
    busy = true; failed = null;
    const run = S.run;
    try {
      const r = await ctx.engine.call('inspect', { ids: Array.from(run.ids) });
      if (S.run === run) { base = { run, T: r.T, resid: r.resid, guesses: r.guesses }; layers = new Map(); }
    } catch (e) { failed = e.message; }
    busy = false;
    render();
  }

  async function ensureLayer(l) {
    if (!base || layers.has(l) || busy) return;
    busy = true;
    const run = base.run;
    try {
      const r = await ctx.engine.call('inspectLayer', { ids: Array.from(run.ids), layer: l });
      if (base?.run === run) {
        layers.set(l, r);
        if (layers.size > KEEP_LAYERS) layers.delete(layers.keys().next().value);
      }
    } catch (e) { failed = e.message; }
    busy = false;
    render();
  }

  function drawStrip() {
    if (!base || ui.stripDrawn === base) return;
    ui.stripDrawn = base;
    const { D, L } = d(), T = base.T;
    const cells = [...ui.strip.children];
    for (let l = 0; l < L; l++) {
      const m = base.resid.subarray((l + 1) * T * D, (l + 2) * T * D);
      drawGrid(cells[l + 1].firstChild, m, T, D, diverging, robust(m));
    }
  }

  // ---------------- render
  function render() {
    if (!ui) return;
    const { D, L } = d();
    const w = where(pos);
    ui.back.disabled = pos === 0;
    ui.next.disabled = pos === last();
    const layer = w.kind === 'layer' ? w.l : w.kind === 'final' ? L - 1 : 0;
    ui.slider.value = layer;
    ui.sliderOut.textContent = String(layer);
    [...ui.strip.children].forEach((c, i) => {
      const on = (w.kind === 'embed' && i === 0) || (w.kind === 'final' && i === L + 1) || (w.kind === 'layer' && i === w.l + 1);
      const done = w.kind === 'final' || (w.kind === 'layer' && i > 0 && i <= w.l);
      c.classList.toggle('on', on);
      c.classList.toggle('done', done && !on);
    });
    [...ui.pills.children].forEach((b, i) => {
      b.classList.toggle('on', w.kind === 'layer' && w.s === i);
      b.disabled = w.kind !== 'layer';
    });
    const c = ui.card;
    c.textContent = '';
    const wait = t => c.append(h('p', { class: 'empty', style: { margin: 0 } }, t));
    if (failed) return wait(`Couldn’t compute the matrices: ${failed}`);
    if (base?.run !== S.run) {
      if (S.running) return wait('The matrices appear when generation finishes.');
      ensureBase();
      return wait(`Running your prompt through all ${L} layers as one matrix…`);
    }
    drawStrip();
    const T = base.T, rh = rowH(T);
    const head2 = (t, sub) => c.append(h('h4', { class: 'tv-title' }, t), h('p', { class: 'tv-plain' }, ...[].concat(sub)));

    if (w.kind === 'embed') {
      const X = base.resid.subarray(0, T * D);
      head2('Start: stack the embeddings into one matrix', [`Each of your ${T} token IDs picks its row of the ${fmtInt(d().V)} × ${fmtInt(D)} embedding table (chapter 3). Stacked, they make one matrix: `, dt(fmtInt(T)), ' × ', dn(fmtInt(D)), '. One row per token, one column per number. This is what flows through the model: every layer takes in a ', dt(fmtInt(T)), ' × ', dn(fmtInt(D)), ' matrix and hands on one of exactly the same shape.']);
      c.append(mrow(tokCol(T, rh), mx(X, T, D, { rh, name: 'X', shape: shp(dt(T), dn(D)), info: (r, col) => `<b>${esc(tokText(r))}</b> (ID ${ids()[r]}), number ${col}: ${fmt(X[r * D + col])}` }), lenCol(X, T, D, rh)));
      c.append(say('Chapter 5 followed one row of this matrix, the last one. Here we follow all of them. Nothing connects the rows yet: each row only knows its own token, not its neighbours or its position.'));
      return;
    }

    if (w.kind === 'final') return renderFinal(c, T, rh, head2);

    const Ld = layers.get(w.l);
    if (!Ld) { ensureLayer(w.l); return wait(`Computing layer ${w.l}…`); }
    const title = `Layer ${w.l} · step ${w.s + 1} of 6: `;
    [stageNorm1, stageQKV, stageAttn, stageCombine, stageNorm2, stageFFN][w.s](c, Ld, w.l, T, rh, title, head2);
    c.append(guessBox(w.l));
  }

  // ---------------- the six steps
  function lengthsSay(before, after, T, D) {
    const lb = Array.from({ length: T }, (_, t) => rowLen(before, t, D));
    const la = Array.from({ length: T }, (_, t) => rowLen(after, t, D));
    const big = lb.indexOf(Math.max(...lb));
    const med = [...lb].sort((a, b) => a - b)[T >> 1];
    let s = `Going in, the rows’ lengths range from ${Math.min(...lb).toFixed(1)} to ${Math.max(...lb).toFixed(1)}`;
    if (lb[big] > 4 * med) s += `; the longest, “${tokText(big)}”, is ${(lb[big] / med).toFixed(0)}× a typical row (that oversized row is the attention sink from chapter 6)`;
    return `${s}. Coming out, they range from ${Math.min(...la).toFixed(1)} to ${Math.max(...la).toFixed(1)}.`;
  }

  function stageNorm1(c, Ld, l, T, rh, title, head2) {
    const { D } = d();
    head2(title + 'normalize every row', `Each row is rescaled on its own to a standard size, then every column is multiplied by a learned factor (this layer’s ${D} norm weights). Rows don’t look at each other here. This keeps the numbers from growing out of control as ${d().L} layers keep adding to them.`);
    c.append(mrow(tokCol(T, rh), mx(Ld.x, T, D, { rh, name: 'X', shape: shp(dt(T), dn(D)) }), lenCol(Ld.x, T, D, rh), op('normalize each row →'),
      mx(Ld.h, T, D, { rh, name: 'H', shape: shp(dt(T), dn(D)) }), lenCol(Ld.h, T, D, rh)));
    c.append(say(lengthsSay(Ld.x, Ld.h, T, D)));
  }

  function stageQKV(c, Ld, l, T, rh, title, head2) {
    const { D, H, KVH, HD } = d();
    const kv = KVH * HD, p = `model.layers.${l}.self_attn.`;
    head2(title + 'make Q, K and V for every row', ['Three matrix multiplications with this layer’s weights. With one row per token they are written ', h('b', {}, 'H × Wᵀ'), ': the weight matrix is used flipped (transposed) so it lines up with each row’s numbers. It is the same arithmetic as chapter 5’s W × x, done for all ', dt(T), ' rows at once, which is why prefill reads each weight only once for the whole prompt (chapter 11).']);
    c.append(mrow(tokCol(T, rh), mx(Ld.h, T, D, { rh, name: 'H', shape: shp(dt(T), dn(D)) }), op('×'), wt('Wq', p + 'q_proj.weight', D, H * HD), op('='),
      mx(Ld.qPre, T, H * HD, { rh, heads: H, name: 'Q', shape: shp(dt(T), dm(H * HD)), info: (r, col) => `<b>${esc(tokText(r))}</b>: query, head ${Math.floor(col / HD)}, number ${col % HD}<br>${fmt(Ld.qPre[r * H * HD + col])}` })));
    c.append(mrow(tokCol(T, rh), mx(Ld.h, T, D, { rh, name: 'H', shape: shp(dt(T), dn(D)) }), op('×'), wt('Wk', p + 'k_proj.weight', D, kv), op('='),
      mx(Ld.kPre, T, kv, { rh, heads: KVH, name: 'K', shape: shp(dt(T), dm(kv)), flex: 360, info: (r, col) => `<b>${esc(tokText(r))}</b>: key, head ${Math.floor(col / HD)}, number ${col % HD}<br>${fmt(Ld.kPre[r * kv + col])}` }),
      op('and ×'), wt('Wv', p + 'v_proj.weight', D, kv), op('='),
      mx(Ld.v, T, kv, { rh, heads: KVH, name: 'V', shape: shp(dt(T), dm(kv)), flex: 360, info: (r, col) => `<b>${esc(tokText(r))}</b>: value, head ${Math.floor(col / HD)}, number ${col % HD}<br>${fmt(Ld.v[r * kv + col])}` })));
    c.append(say([`The white lines split the columns into heads: Q has ${H} heads of ${HD} numbers; K and V have ${KVH}, each shared by ${H / KVH} query heads. Before attention, every row of Q and K is rotated according to its position (chapter 4); V is not. K and V, `, dt(T), ` × ${kv} each, are exactly what this layer writes into the KV cache.`]));
  }

  function headKind(A, T, hh) {
    if (T < 3) return '';
    let sink = 0, prev = 0, self = 0;
    for (let t = 1; t < T; t++) {
      const o = (hh * T + t) * T;
      sink += A[o]; prev += A[o + t - 1]; self += A[o + t];
    }
    sink /= T - 1; prev /= T - 1; self /= T - 1;
    if (sink > 0.5) return 'mostly the first token';
    if (prev > 0.35) return 'the previous token';
    if (self > 0.35) return 'itself';
    return 'spread out';
  }

  function stageAttn(c, Ld, l, T, rh, title, head2) {
    const { D, H, KVH, HD } = d();
    const kv = KVH * HD, g = Math.floor(head / (H / KVH));
    const A = Ld.attn;
    head2(title + 'attention: the rows talk', ['This is the only step where rows exchange information. Each head compares every row’s query with every row’s key: ', dt(T), ` × ${HD} times ${HD} × `, dt(T), ' gives a ', dt(T), ' × ', dt(T), ' table of scores, one for each pair of tokens. A token may only look at itself and earlier tokens, so the upper-right triangle is blocked: the future isn’t written yet. Softmax turns each row into percentages that add up to 100%, and each output row is that mix of the value rows.']);

    // pick a head
    const picker = h('div', { class: 'mx-heads', role: 'group', 'aria-label': 'Choose a head' });
    for (let hh = 0; hh < H; hh++) {
      const cv = h('canvas', { 'aria-hidden': 'true' });
      drawAttn(cv, A.subarray(hh * T * T, (hh + 1) * T * T), T, false);
      const b = h('button', { class: `mx-head${hh === head ? ' on' : ''}`, 'aria-pressed': String(hh === head) }, cv, h('span', {}, `head ${hh}`), h('small', {}, headKind(A, T, hh)));
      b.addEventListener('click', () => { head = hh; render(); });
      picker.append(b);
    }
    c.append(h('p', { class: 'tv-sub' }, `The ${H} heads of layer ${l}. Each square is one head’s ${T} × ${T} table after softmax; the label says where it mostly looks. Choose one:`), picker);

    // q_h · k_hᵀ → scores → softmax → weights
    const qh = cols(Ld.q, T, D, head * HD, HD), kh = cols(Ld.k, T, kv, g * HD, HD), vh = cols(Ld.v, T, kv, g * HD, HD);
    const raw = new Float32Array(T * T);
    for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) {
      let s = 0;
      for (let e = 0; e < HD; e++) s += qh[i * HD + e] * kh[j * HD + e];
      raw[i * T + j] = s / Math.sqrt(HD);
    }
    const Ah = A.subarray(head * T * T, (head + 1) * T * T);
    const khT = new Float32Array(HD * T);
    for (let j = 0; j < T; j++) for (let e = 0; e < HD; e++) khT[e * T + j] = kh[j * HD + e];
    c.append(h('p', { class: 'tv-sub' }, `Head ${head}, step by step (it shares key/value head ${g}):`));
    c.append(mrow(tokCol(T, rh),
      mx(qh, T, HD, { rh, name: `Q, head ${head}`, shape: shp(dt(T), dn(HD)), flex: 140 }), op('×'),
      mx(khT, HD, T, { height: T * rh, name: `K, head ${g}, flipped`, shape: shp(dn(HD), dt(T)), flex: 140, info: (r, col) => `key of <b>${esc(tokText(col))}</b>, number ${r}: ${fmt(khT[r * T + col])}` }),
      op(`÷ √${HD} =`), attnFig(raw, T, rh, 'scores', true, Ah, raw), op('softmax each row →'), attnFig(Ah, T, rh, 'weights', false, Ah, raw)));
    c.append(zoom(Ah, raw, T));
    const outh = cols(Ld.mix, T, D, head * HD, HD);
    c.append(mrow(tokCol(T, rh), attnFig(Ah, T, rh, 'weights', false, Ah, raw), op('×'),
      mx(vh, T, HD, { rh, name: `V, head ${g}`, shape: shp(dt(T), dn(HD)), flex: 140 }), op('='),
      mx(outh, T, HD, { rh, name: `head ${head}’s result`, shape: shp(dt(T), dn(HD)), flex: 140 }),
      op(`all ${H} heads side by side →`),
      mx(Ld.mix, T, D, { rh, heads: H, hi: head, name: 'mixed', shape: shp(dt(T), dn(D)), info: (r, col) => `<b>${esc(tokText(r))}</b>: head ${Math.floor(col / HD)}’s result, number ${col % HD}<br>${fmt(Ld.mix[r * D + col])}` })));
    const lastRow = Ah.subarray((T - 1) * T, T * T);
    const top = Array.from(lastRow.keys()).sort((a, b) => lastRow[b] - lastRow[a]).slice(0, 3);
    const names = top.map(j => `“${tokText(j)}” (${Math.round(lastRow[j] * 100)}%)`);
    c.append(say(`In the bottom row, the last token “${tokText(T - 1)}” splits head ${head}’s attention mostly between ${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}. Hover over any square to read who looks at whom.`));
  }

  function cols(a, T, stride, o, n) {
    const out = new Float32Array(T * n);
    for (let t = 0; t < T; t++) out.set(a.subarray(t * stride + o, t * stride + o + n), t * n);
    return out;
  }

  function drawAttn(cv, M, T, rawScores) {
    cv.width = T; cv.height = T;
    const g = cv.getContext('2d'), img = g.createImageData(T, T);
    let sc = 1;
    if (rawScores) {
      const vals = [];
      for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) vals.push(M[i * T + j]);
      sc = robust(Float32Array.from(vals), 0.99);
    }
    for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
      const k = (i * T + j) * 4;
      const col = j > i ? MASKED : rawScores ? diverging(M[i * T + j] / sc) : sequential(Math.sqrt(M[i * T + j]));
      img.data[k] = col[0]; img.data[k + 1] = col[1]; img.data[k + 2] = col[2]; img.data[k + 3] = 255;
    }
    g.putImageData(img, 0, 0);
  }

  const attnInfo = (Ah, raw, T) => (r, col) => col > r
    ? `<b>${esc(tokText(r))}</b> can’t look at <b>${esc(tokText(col))}</b>: it comes later`
    : `<b>${esc(tokText(r))}</b> looks at <b>${esc(tokText(col))}</b><br>score ${raw[r * T + col].toFixed(2)} → ${(Ah[r * T + col] * 100).toFixed(1)}%`;

  function attnFig(M, T, rh, name, rawScores, Ah, raw) {
    const cv = h('canvas', { class: 'mx-cv', 'aria-hidden': 'true' });
    drawAttn(cv, M, T, rawScores);
    cv.style.height = `${T * rh}px`;
    hover(cv, T, T, attnInfo(Ah, raw, T));
    return h('figure', { class: 'mx', style: { flex: `${Math.max(T * 3, 120)} 1 0` } },
      h('figcaption', {}, h('b', {}, name), ' ', shp(dt(T), dt(T))), h('div', { class: 'mx-box' }, cv));
  }

  // The chosen head's table, big, with every token written on both sides.
  function zoom(Ah, raw, T) {
    const cell = Math.max(5, Math.min(16, Math.floor(440 / T)));
    const cv = h('canvas', { class: 'mx-zcv', 'aria-hidden': 'true' });
    drawAttn(cv, zoomRaw ? raw : Ah, T, zoomRaw);
    cv.style.width = cv.style.height = `${T * cell}px`;
    hover(cv, T, T, attnInfo(Ah, raw, T));
    const top = h('div', { class: 'mx-zcols' });
    const left = h('div', { class: 'mx-zrows' });
    for (const el of [top, left]) el.style.setProperty('--cell', `${cell}px`);
    if (cell >= 8) for (let t = 0; t < T; t++) { top.append(h('span', {}, tokText(t))); left.append(h('span', {}, tokText(t))); }
    const tg = (label, isRaw) => {
      const b = h('button', { class: `tv-pill${zoomRaw === isRaw ? ' on' : ''}`, 'aria-pressed': String(zoomRaw === isRaw) }, label);
      b.addEventListener('click', () => { zoomRaw = isRaw; render(); });
      return b;
    };
    return h('details', { class: 'mx-zoom', open: true },
      h('summary', {}, `Head ${head}’s table, large, with every token labelled`),
      h('div', { class: 'mx-ztools' }, tg('After softmax (%)', false), tg('Raw scores', true),
        h('span', { class: 'dim' }, 'Rows: the token doing the looking. Columns: the token it looks at. Grey: blocked, because it comes later.')),
      h('div', { class: 'mx-zscroll' }, h('div', { class: 'mx-zgrid' }, h('span', {}), top, left, cv)));
  }

  function stageCombine(c, Ld, l, T, rh, title, head2) {
    const { D, H } = d();
    head2(title + 'combine the heads, then add', ['One more weight matrix, Wo, blends the heads’ side-by-side results into a correction for every row: ', dt(T), ' × ', dn(D), ' times Woᵀ gives ', dt(T), ' × ', dm(D), '. Then the correction is ', h('b', {}, 'added'), ' to X, number by number. Nothing is overwritten: each layer only adds to this running matrix, which is called the residual stream.']);
    c.append(mrow(tokCol(T, rh), mx(Ld.mix, T, D, { rh, heads: H, name: 'mixed', shape: shp(dt(T), dn(D)) }), op('×'), wt('Wo', `model.layers.${l}.self_attn.o_proj.weight`, D, D), op('='),
      mx(Ld.o, T, D, { rh, name: 'correction', shape: shp(dt(T), dm(D)) })));
    c.append(mrow(tokCol(T, rh), mx(Ld.x, T, D, { rh, name: 'X', shape: shp(dt(T), dn(D)) }), op('+'),
      mx(Ld.o, T, D, { rh, name: 'correction', shape: shp(dt(T), dn(D)) }), op('='),
      mx(Ld.xmid, T, D, { rh, name: 'X′', shape: shp(dt(T), dn(D)) })));
    c.append(say(changeSay(Ld.x, Ld.o, T, D, 'Attention')));
  }

  function changeSay(x, corr, T, D, who) {
    const r = Array.from({ length: T }, (_, t) => rowLen(corr, t, D) / (rowLen(x, t, D) || 1)).sort((a, b) => a - b);
    const p = v => `${Math.round(v * 100)}%`;
    const big = r[T >> 1] > 1 ? ' That is more than the rows themselves: early on the rows are still small, so the first layers rewrite them almost completely. Later layers make much smaller edits.' : '';
    return `${who} changed a typical row by ${p(r[T >> 1])} of its length (${p(r[0])} to ${p(r[T - 1])} across the rows).${big}`;
  }

  function stageNorm2(c, Ld, l, T, rh, title, head2) {
    const { D } = d();
    head2(title + 'normalize again', 'Same as step 1, with this layer’s second set of norm weights: rescale every row before the feed-forward network.');
    c.append(mrow(tokCol(T, rh), mx(Ld.xmid, T, D, { rh, name: 'X′', shape: shp(dt(T), dn(D)) }), lenCol(Ld.xmid, T, D, rh), op('normalize each row →'),
      mx(Ld.h2, T, D, { rh, name: 'H', shape: shp(dt(T), dn(D)) }), lenCol(Ld.h2, T, D, rh)));
    c.append(say(lengthsSay(Ld.xmid, Ld.h2, T, D)));
  }

  function stageFFN(c, Ld, l, T, rh, title, head2) {
    const { D, I, L } = d();
    const p = `model.layers.${l}.mlp.`;
    head2(title + 'feed-forward, then add', ['Each row is processed on its own again, with no talking between rows. Every row is widened to ', dm(fmtInt(I)), ' numbers twice (gate and up), the two are combined number by number as silu(gate) × up, and the result is narrowed back to ', dm(D), ' and added to the running matrix. Attention moved information between tokens; this step works on what each token now holds. It has two thirds of the layer’s weights.']);
    c.append(mrow(tokCol(T, rh), mx(Ld.h2, T, D, { rh, name: 'H', shape: shp(dt(T), dn(D)), flex: 576 }), op('×'), wt('Wgate', p + 'gate_proj.weight', D, I), op('='),
      mx(Ld.gate, T, I, { rh, name: 'gate', shape: shp(dt(T), dm(fmtInt(I))), flex: 700 }), op('and ×'), wt('Wup', p + 'up_proj.weight', D, I), op('='),
      mx(Ld.up, T, I, { rh, name: 'up', shape: shp(dt(T), dm(fmtInt(I))), flex: 700 })));
    c.append(mrow(tokCol(T, rh), op('silu(gate) × up ='),
      mx(Ld.act, T, I, { rh, name: 'combined', shape: shp(dt(T), dn(fmtInt(I))), flex: 900 }), op('×'), wt('Wdown', p + 'down_proj.weight', I, D), op('='),
      mx(Ld.down, T, D, { rh, name: 'correction', shape: shp(dt(T), dm(D)), flex: 576 })));
    c.append(mrow(tokCol(T, rh), mx(Ld.xmid, T, D, { rh, name: 'X′', shape: shp(dt(T), dn(D)) }), op('+'),
      mx(Ld.down, T, D, { rh, name: 'correction', shape: shp(dt(T), dn(D)) }), op('='),
      mx(Ld.xout, T, D, { rh, name: l + 1 < L ? `X into layer ${l + 1}` : 'X after the last layer', shape: shp(dt(T), dn(D)) })));
    let small = 0;
    const big = robust(Ld.act, 0.99);
    for (const v of Ld.act) if (Math.abs(v) < 0.1 * big) small++;
    c.append(say(`${Math.round((small / Ld.act.length) * 100)}% of the combined numbers are close to zero (under a tenth of the biggest): the gate lets only some through, and a different set for each row. ${changeSay(Ld.xmid, Ld.down, T, D, 'The feed-forward network')}${l + 1 < L ? ` The result is the matrix layer ${l + 1} starts from.` : ''}`));
  }

  // ---------------- the end: every row makes a guess
  function renderFinal(c, T, rh, head2) {
    const { D, V, L } = d();
    const X = base.resid.subarray(L * T * D, (L + 1) * T * D);
    head2('Finish: every row scores every token', [`After layer ${L - 1}, each row is normalized once more and multiplied by the embedding table, flipped: `, dt(T), ' × ', dn(D), ' times ', dn(D), ' × ', dm(fmtInt(V)), ' gives ', dt(T), ' × ', dm(fmtInt(V)), ', a score for every vocabulary entry in every row. Softmax turns each row of scores into percentages.']);
    c.append(mrow(tokCol(T, rh), mx(X, T, D, { rh, name: `X after layer ${L - 1}`, shape: shp(dt(T), dn(D)) }), lenCol(X, T, D, rh), op('normalize, × table ='),
      h('figure', { class: 'mx', style: { flex: '420 1 0' } }, h('figcaption', {}, h('b', {}, 'scores'), ' ', shp(dt(T), dm(fmtInt(V)))),
        h('div', { class: 'mx-huge', style: { height: `${T * rh}px` } }, h('span', {}, `${fmtInt(T * V)} numbers, too wide to draw`)))));

    // each row's guess for the token after it
    const run = S.run;
    const nextOf = t => (t + 1 < T ? run.ids[t + 1] : run.steps[0]?.token);
    let right = 0;
    const list = h('div', { class: 'mx-guesses' });
    list.append(h('div', { class: 'mx-g head' }, h('span', {}, 'Row (token)'), h('span', {}, 'Its top guess for the next token'), h('span', {}, 'What actually came next')));
    for (let t = 0; t < T; t++) {
      const g = base.guesses[t][0], nx = nextOf(t);
      const ok = nx !== undefined && g.id === nx;
      if (ok && t < T - 1) right++;
      list.append(h('div', { class: `mx-g${t === T - 1 ? ' last' : ''}${ok ? ' ok' : ''}` },
        h('code', {}, tokText(t)),
        h('span', {}, h('code', {}, vis(S.tok.tokenText(g.id))), ` ${Math.round(g.p * 100)}%`),
        h('span', {}, nx === undefined ? '' : h('code', {}, vis(S.tok.tokenText(nx))), t === T - 1 ? h('small', { class: 'dim' }, ' (generated)') : null, ok ? h('b', { class: 'mx-tick' }, ' ✓') : null)));
    }
    c.append(say(['Every row guesses the token that comes after it. On your prompt’s own next tokens the model got ', h('b', {}, `${right} of ${T - 1}`), ' right. That is how it was trained: every row is graded at once against the real next token, and backpropagation (chapter 5) adjusts the weights. When generating, only the ', h('b', {}, 'last row'), ' matters: its guess becomes the next token.']));
    c.append(h('details', { class: 'mx-gwrap' }, h('summary', {}, `Every row’s guess (${T} rows)`), list));

    const step0 = run.steps[0];
    if (step0?.logits) {
      const top = softmaxTop(step0.logits, 10);
      const box = h('div', { class: 'tv-top10' });
      const max = top[0].p;
      for (const t of top) {
        box.append(h('div', { class: `tv-score${t.id === step0.token ? ' chosen' : ''}` },
          h('code', {}, vis(S.tok.tokenText(t.id))),
          h('span', { class: 'lg' }, `score ${t.logit.toFixed(2)}`),
          h('span', { class: 'b' }, h('span', { style: { width: `${(t.p / max) * 100}%` } })),
          h('span', { class: 'p' }, `${(t.p * 100).toFixed(1)}%`)));
      }
      c.append(h('p', { class: 'tv-sub' }, `The last row, “${tokText(T - 1)}”: its 10 highest scores as percentages (before temperature and the other sampling settings):`), box,
        h('p', { class: 'caption' }, `Highlighted: the token the sampler picked. That one token then goes through all ${L} layers as a single new row, reading the other rows’ keys and values from the KV cache instead of recomputing them (chapter 10).`));
    }
  }

  function guessBox(l) {
    const box = h('div', { class: 'tv-guess' });
    if (!lens) {
      box.append(h('span', { class: 'dim' }, S.running ? 'The last row’s best guess per layer appears when generation finishes.' : 'Working out the last row’s best guess at each layer…'));
      fetchLens();
      return box;
    }
    box.append(h('b', {}, `The last row’s best guess after layer ${l}: `),
      ...lens[l].slice(0, 3).map(c => h('span', { class: 'tv-gtok' }, h('code', {}, vis(S.tok.tokenText(c.id))), ` ${(c.p * 100).toFixed(0)}%`)),
      h('small', { class: 'dim' }, ' (the logit lens from chapter 7: what the model would say if it stopped here)'));
    return box;
  }

  async function fetchLens() {
    if (lensBusy || S.running || !S.loaded || lensFor === S.run) return;
    lensBusy = true;
    try {
      const r = await ctx.engine.call('lens');
      lens = r.layers; lensFor = S.run;
      render();
    } catch { /* ignore */ } finally { lensBusy = false; }
  }

  // Only run the side pass once the chapter is close to the screen.
  new IntersectionObserver(es => {
    visible = es.some(e => e.isIntersecting);
    if (visible && ui && base?.run !== S.run) render();
  }, { rootMargin: '600px 0px' }).observe($('#tracer'));

  const reset = () => { base = null; layers = new Map(); lens = null; lensFor = null; failed = null; };
  return {
    update(reason) {
      if (reason === 'model') { reset(); build(); }
      if (reason === 'prefill') { reset(); pos = Math.min(pos, last()); build(); }
      if (reason === 'idle' && ui) render();
    },
  };
}
