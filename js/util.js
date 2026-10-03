// Small DOM, formatting and drawing helpers shared by every chapter.

export const $ = (sel, root = document) => root.querySelector(sel);

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'html') el.innerHTML = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export const fmtInt = n => Math.round(n).toLocaleString('en-US');

export function fmtBytes(b, digits = 1) {
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (Math.abs(b) >= 1000 && i < u.length - 1) { b /= 1000; i++; }
  return `${i === 0 ? Math.round(b) : b.toFixed(b >= 100 ? 0 : digits)} ${u[i]}`;
}

export function fmtMs(ms) {
  if (ms === 0) return '0 ms';
  if (ms < 0.001) return `${(ms * 1e6).toFixed(0)} ns`;
  if (ms < 1) return `${(ms * 1000).toFixed(ms < 0.01 ? 2 : 0)} µs`;
  if (ms < 1000) return `${ms.toFixed(ms < 10 ? 1 : 0)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function fmtFlops(f) {
  const u = ['', 'K', 'M', 'G', 'T', 'P'];
  let i = 0;
  while (f >= 1000 && i < u.length - 1) { f /= 1000; i++; }
  return `${f.toFixed(f >= 100 ? 0 : 1)} ${u[i]}FLOP`;
}

export const pct = (p, d = 1) => `${(p * 100).toFixed(d)}%`;

// Show whitespace inside token strings so " the" and "the" look different.
export function vis(s) {
  return s.replace(/ /g, '␣').replace(/\n/g, '↵').replace(/\t/g, '⇥').replace(/\r/g, '␍');
}

// ---------- colour ----------
const NEG = [46, 91, 186], MID = [244, 244, 244], POS = [196, 67, 43];
const LO = [244, 246, 249], HI = [20, 33, 58];

// v in [-1, 1] -> cobalt / white / vermilion
export function diverging(v) {
  v = Math.max(-1, Math.min(1, v));
  const [a, b] = v < 0 ? [MID, NEG] : [MID, POS];
  const t = Math.abs(v) ** 0.8;
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

// v in [0, 1] -> paper to navy
export function sequential(v) {
  v = Math.max(0, Math.min(1, v));
  return [LO[0] + (HI[0] - LO[0]) * v, LO[1] + (HI[1] - LO[1]) * v, LO[2] + (HI[2] - LO[2]) * v];
}

export const rgb = c => `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;

// Draw a rows×cols array as pixels on a canvas (one pixel per value, CSS scales it).
export function drawGrid(canvas, data, rows, cols, color, scale) {
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(cols, rows);
  for (let i = 0; i < rows * cols; i++) {
    const c = color(data[i] / scale);
    img.data[i * 4] = c[0]; img.data[i * 4 + 1] = c[1]; img.data[i * 4 + 2] = c[2]; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// A vector as a one-row barcode.
export function drawVector(canvas, vec, scale) {
  if (!scale) {
    let s = 0;
    for (const v of vec) s += v * v;
    scale = 2.5 * Math.sqrt(s / vec.length) || 1;
  }
  drawGrid(canvas, vec, 1, vec.length, diverging, scale);
}

export function rms(arr) {
  let s = 0;
  for (const v of arr) s += v * v;
  return Math.sqrt(s / arr.length);
}

// ---------- tooltip ----------
let tipEl = null;
export function tip(evt, html) {
  tipEl ??= document.getElementById('tooltip');
  if (!html) { tipEl.hidden = true; return; }
  tipEl.innerHTML = html;
  tipEl.hidden = false;
  const r = tipEl.getBoundingClientRect();
  let x = evt.clientX + 14, y = evt.clientY + 14;
  if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - 14;
  tipEl.style.left = `${x}px`;
  tipEl.style.top = `${y}px`;
}

export const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

export function debounce(f, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => f(...a), ms); };
}

// ---------- sampling ----------
// Returns the candidate list after temperature, top-k and top-p, with renormalised probs.
export function samplingDistribution(logits, { temperature, topK, topP }, keep = 40) {
  const n = logits.length;
  // Sort a shortlist: the top `pool` logits are all that can matter.
  const pool = Math.max(keep, topK || 0, 200);
  const idx = [];
  let minIn = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = logits[i];
    if (idx.length < pool) { idx.push(i); if (idx.length === pool) { idx.sort((a, b) => logits[b] - logits[a]); minIn = logits[idx[pool - 1]]; } }
    else if (v > minIn) {
      let j = pool - 1;
      while (j > 0 && logits[idx[j - 1]] < v) { idx[j] = idx[j - 1]; j--; }
      idx[j] = i;
      minIn = logits[idx[pool - 1]];
    }
  }
  if (idx.length < pool) idx.sort((a, b) => logits[b] - logits[a]);
  const max = logits[idx[0]];
  // Raw softmax (T = 1) over the full vocabulary, for reference.
  let zRaw = 0;
  for (let i = 0; i < n; i++) zRaw += Math.exp(logits[i] - max);
  const greedy = temperature <= 0.001;
  const T = greedy ? 1 : temperature;
  let zT = 0;
  for (let i = 0; i < n; i++) zT += Math.exp((logits[i] - max) / T);
  const cands = idx.map((id, rank) => ({
    id, rank, logit: logits[id],
    pRaw: Math.exp(logits[id] - max) / zRaw,
    pT: Math.exp((logits[id] - max) / T) / zT,
    kept: true,
  }));
  // Filters
  let cum = 0;
  for (const c of cands) {
    if (greedy) c.kept = c.rank === 0;
    else {
      if (topK > 0 && c.rank >= topK) c.kept = false;
      if (topP < 1 && cum >= topP) c.kept = false;
      cum += c.pT;
    }
  }
  const z = cands.reduce((a, c) => a + (c.kept ? c.pT : 0), 0);
  for (const c of cands) c.p = c.kept ? c.pT / z : 0;
  return { cands, shown: cands.slice(0, keep), greedy };
}

export function sampleFrom(dist, rand = Math.random) {
  if (dist.greedy) return dist.cands[0];
  let r = rand(), acc = 0;
  for (const c of dist.cands) {
    if (!c.kept) continue;
    acc += c.p;
    if (r <= acc) return c;
  }
  return dist.cands.find(c => c.kept);
}
