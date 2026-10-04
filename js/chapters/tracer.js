// 8. Follow one token through the whole model: embedding → every stage of every
// layer → final scores → softmax. Same layout at every stage; only numbers change.

import { $, h, fmtInt, vis, drawVector, softmaxTop } from '../util.js';

const STAGES = [
  { key: 'norm1', short: 'Normalize' },
  { key: 'qkv', short: 'Q, K, V' },
  { key: 'attn', short: 'Attention' },
  { key: 'oadd', short: 'Combine + add' },
  { key: 'norm2', short: 'Normalize' },
  { key: 'ffn', short: 'Feed-forward + add' },
];

export default function tracer(ctx) {
  const { S } = ctx;
  let pos = 0;           // 0 = embedding, 1..6L = layer stages, 6L+1 = final
  let lens = null, lensFor = null, lensBusy = false;
  let rowCache = new Map(); // weight row 0 of q_proj per layer, for the worked example
  let ui = null;

  const d = () => ctx.dims();
  const last = () => 6 * d().L + 1;
  const where = p => (p === 0 ? { kind: 'embed' } : p === last() ? { kind: 'final' } : { kind: 'layer', l: Math.floor((p - 1) / 6), s: (p - 1) % 6 });
  const trace = () => S.run?.prefill?.trace;

  const fmt = v => (v >= 0 ? ' ' : '−') + Math.abs(v).toFixed(3);
  const len = (a, o, n) => { let s = 0; for (let i = 0; i < n; i++) s += a[o + i] ** 2; return Math.sqrt(s); };

  // One vector: name, shape, first numbers, colour strip, length.
  function vecRow(label, arr, o, n, note) {
    const cv = h('canvas', { class: 'hm', 'aria-hidden': 'true' });
    drawVector(cv, arr.subarray(o, o + n));
    const first = Array.from(arr.subarray(o, o + 8), fmt).join('  ');
    return h('div', { class: 'tv-row' },
      h('div', { class: 'tv-name' }, h('b', {}, label), h('small', {}, `${fmtInt(n)} numbers`)),
      h('div', { class: 'tv-data' },
        h('code', { class: 'tv-nums' }, `[${first}  …]`),
        cv,
        h('small', { class: 'tv-note' }, `length ${len(arr, o, n).toFixed(1)}${note ? ` · ${note}` : ''}`)));
  }

  const shape = (a, b, c) => h('p', { class: 'tv-shape' }, ...a, h('span', { class: 'sr-op' }, ' × '), ...b, h('span', { class: 'sr-op' }, ' → '), ...c);
  const dm = v => h('span', { class: 'dm' }, v), dn = v => h('span', { class: 'dn' }, v);

  function build() {
    const el = $('#tracer');
    el.textContent = '';
    ui = null;
    if (!d()) return;
    const tr = trace();
    if (!tr?.tok) {
      el.append(h('p', { class: 'empty', style: { margin: 0 } }, S.loaded ? 'Generate something, then come back: the tracer replays the last run.' : 'Load the model and generate to trace a token.'));
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
    // progress strip: embedding, 30 layers, scores
    const strip = h('div', { class: 'tv-strip', role: 'group', 'aria-label': 'Jump to a layer' });
    const mk = (label, target, cls) => {
      const b = h('button', { class: `tv-cell ${cls}`, title: label, 'aria-label': label }, cls === 'end' ? label : '');
      b.addEventListener('click', () => go(target));
      return b;
    };
    strip.append(mk('Embed', 0, 'end'));
    for (let l = 0; l < L; l++) strip.append(mk(`Layer ${l}`, 1 + 6 * l, 'layer'));
    strip.append(mk('Scores', last(), 'end'));
    const pills = h('div', { class: 'tv-pills' });
    STAGES.forEach((st, i) => {
      const b = h('button', { class: 'tv-pill' }, `${i + 1}. ${st.short}`);
      b.addEventListener('click', () => { const w = where(pos); const l = w.kind === 'layer' ? w.l : (w.kind === 'final' ? L - 1 : 0); go(1 + 6 * l + i); });
      pills.append(b);
    });
    const card = h('div', { class: 'tv-card', 'aria-live': 'polite' });
    const tokId = tr.tok && S.run.ids[S.run.ids.length - 1];
    el.append(
      h('div', { class: 'tv-top' },
        h('div', { class: 'tv-nav' }, back, next),
        h('label', { class: 'tv-layer' }, 'Layer ', slider, sliderOut),
        h('span', { class: 'tv-follow' }, 'Following ', h('code', {}, vis(S.tok.tokenText(tokId))), ` (token ${S.run.ids.length - 1} of your prompt)`)),
      strip, pills, card);
    ui = { back, next, slider, sliderOut, strip, pills, card };
    render();
  }

  function go(p) {
    pos = Math.max(0, Math.min(last(), p));
    render();
  }

  function render() {
    if (!ui) return;
    const tr = trace();
    const D = d().D, I = d().I, L = d().L, KVH = d().KVH, HD = d().HD, H = d().H;
    const kvDim = KVH * HD;
    const tk = tr.tok;
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
    const head = (t, sub) => c.append(h('h4', { class: 'tv-title' }, t), h('p', { class: 'tv-plain' }, sub));

    if (w.kind === 'embed') {
      head('Start: look up the embedding', `The token’s ID picks one row of the ${fmtInt(d().V)} × ${fmtInt(D)} embedding table (chapter 3). This vector of ${D} numbers is all the model knows about the token so far: nothing about its neighbours or its position yet.`);
      c.append(vecRow('x', tk.x, 0, D, 'the running vector, entering layer 0'));
      c.append(nextHint('Next, layer 0 begins. Every layer runs the same 6 steps with its own weights.'));
      return;
    }

    if (w.kind === 'final') {
      head('Finish: score every word, then softmax', `After the last layer, the vector is rescaled once more and compared with every row of the vocabulary table: one dot product per token, ${fmtInt(d().V)} in all (chapter 8). Each result is that token’s score. Softmax turns the scores into percentages.`);
      c.append(shape([dm('1'), ' × ', dn(fmtInt(D))], [dn(fmtInt(D)), ' × ', dm(fmtInt(d().V))], [dm('1'), ' × ', dm(fmtInt(d().V)), ' scores']));
      c.append(vecRow('final x', tk.final, 0, D, 'after the final RMSNorm'));
      const step0 = S.run.steps[0];
      if (step0?.logits) {
        const top = softmaxTop(step0.logits, 10);
        const list = h('div', { class: 'tv-top10' });
        const max = top[0].p;
        for (const t of top) {
          list.append(h('div', { class: `tv-score${t.id === step0.token ? ' chosen' : ''}` },
            h('code', {}, vis(S.tok.tokenText(t.id))),
            h('span', { class: 'lg' }, `score ${t.logit.toFixed(2)}`),
            h('span', { class: 'b' }, h('span', { style: { width: `${t.p / max * 100}%` } })),
            h('span', { class: 'p' }, `${(t.p * 100).toFixed(1)}%`)));
        }
        c.append(h('p', { class: 'tv-sub' }, 'The 10 highest scores, as percentages (before temperature and other sampling settings):'), list,
          h('p', { class: 'caption' }, 'Highlighted: the token the sampler actually picked as the first word of the answer. Then the whole journey repeats for that token, one layer at a time, using the KV cache (chapter 10).'));
      }
      return;
    }

    const l = w.l, s = w.s;
    const stageTitle = `Layer ${l} · step ${s + 1} of 6: `;
    const oD = l * D, oK = l * kvDim, oI = l * I;
    if (s === 0) {
      head(stageTitle + 'normalize', 'Rescale the vector to a standard size, then multiply each number by a learned factor. This keeps the numbers from growing out of control as 30 layers keep adding to them. Nothing is learned about context yet.');
      c.append(vecRow('x', tk.x, oD, D, 'coming into this layer'), arrow('rescale'), vecRow('h', tk.h, oD, D, 'normalized'));
    } else if (s === 1) {
      head(stageTitle + 'make the query, key and value', 'Three matrix multiplications, each with this layer’s own weights. The query asks “what am I looking for?”, the key says “what do I contain?”, and the value is what this token hands over when others attend to it (chapter 6). The key and value are also saved into the KV cache.');
      c.append(
        shape([dm('1'), ' × ', dn(fmtInt(D))], [dn(fmtInt(D)), ' × ', dm(fmtInt(D))], [dm('1'), ' × ', dm(fmtInt(D)), ' = q']),
        vecRow('q', tk.q, oD, D, `${H} heads × ${HD}`),
        vecRow('k', tk.k, oK, kvDim, `${KVH} heads × ${HD}, saved to the KV cache`),
        vecRow('v', tk.v, oK, kvDim, `${KVH} heads × ${HD}, saved to the KV cache`),
        worked(l));
    } else if (s === 2) {
      head(stageTitle + 'attention', `Each of the ${H} heads takes its 64-number slice of the query and compares it with the keys of every earlier token, including this one. The scores become percentages, and the head takes that mix of the earlier tokens’ values. The ${H} results sit side by side as one vector of ${D}.`);
      c.append(headsTable(tr, l));
      c.append(vecRow('mixed', tk.att, oD, D, `${H} heads’ results, side by side`));
    } else if (s === 3) {
      head(stageTitle + 'combine the heads, then add', 'One more matrix (o_proj) blends the heads’ results into a correction. The correction is then ADDED to the running vector. Nothing is overwritten: this is the residual stream (chapter 7).');
      c.append(
        shape([dm('1'), ' × ', dn(fmtInt(D))], [dn(fmtInt(D)), ' × ', dm(fmtInt(D))], [dm('1'), ' × ', dm(fmtInt(D)), ' correction']),
        vecRow('x', tk.x, oD, D, 'before'), arrow('+'),
        vecRow('correction', tk.o, oD, D, 'from attention'), arrow('='),
        vecRow('x', tk.xmid, oD, D, 'after: now it carries context from other tokens'));
    } else if (s === 4) {
      head(stageTitle + 'normalize again', 'Same as step 1: rescale before the next block of multiplications.');
      c.append(vecRow('x', tk.xmid, oD, D, 'after attention'), arrow('rescale'), vecRow('h', tk.h2, oD, D, 'normalized'));
    } else {
      head(stageTitle + 'feed-forward, then add', `The token is processed on its own: stretched to ${fmtInt(I)} numbers twice (gate and up), combined as silu(gate) × up, squeezed back to ${D}, and ADDED to the running vector (chapter 7).`);
      const act = tr.ffnAct;
      c.append(
        shape([dm('1'), ' × ', dn(fmtInt(D))], [dn(fmtInt(D)), ' × ', dm(fmtInt(I))], [dm('1'), ' × ', dm(fmtInt(I)), ' (twice)']),
        vecRow('gate', tk.gate, oI, I),
        vecRow('up', tk.up, oI, I),
        vecRow('silu(gate) × up', act, oI, I, 'most near zero, a few fire'),
        shape([dm('1'), ' × ', dn(fmtInt(I))], [dn(fmtInt(I)), ' × ', dm(fmtInt(D))], [dm('1'), ' × ', dm(fmtInt(D)), ' correction']),
        vecRow('x', tk.xmid, oD, D, 'before'), arrow('+'),
        vecRow('correction', tk.down, oD, D, 'from the feed-forward'), arrow('='),
        vecRow('x', l + 1 < L ? tk.x : tr.hidden, l + 1 < L ? (l + 1) * D : oD, D, l + 1 < L ? `leaves layer ${l}, enters layer ${l + 1}` : 'leaves the last layer'));
    }
    c.append(guessBox(l, s));
  }

  const arrow = t => h('div', { class: 'tv-arrow', 'aria-hidden': 'true' }, t);
  const nextHint = t => h('p', { class: 'caption' }, t);

  function headsTable(tr, l) {
    const { H, KVH } = d();
    const { T, N, fullAttn } = tr;
    const group = H / KVH;
    const qpos = N - 1;
    const box = h('div', { class: 'tv-heads' });
    let sinks = 0, best = null;
    for (let hh = 0; hh < H; hh++) {
      const o = fullAttn ? ((l * H + hh) * T + (T - 1)) * N : (l * H + hh) * N;
      const row = tr.attn.subarray(o, o + N);
      const idx = Array.from({ length: qpos + 1 }, (_, j) => j).sort((a, b) => row[b] - row[a]).slice(0, 3);
      if (row[0] > 0.5) sinks++;
      else if (!best || row[0] < best.sink) best = { hh, sink: row[0], j: idx[0], p: row[idx[0]] };
      box.append(h('div', { class: 'tv-head' },
        h('span', { class: 'tv-hname' }, `head ${hh}`, h('small', {}, ` key/value head ${Math.floor(hh / group)}`)),
        h('span', { class: 'tv-hits' }, idx.map(j => h('span', { class: 'tv-hit' },
          h('code', {}, vis(S.tok.tokenText(tokenAtPos(j)))),
          h('span', { class: 'tv-pct' }, `${Math.round(row[j] * 100)}%`))))));
    }
    const note = sinks === H
      ? `In this layer every head parks most of its attention on the first token (the attention sink from chapter 6), so attention adds little here.`
      : `${sinks} of ${H} heads park most of their attention on the first token (the attention sink from chapter 6). The others are doing real work: head ${best.hh} puts ${Math.round(best.p * 100)}% on “${vis(S.tok.tokenText(tokenAtPos(best.j)))}”.`;
    return h('div', {}, h('p', { class: 'tv-sub' }, 'Where each head looked (its top 3 earlier tokens):'), box, h('p', { class: 'caption' }, note));
  }

  function tokenAtPos(p) {
    const run = S.run;
    return p < run.ids.length ? run.ids[p] : run.steps[p - run.ids.length]?.token;
  }

  // One output number of q, computed by hand from real weights.
  function worked(l) {
    const box = h('div', { class: 'tv-worked' });
    const tr = trace();
    const D = d().D;
    const draw = W => {
      const hv = tr.tok.h.subarray(l * D, l * D + D);
      let sum = 0;
      for (let i = 0; i < D; i++) sum += hv[i] * W[i];
      const term = i => h('span', { class: 'w' }, `(${hv[i].toFixed(3)} × ${W[i].toFixed(3)})`);
      box.textContent = '';
      box.append(h('p', { class: 'tv-sub' }, 'Where one of those numbers comes from: the first number of q is h times the first row of this layer’s Wq.'),
        h('p', { class: 'mv-work' }, 'q[0] = ', term(0), ' + ', term(1), ' + ', term(2), ` + … ${fmtInt(D - 3)} more pairs … = `, h('b', {}, sum.toFixed(3))),
        h('p', { class: 'caption' }, `That’s one dot product. Doing it for all ${fmtInt(D)} rows gives all of q, and the same for k and v. Then q and k are rotated by position (chapter 4).`));
    };
    const cached = rowCache.get(l);
    if (cached) draw(cached);
    else {
      box.append(h('p', { class: 'caption' }, 'Fetching the weights…'));
      ctx.engine.call('slice', { name: `model.layers.${l}.self_attn.q_proj.weight`, rows: 1, cols: D })
        .then(r => { rowCache.set(l, r.data); if (where(pos).l === l) draw(r.data); }).catch(() => {});
    }
    return box;
  }

  function guessBox(l, s) {
    const box = h('div', { class: 'tv-guess' });
    if (!lens) {
      box.append(h('span', { class: 'dim' }, S.running ? 'The model’s best guess per layer appears when generation finishes.' : 'Working out the model’s best guess at each layer…'));
      fetchLens();
      return box;
    }
    const top = lens[l];
    box.append(h('b', {}, s === 5 ? `Best guess after layer ${l}: ` : `Best guess after this layer finishes: `),
      ...top.slice(0, 3).map((c, i) => h('span', { class: 'tv-gtok' }, h('code', {}, vis(S.tok.tokenText(c.id))), ` ${(c.p * 100).toFixed(0)}%${i < 2 ? '' : ''}`)),
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

  return {
    update(reason) {
      if (reason === 'model') { rowCache = new Map(); lens = null; lensFor = null; build(); }
      if (reason === 'prefill') { lens = null; lensFor = null; pos = Math.min(pos, last()); build(); }
      if (reason === 'idle' && ui && !lens) fetchLens();
    },
  };
}
