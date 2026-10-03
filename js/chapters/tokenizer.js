// 2. Tokenizer: pre-split, BPE merge replay, vocabulary lookup.

import { $, h, fmtInt, fmtMs, vis } from '../util.js';

// Byte-mapped symbol -> readable label (Ġ stays visible: it is what the vocab stores).
const showSym = s => s.replace(/\n/g, '↵').replace(/\t/g, '⇥');

export default function tokenizer(ctx) {
  const { S } = ctx;
  let selWord = -1;      // index into enc.segments

  function pickDefault(segs) {
    // Prefer the word with the most merges in the user's part of the prompt.
    let best = -1, bestSteps = -1;
    const user = document.getElementById('prompt').value;
    segs.forEach((s, i) => {
      if (s.kind !== 'word') return;
      const bonus = s.text.trim().length > 2 && user.includes(s.text.trim()) ? 100 : 0;
      const score = s.steps.length + bonus;
      if (score > bestSteps) { bestSteps = score; best = i; }
    });
    return best;
  }

  function renderPieces() {
    const el = $('#tok-pieces');
    el.textContent = '';
    const segs = S.enc.segments;
    segs.forEach((s, i) => {
      const b = h('button', {
        class: `piece${s.kind === 'special' ? ' special' : ''}`,
        'aria-pressed': String(i === selWord),
        title: s.kind === 'special' ? `Special token, id ${s.ids[0]}` : `${s.ids.length} token${s.ids.length > 1 ? 's' : ''} after BPE`,
      }, s.kind === 'special' ? s.text : vis(s.text));
      if (s.kind === 'word') b.addEventListener('click', () => { selWord = i; tryText = null; if (tryInput) tryInput.value = ''; renderPieces(); drawBpe(); });
      el.append(b);
    });
  }


  // ---------------- BPE as a staircase: every glue step at once
  let tryText = null;            // a word typed into "try any word", or null to follow the prompt
  let bpeBody = null, tryInput = null;
  const EXAMPLES = [' cat', ' tokenization', ' unbelievably', ' SmolLM', ' 2026', ' antidisestablishmentarianism'];

  function buildBpe() {
    const el = $('#bpe');
    el.textContent = '';
    tryInput = h('input', { type: 'search', placeholder: 'Try any word', 'aria-label': 'Try any word', value: tryText ?? '' });
    tryInput.addEventListener('input', () => { tryText = tryInput.value ? tryInput.value : null; drawBpe(); });
    const ex = EXAMPLES.map(w => {
      const b = h('button', { class: 'btn small' }, w.trim());
      b.addEventListener('click', () => { tryText = w; tryInput.value = w; drawBpe(); });
      return b;
    });
    bpeBody = h('div', { 'aria-live': 'polite' });
    el.append(h('div', { class: 'vocab-ctl' }, tryInput, ...ex), bpeBody);
    drawBpe();
  }

  function staircase(seg) {
    const box = h('div', { class: 'stair' });
    const symRow = (symbols, merged) => h('div', { class: 'bpe-row' }, symbols.map(x => h('span', { class: `sym${x === merged ? ' merged' : ''}` }, showSym(x))));
    box.append(h('div', { class: 'stair-step' },
      h('span', { class: 'stair-label' }, `Start: ${seg.bytes.length} byte${seg.bytes.length > 1 ? 's' : ''}`),
      symRow(seg.initial, null)));
    seg.steps.forEach((st, i) => {
      box.append(h('div', { class: 'stair-step' },
        h('span', { class: 'stair-label' }, `Glue ${i + 1}: `, h('code', {}, showSym(st.pair[0])), ' + ', h('code', {}, showSym(st.pair[1])), h('small', {}, ` rule #${fmtInt(st.rank)}`)),
        symRow(st.symbols, st.pair.join(''))));
    });
    box.append(h('div', { class: 'stair-step done' },
      h('span', { class: 'stair-label' }, seg.steps.length ? 'No rule applies any more.' : 'No rule applies.'),
      h('div', { class: 'bpe-row' }, seg.symbols.map((x, i) => h('span', { class: 'tid' }, h('span', { class: 's' }, showSym(x)), h('span', { class: 'n' }, String(seg.ids[i])))),
        h('span', { class: 'dim' }, ` → ${seg.ids.length} token${seg.ids.length > 1 ? 's' : ''}`))));
    return box;
  }

  function drawBpe() {
    if (!bpeBody) return;
    bpeBody.textContent = '';
    let segs;
    if (tryText !== null) {
      segs = S.tok.encodeWithTrace(tryText).segments.filter(x => x.kind === 'word');
      if (!segs.length) { bpeBody.append(h('p', { class: 'caption' }, 'Type a word.')); return; }
      if (segs.length > 1) bpeBody.append(h('p', { class: 'caption' }, `Step 1 cut this into ${segs.length} separate words first (digits are always cut one by one), so each is glued on its own:`));
    } else {
      const seg = S.enc.segments[selWord];
      if (!seg || seg.kind !== 'word') { bpeBody.append(h('p', { class: 'caption' }, 'Pick a word above, or type one.')); return; }
      segs = [seg];
      bpeBody.append(h('p', { class: 'caption' }, 'Showing ', h('code', {}, vis(seg.text)), ' from your prompt. Type a word or pick an example to try others.'));
    }
    for (const seg of segs.slice(0, 6)) bpeBody.append(staircase(seg));
  }

  function renderIds() {
    const el = $('#token-ids');
    el.textContent = '';
    const run = S.run;
    const cachedN = run && run.prefill && run.ids.length === S.ids.length && run.ids.every((x, i) => x === S.ids[i]) ? run.prefill.reused : 0;
    S.ids.forEach((id, i) => {
      const special = S.tok.specialById.has(id);
      el.append(h('span', { class: `tid${special ? ' special' : ''}${i < cachedN ? ' cached' : ''}`, title: `Position ${i}` },
        h('span', { class: 's' }, special ? S.tok.tokenString(id) : vis(S.tok.tokenText(id))),
        h('span', { class: 'n' }, String(id))));
    });
    const chars = [...S.text].length;
    const bytes = new TextEncoder().encode(S.text).length;
    const f = $('#tok-facts');
    f.textContent = '';
    f.append(
      h('span', {}, 'Characters ', h('b', {}, fmtInt(chars))),
      h('span', {}, 'UTF-8 bytes ', h('b', {}, fmtInt(bytes))),
      h('span', {}, 'Tokens ', h('b', {}, fmtInt(S.ids.length))),
      h('span', {}, 'Bytes per token ', h('b', {}, (bytes / Math.max(1, S.ids.length)).toFixed(2))),
      h('span', {}, 'Tokenized in ', h('b', {}, fmtMs(S.tokenizeMs))),
      h('span', {}, 'Sent to the GPU ', h('b', {}, `${fmtInt(S.ids.length * 4)} bytes`)),
    );
  }

  // ---------------- vocabulary explorer
  let vocabTexts = null;
  let vocabMode = 'common';
  let vocabQuery = '';
  function texts() {
    if (!vocabTexts) vocabTexts = S.tok.idToToken.map((_, id) => S.tok.tokenText(id));
    return vocabTexts;
  }
  function buildVocab() {
    const el = $('#vocab-explorer');
    el.textContent = '';
    if (!S.tok) return;
    vocabTexts = null;
    const search = h('input', { type: 'search', placeholder: 'Search, e.g. cat, ing, 202, 🙂', 'aria-label': 'Search the vocabulary', value: vocabQuery });
    const list = h('div', { class: 'vocab-list', 'aria-live': 'polite' });
    const note = h('p', { class: 'vocab-note' });
    const modes = [['common', 'Common pieces'], ['special', 'Special tokens'], ['long', 'Longest'], ['random', 'Random 40']];
    const btns = modes.map(([m, label]) => {
      const b = h('button', { class: 'btn small', 'aria-pressed': String(vocabMode === m && !vocabQuery) }, label);
      b.addEventListener('click', () => { vocabMode = m; vocabQuery = ''; search.value = ''; draw(); });
      return b;
    });
    search.addEventListener('input', () => { vocabQuery = search.value; draw(); });
    function chip(id) {
      const t = texts()[id];
      const special = S.tok.specialById.has(id);
      return h('span', { class: `tid${special ? ' special' : ''}`, title: `ID ${id}` },
        h('span', { class: 's' }, special ? t : vis(t)), h('span', { class: 'n' }, String(id)));
    }
    function draw() {
      btns.forEach((b, i) => b.setAttribute('aria-pressed', String(!vocabQuery && vocabMode === modes[i][0])));
      list.textContent = '';
      const all = texts();
      let ids = [];
      if (vocabQuery.trim()) {
        const q = vocabQuery.toLowerCase();
        for (let id = 0; id < all.length && ids.length < 400; id++) if (all[id].toLowerCase().includes(q)) ids.push(id);
        const total = ids.length;
        ids.sort((a, b) => all[a].length - all[b].length || a - b);
        ids = ids.slice(0, 80);
        note.textContent = total ? `${total >= 400 ? '400+' : total} entries contain “${vocabQuery}”. Showing the shortest ${ids.length}.` : `No entry contains “${vocabQuery}”, so it would be built from smaller pieces.`;
      } else if (vocabMode === 'common') {
        ids = Array.from({ length: 80 }, (_, i) => 300 + i);
        note.textContent = 'IDs roughly follow the order pieces were learned: low IDs are pieces that showed up most often in the training text. These are IDs 300–379.';
      } else if (vocabMode === 'special') {
        ids = [...S.tok.specialById.keys()].sort((a, b) => a - b);
        note.textContent = 'Markers that never come from ordinary text. The chat ones tell the model who is speaking and where a message ends.';
      } else if (vocabMode === 'long') {
        ids = all.map((t, id) => id).sort((a, b) => all[b].length - all[a].length).slice(0, 40);
        note.textContent = 'The longest entries: runs of spaces or symbols that are common in code, and long frequent words.';
      } else {
        ids = Array.from({ length: 40 }, () => Math.floor(Math.random() * all.length)).sort((a, b) => a - b);
        note.textContent = 'A random handful. Press again for more.';
      }
      list.append(...ids.map(chip));
    }
    el.append(h('div', { class: 'vocab-ctl' }, search, ...btns), list, note);
    draw();
  }

  return {
    update(reason) {
      if (reason === 'model') {
        buildVocab();
        document.querySelectorAll('[data-bind="vocab"]').forEach(e => { e.textContent = fmtInt(S.tok.vocabSize); });
      }
      if (reason === 'tokens') {
        const segs = S.enc.segments;
        if (selWord < 0 || selWord >= segs.length || segs[selWord].kind !== 'word') selWord = pickDefault(segs);
        renderPieces(); if (!bpeBody) buildBpe(); else drawBpe(); renderIds();
      }
      if (reason === 'prefill') renderIds();
    },
  };
}
