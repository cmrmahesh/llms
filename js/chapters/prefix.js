// 10. Prefix caching: hash-chained 16-token blocks, reuse, measured TTFT.

import { $, h, fmtMs, vis, fmtInt } from '../util.js';

const BLOCK = 16;
const QUESTIONS = [
  'What is the capital of Japan?',
  'Name three primary colors.',
  'Why is the sky blue? Answer in one sentence.',
  'What does a GPU do?',
  'Give me a synonym for happy.',
  'What is 12 times 12?',
];

// FNV-1a over (parent hash, token ids): a block's identity includes everything before it.
function blockHashes(ids) {
  const out = [];
  let parent = 0x811c9dc5;
  for (let b = 0; b + BLOCK <= ids.length; b += BLOCK) {
    let x = parent;
    const mix = v => { for (let k = 0; k < 4; k++) { x ^= (v >>> (8 * k)) & 0xff; x = Math.imul(x, 0x01000193) >>> 0; } };
    mix(parent);
    for (let i = b; i < b + BLOCK; i++) mix(ids[i]);
    out.push(x >>> 0);
    parent = x >>> 0;
  }
  return out;
}

export default function prefix(ctx) {
  const { S } = ctx;
  let qi = 0;
  let prevIds = null;     // ids of the request before the current one
  let lastRunIds = null;

  const LONG_SYSTEM = 'You are a helpful AI assistant named SmolLM, trained by Hugging Face. You answer questions about science, geography and everyday life. Keep answers short: one or two sentences. Use plain words, avoid jargon, and never make up facts. If you are not sure about something, say so.';
  $('#btn-long-sys').addEventListener('click', () => {
    if (S.running || !S.meta.chat) return;
    ctx.setSystem(LONG_SYSTEM);
    document.getElementById('system-wrap').open = true;
  });
  $('#opt-prefix').addEventListener('change', e => { S.settings.prefixCache = e.target.checked; });
  $('#btn-new-q').addEventListener('click', () => {
    if (!S.loaded || S.running) return;
    const cur = $('#prompt').value.trim();
    let q = QUESTIONS[qi++ % QUESTIONS.length];
    if (q === cur) q = QUESTIONS[qi++ % QUESTIONS.length];
    ctx.setPrompt(q, true);
  });

  function row(ids, reused, label) {
    const hashes = blockHashes(ids);
    const wrap = h('div', {});
    wrap.append(h('p', { class: 'caption', style: { margin: '6px 0 0' } }, label));
    const r = h('div', { class: 'blk-row' });
    hashes.forEach((hs, b) => {
      const hit = reused !== null && (b + 1) * BLOCK <= reused;
      const text = ids.slice(b * BLOCK, b * BLOCK + BLOCK).map(id => S.tok.tokenText(id)).join('');
      r.append(h('div', { class: `blk${hit ? ' hit' : ''}`, title: vis(text) },
        h('div', { class: 'h' }, `#${hs.toString(16).padStart(8, '0').slice(0, 6)}`),
        h('div', { class: 't' }, vis(text)),
        h('div', {}, reused === null ? `tokens ${b * BLOCK}–${b * BLOCK + BLOCK - 1}` : hit ? 'reused' : 'computed')));
    });
    const rest = ids.length % BLOCK;
    if (rest) {
      const text = ids.slice(ids.length - rest).map(id => S.tok.tokenText(id)).join('');
      r.append(h('div', { class: 'blk partial', title: vis(text) },
        h('div', { class: 'h' }, 'no hash yet'),
        h('div', { class: 't' }, vis(text)),
        h('div', {}, `${rest} of ${BLOCK}: not cacheable until full`)));
    }
    wrap.append(r);
    return wrap;
  }

  function render() {
    const el = $('#blocks');
    el.textContent = '';
    const run = S.run;
    if (!run?.prefill) {
      if (S.tok && S.ids.length) el.append(row(S.ids, null, `Your current prompt as ${Math.ceil(S.ids.length / BLOCK)} blocks (the last one partial):`));
      else el.append(h('p', { class: 'empty' }, 'Generate to see the blocks.'));
    } else {
      if (prevIds) el.append(row(prevIds, null, 'Previous request'));
      el.append(row(run.ids, run.prefill.reused, `This request: ${run.prefill.reused} of ${run.ids.length} prompt tokens reused`));
      el.append(h('p', { class: 'caption' }, `Even when the whole prompt matches, the engine recomputes the last block: it needs fresh logits for the last token. Real engines keep thousands of these blocks per GPU from many requests and evict the least recently used.`));
    }
    const t = $('#ttft');
    t.textContent = '';
    if (S.runs.length) {
      t.append(h('table', { class: 'data' },
        h('thead', {}, h('tr', {}, h('th', {}, 'Run'), h('th', {}, 'Prompt tokens'), h('th', {}, 'Reused'), h('th', {}, 'Computed'), h('th', {}, 'Prefill time'), h('th', {}, 'Time to first token'))),
        h('tbody', {}, S.runs.slice(-8).map(r => h('tr', { class: r.reused ? 'hl' : '' },
          h('td', {}, `#${r.n}${r.prefixCache ? '' : ' (cache off)'}`), h('td', {}, fmtInt(r.tokens)), h('td', {}, fmtInt(r.reused)), h('td', {}, fmtInt(r.computed)),
          h('td', {}, fmtMs(r.ms)), h('td', {}, fmtMs(r.ttft)))))));
    }
  }

  return {
    update(reason) {
      if (reason === 'model') { prevIds = lastRunIds = null; $('#btn-long-sys').hidden = !S.meta.chat; }
      if (reason === 'run-start') prevIds = lastRunIds;
      if (['model', 'tokens', 'prefill', 'weights'].includes(reason)) render();
      if (reason === 'prefill') lastRunIds = S.run.ids;
    },
  };
}
