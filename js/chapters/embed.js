// 3. Embedding lookup: token IDs select rows of the [V × D] table.

import { $, h, fmtInt, fmtBytes, drawGrid, diverging, vis, debounce, rms } from '../util.js';

export default function embed(ctx) {
  const { S } = ctx;
  let rows = null;      // Float32Array T×D for current ids
  let rowsFor = null;   // ids those rows belong to
  let nbToken = null;

  const MAX_ROWS = 160;

  function tableCanvas() {
    const wrap = h('div', { class: 'embed-table' });
    const cv = h('canvas', { 'aria-label': 'The embedding table, with a mark at each row your prompt reads' });
    wrap.append(cv, h('p', { class: 'caption' }, `${fmtInt(S.config.vocab_size)} rows`));
    requestAnimationFrame(() => {
      const r = cv.getBoundingClientRect();
      const dpr = devicePixelRatio || 1;
      cv.width = Math.max(1, r.width * dpr); cv.height = Math.max(1, r.height * dpr);
      const g = cv.getContext('2d');
      const p = S.previews?.find(x => x.name === 'model.embed_tokens.weight');
      if (p?.grid) {
        const off = document.createElement('canvas');
        drawGrid(off, p.grid, p.gr, p.gc, diverging, 3 * p.std);
        g.imageSmoothingEnabled = false;
        g.globalAlpha = 0.55;
        g.drawImage(off, 0, 0, cv.width, cv.height);
        g.globalAlpha = 1;
      } else {
        g.fillStyle = '#e2e7ed';
        g.fillRect(0, 0, cv.width, cv.height);
      }
      g.fillStyle = getComputedStyle(document.body).getPropertyValue('--host');
      for (const id of new Set(S.ids)) {
        const y = Math.floor(id / S.config.vocab_size * cv.height);
        g.fillRect(0, y, cv.width, Math.max(2, dpr * 1.5));
      }
    });
    return wrap;
  }

  function render() {
    const el = $('#embed-viz');
    el.textContent = '';
    if (!S.config) return;
    const D = S.config.hidden_size;
    el.append(tableCanvas());
    const list = h('div', { class: 'embed-rows' });
    const ids = S.ids.slice(0, MAX_ROWS);
    let scale = 0.05;
    if (rows && rowsFor === S.ids) scale = 3 * rms(rows) || 0.05;
    ids.forEach((id, i) => {
      const cv = h('canvas', { class: 'hm', 'aria-hidden': 'true' });
      if (rows && rowsFor === S.ids) drawGrid(cv, rows.subarray(i * D, (i + 1) * D), 1, D, diverging, scale);
      list.append(h('div', { class: 'erow' },
        h('span', { class: 's', title: S.tok.tokenText(id) }, vis(S.tok.tokenText(id))),
        h('span', { class: 'n' }, String(id)),
        cv));
    });
    if (S.ids.length > MAX_ROWS) list.append(h('p', { class: 'caption' }, `…and ${S.ids.length - MAX_ROWS} more rows.`));
    el.append(list);
    const T = S.ids.length;
    const facts = h('p', { class: 'caption', style: { gridColumn: '1 / -1' } },
      S.loaded
        ? `Result: a ${T} × ${fmtInt(D)} matrix (${fmtBytes(T * D * 2)} in bf16), read from a ${fmtBytes(S.config.vocab_size * D * 2)} table. Colour scale ±${scale.toFixed(3)}.`
        : `Load the weights to fill in the ${fmtInt(D)} numbers of each row. The marks on the left show which of the ${fmtInt(S.config.vocab_size)} rows your prompt touches.`);
    el.append(facts);
  }

  const fetchRows = debounce(async () => {
    if (!S.loaded || !S.ids.length) return;
    const ids = S.ids;
    const r = await ctx.engine.call('rows', { ids: ids.slice(0, MAX_ROWS) });
    if (ids !== S.ids) return;
    rows = r.data; rowsFor = ids;
    render();
  }, 200);

  function renderNeighbors(list) {
    const el = $('#neighbors');
    el.textContent = '';
    if (!S.tok) return;
    const uniq = [...new Set(S.ids)].filter(id => !S.tok.specialById.has(id) && /\p{L}{2,}/u.test(S.tok.tokenText(id)));
    if (!uniq.length) { el.append(h('p', { class: 'empty' }, 'Type a prompt with some words to compare.')); return; }
    if (!uniq.includes(nbToken)) nbToken = uniq[uniq.length - 1];
    const sel = h('select', { class: 'inline', 'aria-label': 'Token to compare' },
      uniq.map(id => h('option', { value: id, selected: id === nbToken }, `${vis(S.tok.tokenText(id))}  (${id})`)));
    sel.addEventListener('change', () => { nbToken = +sel.value; askNeighbors(); });
    el.append(h('label', { class: 'num' }, 'Nearest rows to ', sel));
    if (!S.loaded) { el.append(h('p', { class: 'empty' }, 'Load the weights to search the table.')); return; }
    const row = h('div', { class: 'nb-row' });
    if (list) for (const n of list) row.append(h('span', { class: 'nb' }, vis(S.tok.tokenText(n.id)), h('small', {}, n.s.toFixed(2))));
    else row.append(h('span', { class: 'dim' }, 'Searching…'));
    el.append(row, h('p', { class: 'caption' }, `One ${fmtInt(S.config.vocab_size)} × ${fmtInt(S.config.hidden_size)} matrix-vector product, then the highest cosine similarities. Numbers are similarity, where 1 means the same direction.`));
  }

  async function askNeighbors() {
    renderNeighbors(null);
    if (!S.loaded) return;
    const t = nbToken;
    const r = await ctx.engine.call('neighbors', { token: t, k: 12 });
    if (t === nbToken) renderNeighbors(r.list);
  }
  const askNeighborsSoon = debounce(askNeighbors, 400);

  return {
    update(reason) {
      if (reason === 'model') {
        const c = S.config;
        document.querySelectorAll('[data-bind="embedShape"]').forEach(e => { e.textContent = `${fmtInt(c.vocab_size)} × ${fmtInt(c.hidden_size)}`; });
        document.querySelectorAll('[data-bind="D"]').forEach(e => { e.textContent = fmtInt(c.hidden_size); });
        rows = null;
      }
      if (reason === 'tokens') { render(); fetchRows(); askNeighborsSoon(); }
      if (reason === 'weights') { render(); fetchRows(); askNeighbors(); }
    },
  };
}
