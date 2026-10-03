// 11. One request, every byte: phase-by-phase table for the last run.

import { $, h, fmtBytes, fmtMs, fmtFlops, fmtInt } from '../util.js';

const BW = 3.35e12, PEAK = 989e12, PCIE = 64e9;

export default function summary(ctx) {
  const { S } = ctx;

  function render() {
    const el = $('#summary');
    el.textContent = '';
    const d = ctx.dims();
    const run = S.run;
    if (!d || !run?.prefill) { el.append(h('p', { class: 'empty' }, 'Generate something to fill in this table with your own run.')); return; }
    const P = d.activeParams;
    const T = run.ids.length, Tc = run.prefill.computed;
    const dec = run.steps.filter(s => s.kind === 'decode');
    const n = dec.length;
    const textBytes = new TextEncoder().encode(run.text).length;
    // Prefill
    const preFlops = 2 * P * Tc + 4 * d.L * d.H * d.HD * Tc * T / 2;
    const preBytes = 2 * P + Tc * d.kvBytesPerToken;
    const preH100 = Math.max(preFlops / PEAK, preBytes / BW);
    // Decode: context grows from T to T + n - 1
    let decBytes = 0, decFlops = 0, decH100 = 0;
    for (let i = 0; i < n; i++) {
      const ctxLen = T + i + 1;
      const b = 2 * P + ctxLen * d.kvBytesPerToken;
      const f = 2 * P + 4 * d.L * d.H * d.HD * ctxLen;
      decBytes += b; decFlops += f; decH100 += Math.max(f / PEAK, b / BW);
    }
    const decMs = dec.reduce((a, s) => a + s.ms, 0);
    const li = S.loadInfo;
    const rows = [
      ['Load weights (once per server start)', 'Disk → RAM → HBM', fmtBytes(li.fileBytes), '—', `≥ ${fmtMs(li.fileBytes / PCIE * 1000)} over PCIe`, fmtMs(li.tDownload + li.tConvert), 'link'],
      ['Tokenize', 'CPU', `${fmtInt(textBytes)} B in, ${fmtInt(T * 4)} B out`, '—', 'microseconds', fmtMs(run.tokenizeMs), 'host'],
      ['Send token IDs', 'PCIe', `${fmtInt(T * 4)} B`, '—', '~10 µs (latency)', '—', 'link'],
      [`Prefill (${fmtInt(Tc)} tokens${run.prefill.reused ? `, ${run.prefill.reused} reused` : ''})`, 'HBM → SMs', `${fmtBytes(preBytes)} read`, fmtFlops(preFlops), `${fmtMs(preH100 * 1000)}, ${preFlops / PEAK > preBytes / BW ? 'compute' : 'memory'}-bound`, fmtMs(run.prefill.ms), 'sm'],
      [`Decode (${n} step${n === 1 ? '' : 's'})`, 'HBM → SMs', `${fmtBytes(decBytes)} read`, fmtFlops(decFlops), n ? `${fmtMs(decH100 * 1000)}, memory-bound` : '—', n ? fmtMs(decMs) : '—', 'hbm'],
      ['Return token IDs', 'PCIe', `${fmtInt(run.steps.length * 4)} B`, '—', 'microseconds', '—', 'link'],
      ['Detokenize and stream', 'CPU', `${fmtInt(run.steps.length)} tokens`, '—', 'microseconds', '< 1 ms', 'host'],
    ];
    el.append(h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, ['Phase', 'Where', 'Data moved', 'Math', 'H100 estimate', 'This tab'].map(t => h('th', {}, t)))),
      h('tbody', {}, rows.map(r => h('tr', {},
        h('td', {}, r[0]),
        h('td', { class: 'l' }, h('span', { class: `loc ${r[6]}` }, r[1])),
        ...r.slice(2, 6).map(c => h('td', {}, c)))))));
    const ratio = decBytes / Math.max(1, preBytes);
    el.append(h('p', { class: 'caption' }, n
      ? `Decode read ${ratio.toFixed(0)}× more bytes than prefill to produce ${n} tokens, because every step re-reads all ${fmtBytes(2 * P)} of weights. Per byte it did ${(decFlops / decBytes).toFixed(1)} FLOPs, against ${(preFlops / preBytes).toFixed(0)} for prefill.`
      : 'Generate a few tokens to see the decode row.'));
  }

  return {
    update(reason) {
      if (['model', 'prefill', 'idle'].includes(reason)) render();
    },
  };
}
