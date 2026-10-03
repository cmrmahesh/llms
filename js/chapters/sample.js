// 7. Logits -> probabilities -> sampled token -> bytes -> text.

import { $, h, fmtInt, vis, samplingDistribution } from '../util.js';
import { tokenAt } from './attention.js';

export default function sample(ctx) {
  const { S } = ctx;
  let ctlBuilt = false;

  function buildControls() {
    const el = $('#sampler-controls');
    el.textContent = '';
    const st = S.settings;
    const mk = (label, key, min, max, step, fmt) => {
      const out = h('output', {}, fmt(st[key]));
      const inp = h('input', { type: 'range', min, max, step, value: st[key], 'aria-label': label });
      inp.addEventListener('input', () => { st[key] = +inp.value; out.textContent = fmt(st[key]); draw(); });
      inp.dataset.key = key;
      return h('label', {}, label, inp, out);
    };
    const greedy = h('button', { class: 'btn small' }, 'Greedy');
    greedy.addEventListener('click', () => {
      st.temperature = 0;
      el.querySelector('[data-key="temperature"]').value = 0;
      el.querySelector('[data-key="temperature"]').nextElementSibling.textContent = 'greedy';
      draw();
    });
    el.append(
      mk('Temperature', 'temperature', 0, 2, 0.05, v => (v <= 0.001 ? 'greedy' : v.toFixed(2))),
      mk('Top-k', 'topK', 0, 100, 1, v => (v === 0 ? 'off' : String(v))),
      mk('Top-p', 'topP', 0.05, 1, 0.05, v => (v >= 1 ? 'off' : v.toFixed(2))),
      greedy,
    );
    ctlBuilt = true;
  }

  function draw() {
    const el = $('#dist');
    el.textContent = '';
    const run = S.run;
    if (!run?.steps.length) {
      el.append(h('p', { class: 'empty', style: { margin: 0 } }, S.loaded ? 'Generate to see the model’s probability distribution for each token it writes.' : 'Load the weights and generate to see real probabilities.'));
      drawDetok(null);
      return;
    }
    const i = Math.max(0, Math.min(S.selStep < 0 ? run.steps.length - 1 : S.selStep, run.steps.length - 1));
    const step = run.steps[i];
    // Navigator
    const prev = h('button', { class: 'btn small', 'aria-label': 'Previous token', disabled: i === 0 }, '‹');
    const next = h('button', { class: 'btn small', 'aria-label': 'Next token', disabled: i === run.steps.length - 1 }, '›');
    prev.addEventListener('click', () => ctx.selectStep(i - 1));
    next.addEventListener('click', () => ctx.selectStep(i + 1));
    const ctxTokens = [];
    for (let p = Math.max(0, step.pos - 6); p < step.pos; p++) ctxTokens.push(S.tok.tokenText(tokenAt(S, p)));
    el.append(h('div', { class: 'dist-head' },
      h('span', {}, prev, ' ', h('b', {}, `Token ${i + 1} of ${run.steps.length}`), ' ', next),
      h('span', {}, 'Predicting what follows ', h('code', {}, `…${vis(ctxTokens.join(''))}`))));
    if (!step.logits) {
      el.append(h('p', { class: 'caption' }, 'Logits for this older step were freed to save memory.'));
      drawDetok(step);
      return;
    }
    const dist = samplingDistribution(step.logits, S.settings, 18);
    const kept = dist.cands.filter(c => c.kept).length;
    let ent = 0;
    for (const c of dist.cands) if (c.pRaw > 0) ent -= c.pRaw * Math.log2(c.pRaw);
    const maxP = Math.max(...dist.shown.map(c => Math.max(c.p, c.pRaw)));
    const list = h('div', {});
    for (const c of dist.shown) {
      const chosen = c.id === step.token;
      const b = h('button', {
        class: `dbar${c.kept ? '' : ' out'}${chosen ? ' chosen' : ''}`,
        title: S.running ? '' : (chosen ? 'The token that was sampled' : 'Use this token instead and continue from here'),
        disabled: S.running,
      },
      h('span', { class: 's' }, vis(S.tok.tokenText(c.id))),
      h('span', { class: 'lg' }, c.logit.toFixed(2)),
      h('span', { class: 'b' }, h('span', { style: { width: `${(c.kept ? c.p : c.pRaw) / maxP * 100}%` } })),
      h('span', { class: 'p' }, c.kept ? `${(c.p * 100).toFixed(1)}%` : 'cut'));
      if (!chosen) b.addEventListener('click', () => ctx.branch(i, c.id));
      list.append(b);
    }
    const chosenC = dist.cands.find(c => c.id === step.token);
    el.append(
      h('div', { class: 'dist-head', style: { marginBottom: '4px' } },
        h('span', {}, 'Token · logit · probability after your settings'),
        h('span', {}, `${kept === dist.cands.length && kept >= 200 ? '200+' : kept} of ${fmtInt(S.config.vocab_size)} tokens can be drawn · uncertainty ${ent.toFixed(2)} bits`)),
      list,
      h('p', { class: 'caption' }, `Sampled `, h('code', {}, vis(S.tok.tokenText(step.token))),
        chosenC ? ` with ${(step.p * 100).toFixed(1)}% probability under the settings used at the time (${(chosenC.pRaw * 100).toFixed(1)}% before temperature and filters).` : '.',
        step.branched ? ' You picked this one by hand.' : '',
        ' Click another row to swap it in and continue from there.'));
    drawDetok(step);
  }

  function drawDetok(step) {
    const el = $('#detok');
    el.textContent = '';
    if (!step) { el.append(h('span', { class: 'dim' }, 'The chosen token’s journey back to text appears here.')); return; }
    const id = step.token;
    const special = S.tok.specialById.has(id);
    const bytes = Array.from(S.tok.tokenBytes(id));
    const stage = (label, val) => h('div', { class: 'stage' }, h('small', {}, label), h('span', {}, val));
    const arrow = () => h('span', { class: 'arrow', 'aria-hidden': 'true' }, '→');
    el.append(
      stage('Token ID', String(id)), arrow(),
      stage('Vocabulary entry', S.tok.tokenString(id)), arrow(),
      stage(special ? 'Special token' : 'Bytes (hex)', special ? 'control marker, not text' : bytes.map(b => b.toString(16).padStart(2, '0')).join(' ')), arrow(),
      stage('Text', special ? (ctx.eosIds().includes(id) ? '(end of answer)' : S.tok.tokenText(id)) : `“${vis(S.tok.tokenText(id))}”`),
    );
  }

  return {
    update(reason) {
      if (reason === 'model') { buildControls(); draw(); }
      if (['weights', 'prefill', 'step', 'select', 'idle', 'branch'].includes(reason)) draw();
      if (reason === 'model' && !ctlBuilt) buildControls();
    },
  };
}

