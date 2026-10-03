// Plain-language definitions. Any <dfn data-term="key"> on the page shows its
// definition on hover, keyboard focus, or tap.

export const TERMS = {
  vector: 'A list of numbers. A colour is a vector of 3 (red, green, blue). In this model every token becomes a vector of 576 numbers.',
  matrix: 'A grid of numbers with rows and columns, like a spreadsheet. A 192 × 576 matrix has 192 rows of 576 numbers each.',
  tensor: 'The general name for a box of numbers with any number of directions: a vector has one (a list), a matrix has two (a grid), and a stack of matrices has three.',
  weights: 'The numbers inside the model’s matrices. Training chose them; during inference they never change. Also called parameters.',
  parameters: 'Another word for weights: the learned numbers in the model. “135M parameters” means 135 million of them.',
  token: 'A piece of text from the model’s fixed vocabulary: a whole word, part of a word, a punctuation mark or a single byte. Each has an ID number.',
  vocabulary: 'The fixed list of every token the model knows, each with an ID. This model has 49,152.',
  embedding: 'The vector of numbers that stands for a token inside the model. It is looked up from a table using the token’s ID.',
  dot: 'Dot product: multiply two vectors number by number and add the results. A big positive answer means they point the same way, which is a way of measuring similarity.',
  matmul: 'Matrix multiplication: for every row of the matrix, take the dot product with the input vector. Each row gives one output number.',
  layer: 'One repeated block of the model: attention followed by a feed-forward network. This model stacks 30 identical-shaped layers, each with its own weights.',
  softmax: 'Turns a list of scores into percentages that add up to 100%. Higher scores get much bigger shares.',
  logits: 'The raw score the model gives each vocabulary entry for being the next token, before turning scores into percentages.',
  attention: 'The step where each token gathers information from earlier tokens, deciding how much to take from each.',
  head: 'Attention runs several times side by side with different weights. Each copy is a head and can track a different kind of relationship.',
  query: 'In attention, the vector a token uses to ask “what am I looking for?”',
  key: 'In attention, the vector a token uses to advertise “this is what I contain.” Queries are compared against keys.',
  value: 'In attention, the information a token hands over when another token pays attention to it.',
  residual: 'The running vector for each token. Each layer adds its contribution to it instead of replacing it.',
  kvcache: 'KV cache: stored keys and values for every earlier token in every layer, so they don’t have to be recomputed for each new token.',
  prefill: 'The first phase: the whole prompt goes through the model in one pass, all tokens at once.',
  decode: 'The second phase: the model generates one token per pass, each pass feeding in the previous output.',
  cpu: 'The computer’s main processor. A few fast, flexible cores. Here it handles text: tokenizing, sampling and turning tokens back into text.',
  gpu: 'Graphics processor: thousands of small cores built to do huge numbers of multiply-adds at once. This is where model math runs.',
  ram: 'The computer’s main memory, attached to the CPU. Large but far slower to reach from the GPU.',
  hbm: 'High-bandwidth memory: the GPU’s own memory, stacked right next to the chip. An H100 has 80 GB that it can read at 3.35 terabytes per second.',
  pcie: 'The connection between the CPU side and the GPU card. Fast for a cable, but about 50× slower than the GPU reading its own memory.',
  sm: 'Streaming multiprocessor: one of the GPU’s compute blocks. An H100 has 132, each with a little very fast on-chip memory (SRAM) and tensor cores for matrix math.',
  sram: 'Tiny on-chip memory inside each SM. Much faster than HBM, but only a few hundred kilobytes per SM.',
  l2: 'A 50 MB cache shared by all SMs, sitting between them and HBM.',
  bandwidth: 'How many bytes per second can be moved. The GPU’s memory bandwidth is often the real speed limit for generating text.',
  flop: 'One floating-point operation, such as one multiply or one add. GFLOP = a billion, TFLOP = a trillion.',
  bf16: 'bfloat16: a 2-byte number format. It keeps the range of a normal 4-byte float but with less precision, which is plenty for weights.',
  quantization: 'Storing weights with fewer bits (8 or 4 instead of 16) to shrink the model and the bytes read per token, at a small cost in quality.',
  latency: 'How long you wait. Time to first token (TTFT) is how long before the first word appears.',
  batch: 'Several requests processed together in one pass, so every weight fetched from memory is used for all of them.',
  bpe: 'Byte Pair Encoding: build the vocabulary by repeatedly gluing together the most common neighbouring pair of pieces in a big pile of text.',
  rope: 'Rotary position embedding: the model marks where each token sits by rotating its query and key vectors by an amount that depends on position.',
};

export function initGlossary(tipEl) {
  let pinned = null;
  const show = el => {
    const text = TERMS[el.dataset.term];
    if (!text) return;
    tipEl.textContent = text;
    tipEl.hidden = false;
    const r = el.getBoundingClientRect();
    const t = tipEl.getBoundingClientRect();
    let x = Math.min(Math.max(8, r.left), innerWidth - t.width - 8);
    let y = r.bottom + 8;
    if (y + t.height > innerHeight - 8) y = r.top - t.height - 8;
    tipEl.style.left = `${x}px`;
    tipEl.style.top = `${y}px`;
  };
  const hide = () => { if (!pinned) tipEl.hidden = true; };
  const prep = el => {
    if (el.dataset.ready) return;
    el.dataset.ready = '1';
    el.tabIndex = 0;
    el.setAttribute('role', 'button');
    el.setAttribute('aria-label', `${el.textContent}: ${TERMS[el.dataset.term] || ''}`);
  };
  document.querySelectorAll('dfn[data-term]').forEach(prep);
  document.addEventListener('mouseover', e => { const d = e.target.closest?.('dfn[data-term]'); if (d) { prep(d); show(d); } });
  document.addEventListener('mouseout', e => { if (e.target.closest?.('dfn[data-term]')) hide(); });
  document.addEventListener('focusin', e => { const d = e.target.closest?.('dfn[data-term]'); if (d) show(d); });
  document.addEventListener('focusout', e => { if (e.target.closest?.('dfn[data-term]')) { pinned = null; hide(); } });
  document.addEventListener('click', e => {
    const d = e.target.closest?.('dfn[data-term]');
    if (d) { pinned = pinned === d ? null : d; if (pinned) show(d); else tipEl.hidden = true; }
    else if (pinned) { pinned = null; tipEl.hidden = true; }
  });
  addEventListener('scroll', () => { if (!pinned) tipEl.hidden = true; else show(pinned); }, { passive: true });
}
