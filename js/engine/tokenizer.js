// Byte-level BPE tokenizer (GPT-2 / SmolLM / Llama-3 style) that can explain itself.
// Reads a HuggingFace tokenizer.json and records every merge it applies so the UI
// can replay the process step by step.

// GPT-2's reversible byte -> printable-unicode table. Bytes that are already
// printable map to themselves; the rest (space, control chars...) are shifted to
// code points >= 256. This is why a leading space shows up as "Ġ" (U+0120).
function buildByteTables() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const byteToChar = new Array(256);
  const charToByte = new Map();
  bs.forEach((b, i) => { const c = String.fromCharCode(cs[i]); byteToChar[b] = c; charToByte.set(c, b); });
  return { byteToChar, charToByte };
}

// The GPT-2 pre-tokenization regex: contractions, letter runs with an optional
// leading space, number runs, punctuation runs, and whitespace.
const SPLIT_RE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

const utf8 = new TextEncoder();

export class BPETokenizer {
  constructor(json) {
    const model = json.model;
    if (model.type !== 'BPE') throw new Error(`Unsupported tokenizer model: ${model.type}`);
    this.vocab = new Map(Object.entries(model.vocab));
    this.idToToken = [];
    for (const [tok, id] of this.vocab) this.idToToken[id] = tok;
    this.mergeRank = new Map();
    model.merges.forEach((m, i) => {
      const key = Array.isArray(m) ? `${m[0]} ${m[1]}` : m;
      this.mergeRank.set(key, i);
    });
    this.numMerges = model.merges.length;
    this.special = (json.added_tokens || [])
      .filter(t => t.special)
      .map(t => ({ content: t.content, id: t.id }))
      .sort((a, b) => b.content.length - a.content.length);
    this.specialById = new Map(this.special.map(t => [t.id, t.content]));
    for (const t of json.added_tokens || []) this.idToToken[t.id] = t.content;
    this.splitDigits = JSON.stringify(json.pre_tokenizer || {}).includes('"Digits"');
    const { byteToChar, charToByte } = buildByteTables();
    this.byteToChar = byteToChar;
    this.charToByte = charToByte;
    this.cache = new Map();
  }

  get vocabSize() { return this.idToToken.length; }

  // Split on special tokens first; they are matched literally and never merged.
  splitSpecial(text) {
    const out = [];
    let i = 0, start = 0;
    while (i < text.length) {
      const hit = text[i] === '<' && this.special.find(s => text.startsWith(s.content, i));
      if (hit) {
        if (i > start) out.push({ text: text.slice(start, i) });
        out.push({ text: hit.content, special: hit.id });
        i += hit.content.length;
        start = i;
      } else i++;
    }
    if (start < text.length) out.push({ text: text.slice(start) });
    return out;
  }

  pretokenize(text) {
    let pieces = [text];
    if (this.splitDigits) {
      pieces = text.split(/(\p{N})/u).filter(p => p.length);
    }
    const words = [];
    for (const p of pieces) for (const m of p.matchAll(SPLIT_RE)) words.push(m[0]);
    return words;
  }

  // Run BPE on one pre-token. Returns the merge trace.
  bpe(word) {
    const cached = this.cache.get(word);
    if (cached) return cached;
    const bytes = Array.from(utf8.encode(word));
    let symbols = bytes.map(b => this.byteToChar[b]);
    const initial = symbols.slice();
    const steps = [];
    while (symbols.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < symbols.length - 1; i++) {
        const r = this.mergeRank.get(symbols[i] + ' ' + symbols[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      const a = symbols[best], b = symbols[best + 1];
      const next = [];
      for (let i = 0; i < symbols.length; i++) {
        if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
          next.push(a + b); i++;
        } else next.push(symbols[i]);
      }
      symbols = next;
      steps.push({ pair: [a, b], rank: bestRank, symbols: symbols.slice() });
    }
    const ids = symbols.map(s => {
      const id = this.vocab.get(s);
      if (id === undefined) throw new Error(`Token not in vocab: ${s}`);
      return id;
    });
    const result = { word, bytes, initial, steps, symbols, ids };
    this.cache.set(word, result);
    return result;
  }

  // Full encode with an explanation trace.
  encodeWithTrace(text) {
    const ids = [];
    const segments = [];
    for (const seg of this.splitSpecial(text)) {
      if (seg.special !== undefined) {
        ids.push(seg.special);
        segments.push({ kind: 'special', text: seg.text, ids: [seg.special] });
        continue;
      }
      for (const w of this.pretokenize(seg.text)) {
        const r = this.bpe(w);
        ids.push(...r.ids);
        segments.push({ kind: 'word', text: w, ...r });
      }
    }
    return { ids, segments };
  }

  encode(text) { return this.encodeWithTrace(text).ids; }

  // Raw bytes of a token (special tokens return their literal text as UTF-8).
  tokenBytes(id) {
    if (this.specialById.has(id)) return utf8.encode(this.specialById.get(id));
    const tok = this.idToToken[id];
    return Uint8Array.from(Array.from(tok, c => this.charToByte.get(c)));
  }

  decode(ids) {
    const parts = [];
    for (const id of ids) parts.push(...this.tokenBytes(id));
    return new TextDecoder().decode(Uint8Array.from(parts));
  }

  // Human-readable form of a vocab entry: the byte-mapped chars shown as-is.
  tokenString(id) { return this.idToToken[id]; }

  // Text form for display: decoded bytes, with invisible chars made visible.
  tokenText(id) {
    if (this.specialById.has(id)) return this.specialById.get(id);
    const s = new TextDecoder('utf-8', { fatal: false }).decode(this.tokenBytes(id));
    return s;
  }
}

export function visibleWhitespace(s) {
  return s.replace(/ /g, '␣').replace(/\n/g, '↵').replace(/\t/g, '⇥');
}
