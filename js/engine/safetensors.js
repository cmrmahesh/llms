// Minimal safetensors reader. Format: [u64 little-endian header length N]
// [N bytes of JSON header] [raw tensor bytes]. The header maps every tensor name
// to {dtype, shape, data_offsets: [begin, end]} relative to the end of the header.

export function parseHeader(buffer) {
  const view = new DataView(buffer);
  const n = Number(view.getBigUint64(0, true));
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, n)));
  const meta = json.__metadata__ || {};
  delete json.__metadata__;
  const dataStart = 8 + n;
  const tensors = Object.entries(json).map(([name, t]) => ({
    name,
    dtype: t.dtype,
    shape: t.shape,
    begin: dataStart + t.data_offsets[0],
    end: dataStart + t.data_offsets[1],
  }));
  return { headerBytes: n, dataStart, meta, tensors };
}

// Fetch only the header with two HTTP Range requests (8 bytes, then the JSON).
export async function fetchHeader(url) {
  const r1 = await fetch(url, { headers: { Range: 'bytes=0-7' } });
  if (r1.status !== 206) throw new Error('Range requests not supported');
  const lenBuf = await r1.arrayBuffer();
  const n = Number(new DataView(lenBuf).getBigUint64(0, true));
  const r2 = await fetch(url, { headers: { Range: `bytes=0-${7 + n}` } });
  const buf = await r2.arrayBuffer();
  return parseHeader(buf);
}

const BYTES = { BF16: 2, F16: 2, F32: 4 };

export function dtypeBytes(dtype) { return BYTES[dtype] ?? 0; }

export function numel(t) { return t.shape.reduce((a, b) => a * b, 1); }

// Convert one tensor to float32, into `out` if given (e.g. a view into device memory).
export function toFloat32(buffer, t, out) {
  const count = numel(t);
  out ??= new Float32Array(count);
  if (t.dtype === 'F32') {
    out.set(new Float32Array(buffer.slice(t.begin, t.end)));
    return out;
  }
  const src = new Uint16Array(buffer, t.begin, count);
  if (t.dtype === 'BF16') {
    // bfloat16 is the top 16 bits of a float32: shift left and reinterpret.
    const u = new Uint32Array(out.buffer, out.byteOffset, count);
    for (let i = 0; i < count; i++) u[i] = src[i] << 16;
  } else if (t.dtype === 'F16') {
    for (let i = 0; i < count; i++) out[i] = f16(src[i]);
  } else {
    throw new Error(`Unsupported dtype ${t.dtype}`);
  }
  return out;
}

function f16(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (m / 1024);
  if (e === 31) return m ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + m / 1024);
}
