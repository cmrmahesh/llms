// The "device": one WebAssembly linear memory holding every weight plus scratch
// activations, and a SIMD matmul kernel that streams over it. This mirrors a GPU:
// weights are copied once into device memory (HBM), and each layer is a kernel
// launch that reads weights from that memory.

export async function createRuntime(wasmUrl) {
  try {
    const bytes = await (await fetch(wasmUrl)).arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes);
    return new WasmRuntime(instance);
  } catch (e) {
    console.warn('WASM SIMD unavailable, falling back to JS kernels', e);
    return new JsRuntime();
  }
}

class WasmRuntime {
  constructor(instance) {
    this.kind = 'wasm-simd';
    this.ex = instance.exports;
    this.memory = this.ex.memory;
    this.top = 1024; // keep address 0 unused
  }
  // Grow memory once up-front: growing detaches every existing view.
  reserve(floats) {
    const need = this.top + floats * 4 + 64 * 1024;
    const have = this.memory.buffer.byteLength;
    if (need > have) this.memory.grow(Math.ceil((need - have) / 65536));
  }
  alloc(floats) {
    this.top = (this.top + 15) & ~15;
    if (this.top + floats * 4 > this.memory.buffer.byteLength) throw new Error('arena overflow: reserve() more');
    const view = new Float32Array(this.memory.buffer, this.top, floats);
    this.top += floats * 4;
    return view;
  }
  get bytesUsed() { return this.top; }
  matmul(out, x, W, T, inD, outD) {
    this.ex.matmul(out.byteOffset, x.byteOffset, W.byteOffset, T, inD, outD);
  }
}

class JsRuntime {
  constructor() { this.kind = 'js'; this.top = 0; }
  reserve() {}
  alloc(floats) { this.top += floats * 4; return new Float32Array(floats); }
  get bytesUsed() { return this.top; }
  matmul(out, x, W, T, inD, outD) {
    for (let o = 0; o < outD; o++) {
      const w = o * inD;
      for (let t = 0; t < T; t++) {
        const xo = t * inD;
        let s = 0;
        for (let i = 0; i < inD; i++) s += W[w + i] * x[xo + i];
        out[t * outD + o] = s;
      }
    }
  }
}
