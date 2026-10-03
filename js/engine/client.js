// Promise wrapper around the engine worker.

export class EngineClient {
  constructor() {
    this.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    this.pending = new Map();
    this.nextId = 1;
    this.listeners = { progress: [], layer: [] };
    this.worker.onmessage = e => {
      const m = e.data;
      if (m.type === 'progress' || m.type === 'layer') {
        for (const f of this.listeners[m.type]) f(m);
        return;
      }
      const p = this.pending.get(m.id);
      if (!p) return;
      this.pending.delete(m.id);
      if (m.type === 'error') p.reject(new Error(m.error)); else p.resolve(m);
    };
    this.worker.onerror = e => {
      for (const p of this.pending.values()) p.reject(new Error(e.message || 'Worker crashed'));
      this.pending.clear();
    };
  }
  on(type, f) { this.listeners[type].push(f); }
  call(type, payload = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ type, id, ...payload });
    });
  }
}
