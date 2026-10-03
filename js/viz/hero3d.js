// Hero: every weight tensor of the model as a tile, to scale, textured with real
// values once loaded. Layers stack upward; the embedding table is the long runway
// underneath. Token lookups light up rows of the runway; forward passes sweep
// through the layers.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { diverging, fmtInt, fmtBytes } from '../util.js';

const SCALE = 1 / 900;   // scene units per matrix row/column
const LAYER_GAP = 0.15;
const PAPER = 0xedf0f3;
const SM = 0xa8126f;

export function createHero(container, ctx) {
  const { S } = ctx;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  container.append(renderer.domElement);
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(PAPER, 18, 52);
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 200);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableZoom = false;
  controls.enablePan = false;
  controls.enableDamping = true;
  controls.minPolarAngle = 0.25;
  controls.maxPolarAngle = 1.45;

  // HTML labels pinned to points in the scene
  const labelLayer = document.createElement('div');
  labelLayer.className = 'stage-labels';
  labelLayer.setAttribute('aria-hidden', 'true');
  container.append(labelLayer);
  let labels = []; // { el, pos: Vector3 }
  function addLabel(text, pos, cls = '') {
    const el = document.createElement('span');
    el.className = `stage-label ${cls}`;
    el.textContent = text;
    labelLayer.append(el);
    labels.push({ el, pos });
  }
  function placeLabels() {
    const w = renderer.domElement.clientWidth, hgt = renderer.domElement.clientHeight;
    const v = new THREE.Vector3();
    for (const l of labels) {
      v.copy(l.pos).project(camera);
      const hidden = v.z > 1 || v.x < -1.1 || v.x > 1.1 || v.y < -1.1 || v.y > 1.1;
      l.el.style.display = hidden ? 'none' : '';
      l.el.style.left = `${(v.x + 1) / 2 * w}px`;
      l.el.style.top = `${(1 - v.y) / 2 * hgt}px`;
    }
  }

  const root = new THREE.Group();
  scene.add(root);
  let tiles = [];          // { mesh, name, shape, layer }
  let sweep = null, markers = null, runway = null;
  let stackH = 0, rowW = 0;
  let sweepAnim = null;    // { t0, from, to, dur }
  let builtFor = null;
  let edgeMaterial = null; // shared edge material: faded once real textures arrive

  function shapes() {
    const d = ctx.dims();
    if (!d) return null;
    const kvw = d.KVH * d.HD;
    // [name suffix, x-extent (non-D dim), transpose?]
    return { d, row: [['self_attn.q_proj.weight', d.D], ['self_attn.k_proj.weight', kvw], ['self_attn.v_proj.weight', kvw], ['self_attn.o_proj.weight', d.D], ['mlp.gate_proj.weight', d.I], ['mlp.up_proj.weight', d.I], ['mlp.down_proj.weight', d.I, true]] };
  }

  function clear() {
    root.traverse(o => { o.geometry?.dispose(); for (const m of [].concat(o.material || [])) { m.map?.dispose(); m.dispose(); } });
    root.clear();
    tiles = [];
  }

  function build() {
    const sh = shapes();
    if (!sh) return;
    const key = S.meta.id;
    if (builtFor === key) return;
    builtFor = key;
    clear();
    const { d, row } = sh;
    const gap = 0.06;
    rowW = row.reduce((a, r) => a + r[1] * SCALE + gap, -gap);
    const depth = d.D * SCALE;
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x9aa6b8, transparent: true, opacity: 0.6 });
    edgeMaterial = edgeMat;
    for (let l = 0; l < d.L; l++) {
      let x = -rowW / 2;
      const y = 0.12 + l * LAYER_GAP;
      for (const [suffix, w, transpose] of row) {
        const width = w * SCALE;
        const geo = new THREE.BoxGeometry(width, 0.03, depth);
        // Box faces: +x, -x, +y (top), -y, +z, -z. Real weights go on the top face only;
        // the sides are coloured like chapter 1's split bar: attention light, feed-forward dark.
        const side = new THREE.MeshBasicMaterial({ color: suffix.startsWith('mlp') ? 0x2f4570 : 0x7f95b8 });
        const top = new THREE.MeshBasicMaterial({ color: 0xf6f8fa });
        const mesh = new THREE.Mesh(geo, [side, side, top, side, side, side]);
        mesh.position.set(x + width / 2, y, 0);
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgeMat);
        mesh.add(edges);
        root.add(mesh);
        const name = `model.layers.${l}.${suffix}`;
        tiles.push({ mesh, name, layer: l, transpose, w });
        x += width + gap;
      }
    }
    stackH = 0.12 + d.L * LAYER_GAP;
    // Embedding runway: V rows long, D wide, starting under the stack.
    const len = d.V * SCALE;
    const rgeo = new THREE.BoxGeometry(len, 0.03, depth);
    const rmat = new THREE.MeshBasicMaterial({ color: 0xdfe5ec });
    runway = new THREE.Mesh(rgeo, rmat);
    runway.position.set(-rowW / 2 + len / 2, -0.04, 0);
    runway.add(new THREE.LineSegments(new THREE.EdgesGeometry(rgeo), edgeMat));
    root.add(runway);
    tiles.push({ mesh: runway, name: 'model.embed_tokens.weight', layer: -1, w: d.V, runway: true });
    // Forward-pass sweep slab
    sweep = new THREE.Mesh(new THREE.BoxGeometry(rowW + 0.16, 0.05, depth + 0.16),
      new THREE.MeshBasicMaterial({ color: SM, transparent: true, opacity: 0, depthWrite: false }));
    root.add(sweep);
    markers = new THREE.Group();
    root.add(markers);
    // Labels
    labelLayer.textContent = '';
    labels = [];
    const attnW = row.slice(0, 4).reduce((a, r) => a + r[1] * SCALE + gap, -gap);
    const left = -rowW / 2;
    addLabel('Layer 0 (first)', new THREE.Vector3(rowW / 2 + 0.12, 0.12, depth / 2), 'right');
    addLabel(`Layer ${d.L - 1} (last)`, new THREE.Vector3(rowW / 2 + 0.12, stackH - LAYER_GAP + 0.12, depth / 2), 'right');
    addLabel('Attention', new THREE.Vector3(left + attnW / 2, stackH + 0.12, 0));
    addLabel('Feed-forward', new THREE.Vector3(left + attnW + gap + (rowW - attnW - gap) / 2, stackH + 0.12, 0));
    addLabel(`Vocabulary table, ${fmtInt(d.V)} rows →`, new THREE.Vector3(-rowW / 2 + 2.2, -0.1, depth / 2 + 0.2), 'below');
    frame();
    applyTextures();
    placeMarkers();
  }

  function frame() {
    // Centre the stack, leave the runway to recede into the fog on the right.
    const target = new THREE.Vector3(0.9, stackH * 0.45, 0);
    controls.target.copy(target);
    const end = new THREE.Vector3(-5.5, stackH + 7.5, 13);
    if (reduced) { camera.position.copy(end); }
    else {
      const start = new THREE.Vector3(-2, stackH + 16, 4);
      camera.position.copy(start);
      intro = { t0: performance.now(), start, end, dur: 1800 };
    }
    camera.lookAt(target);
    controls.update();
  }
  let intro = null;

  function texture(p, transpose, maxW = 128) {
    // Grid is [rows=out, cols=in]; the tile's x-extent is the non-D dim.
    let gw = transpose ? p.gc : p.gr, gh = transpose ? p.gr : p.gc;
    const cv = document.createElement('canvas');
    cv.width = Math.min(gw, maxW); cv.height = gh;
    const g = cv.getContext('2d');
    const img = g.createImageData(cv.width, cv.height);
    const scale = 1.5 * p.std || 1; // more contrast than the 2-D heatmaps: this view is about texture, not reading values
    for (let y = 0; y < cv.height; y++) {
      for (let x = 0; x < cv.width; x++) {
        const gx = Math.floor(x * gw / cv.width);
        const v = transpose ? p.grid[y * p.gc + gx] : p.grid[gx * p.gc + y];
        const c = diverging(v / scale);
        const i = (y * cv.width + x) * 4;
        img.data[i] = c[0]; img.data[i + 1] = c[1]; img.data[i + 2] = c[2]; img.data[i + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);
    const t = new THREE.CanvasTexture(cv);
    t.magFilter = THREE.NearestFilter;
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }

  function applyTextures() {
    if (!S.previews) return;
    const byName = new Map(S.previews.map(p => [p.name, p]));
    for (const t of tiles) {
      const p = byName.get(t.name);
      if (!p?.grid) continue;
      if (t.runway) {
        // The runway is very long: lay a strip texture along it.
        const cv = document.createElement('canvas');
        cv.width = p.gr; cv.height = p.gc;
        const g = cv.getContext('2d');
        const img = g.createImageData(cv.width, cv.height);
        for (let x = 0; x < p.gr; x++) for (let y = 0; y < p.gc; y++) {
          const c = diverging(p.grid[x * p.gc + y] / (1.5 * p.std));
          const i = (y * cv.width + x) * 4;
          img.data[i] = c[0]; img.data[i + 1] = c[1]; img.data[i + 2] = c[2]; img.data[i + 3] = 255;
        }
        g.putImageData(img, 0, 0);
        const tex = new THREE.CanvasTexture(cv);
        tex.magFilter = THREE.NearestFilter;
        tex.colorSpace = THREE.SRGBColorSpace;
        t.mesh.material.map = tex;
        t.mesh.material.color.set(0xffffff);
        t.mesh.material.needsUpdate = true;
        continue;
      }
      const top = t.mesh.material[2];
      top.map = texture(p, t.transpose);
      top.color.set(0xffffff);
      top.needsUpdate = true;
    }
    if (edgeMaterial) edgeMaterial.opacity = 0.18;
    wake();
  }

  function placeMarkers() {
    if (!markers) return;
    markers.clear();
    const d = ctx.dims();
    if (!d) return;
    const ids = new Set(S.ids);
    const run = S.run;
    if (run) for (const s of run.steps) ids.add(s.token);
    const latest = run?.steps.at(-1)?.token;
    const geo = new THREE.BoxGeometry(0.035, 0.07, d.D * SCALE + 0.04);
    const mat = new THREE.MeshBasicMaterial({ color: 0x0d7480 });
    const hot = new THREE.MeshBasicMaterial({ color: SM });
    for (const id of ids) {
      const m = new THREE.Mesh(geo, id === latest ? hot : mat);
      m.position.set(-rowW / 2 + id * SCALE, 0.0, 0);
      markers.add(m);
    }
    wake();
  }

  // --- render loop (sleeps when idle or offscreen)
  let awake = false, visible = true;
  function wake() { if (!awake && visible) { awake = true; requestAnimationFrame(tick); } }
  function tick(now) {
    let busy = false;
    if (intro) {
      const f = Math.min(1, (now - intro.t0) / intro.dur);
      const e = 1 - (1 - f) ** 3;
      camera.position.lerpVectors(intro.start, intro.end, e);
      camera.lookAt(controls.target);
      if (f >= 1) intro = null; else busy = true;
    }
    if (sweepAnim && sweep) {
      const f = Math.min(1, (now - sweepAnim.t0) / sweepAnim.dur);
      const y = sweepAnim.from + (sweepAnim.to - sweepAnim.from) * f;
      sweep.position.y = y;
      sweep.material.opacity = 0.35 * (1 - Math.max(0, f - 0.85) / 0.15);
      if (f >= 1) { sweepAnim = null; sweep.material.opacity = 0; } else busy = true;
    }
    busy = controls.update() || busy;
    renderer.render(scene, camera);
    placeLabels();
    if (busy || dragging) requestAnimationFrame(tick); else awake = false;
  }
  let dragging = false;
  controls.addEventListener('start', () => { dragging = true; intro = null; wake(); });
  controls.addEventListener('end', () => { dragging = false; wake(); });
  controls.addEventListener('change', wake);

  function resize() {
    const r = container.getBoundingClientRect();
    if (!r.width) return;
    renderer.setSize(r.width, r.height, false);
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    camera.aspect = r.width / r.height;
    camera.updateProjectionMatrix();
    wake();
  }
  new ResizeObserver(resize).observe(container);
  new IntersectionObserver(es => { visible = es[0].isIntersecting; if (visible) wake(); }).observe(container);

  // --- hover labels
  const ray = new THREE.Raycaster();
  const hover = document.getElementById('stage-hover');
  renderer.domElement.addEventListener('pointermove', e => {
    if (dragging) return;
    const r = renderer.domElement.getBoundingClientRect();
    const v = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(v, camera);
    const hit = ray.intersectObjects(tiles.map(t => t.mesh), false)[0];
    const t = hit && tiles.find(x => x.mesh === hit.object);
    if (!t) { hover.hidden = true; return; }
    const d = ctx.dims();
    const shape = t.runway ? [d.V, d.D] : t.name.includes('down_proj') ? [d.D, t.w] : [t.w, d.D];
    const n = shape[0] * shape[1];
    hover.hidden = false;
    hover.innerHTML = `<code>${t.name.replace('model.', '')}</code><br>${shape.join(' × ')} = ${fmtInt(n)} numbers · ${fmtBytes(n * 2)} in bf16${t.runway ? `<br>The vocabulary table: one row per token${d.tied ? ', reused as the LM head' : ''}.` : ''}`;
  });
  renderer.domElement.addEventListener('pointerleave', () => { hover.hidden = true; });

  function startSweep(fromLayer, toLayer, dur) {
    if (!sweep || reduced) return;
    sweepAnim = { t0: performance.now(), from: 0.12 + fromLayer * LAYER_GAP, to: 0.12 + toLayer * LAYER_GAP, dur };
    wake();
  }

  return {
    update(reason, data) {
      if (reason === 'model' || reason === 'header') { builtFor = null; build(); }
      if (reason === 'weights') applyTextures();
      if (reason === 'tokens' || reason === 'step' || reason === 'branch') placeMarkers();
      if (reason === 'layer' && sweep) {
        sweep.position.y = 0.12 + data.l * LAYER_GAP;
        sweep.material.opacity = reduced ? 0.25 : 0.35;
        if (data.l === data.L - 1) startSweep(data.l, data.l, 400);
        wake();
      }
      if (reason === 'step') {
        const d = ctx.dims();
        startSweep(0, d.L - 1, 260);
      }
    },
  };
}
