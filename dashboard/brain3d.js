// FlyLab 3D brain: all 138,639 FlyWire neurons as points; neurons glow when they spike.
// Time source: the fly video (behaviour runs, video time x play speed = simulated time),
// or a looping 0.25x clock for plain simulations. Data: /api/brain/positions, /api/brain/activity.
import * as THREE from "three";
import { OrbitControls } from "/vendor/OrbitControls.js";

const ROLE = { other: 0, stim: 1, output: 2, silenced: 3 };
const ROLE_RGB = [[0.94, 0.93, 0.91], [0.22, 0.53, 0.90], [0.85, 0.35, 0.15], [0.90, 0.40, 0.40]];
const BASE = [0.0, 0.35, 0.45, 1.0];        // resting brightness by role (others invisible at rest)
const BASE_SIZE = [0.0, 3.0, 4.5, 7.0];
const TAU_S = 0.02;                          // glow decay (simulated seconds)

const box = document.getElementById("brain3d");
const hud = document.getElementById("brain3dhud");
const labelLayer = document.getElementById("brain3dlabels");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x1a1a19, 1);
box.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 50);
camera.position.set(0.0, 0.3, 2.05);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(-0.06, 0.04, 0.06);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.5;

const vsh = `attribute float size; attribute vec3 color; varying vec3 vColor; uniform float pr;
  void main(){ vColor = color; vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = size * pr * (2.6 / -mv.z); gl_Position = projectionMatrix * mv; }`;
const fsh = `varying vec3 vColor; void main(){ vec2 c = gl_PointCoord - 0.5; float d = length(c);
  if (d > 0.5) discard; float a = smoothstep(0.5, 0.05, d); gl_FragColor = vec4(vColor * a, a); }`;
const mat = () => new THREE.ShaderMaterial({ vertexShader: vsh, fragmentShader: fsh, transparent: true,
  depthWrite: false, blending: THREE.AdditiveBlending, uniforms: { pr: { value: renderer.getPixelRatio() } } });

let P = null, N = 0;          // positions
let fg = null, act = null;    // foreground points + activity state
let currentRun = null, loading = false, clock0 = performance.now();

async function loadPositions() {
  const buf = await (await fetch("/api/brain/positions")).arrayBuffer();
  P = new Float32Array(buf); N = P.length / 3;
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(P, 3));
  const col = new Float32Array(N * 3).fill(0.10), size = new Float32Array(N).fill(1.7);
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.setAttribute("size", new THREE.BufferAttribute(size, 1));
  scene.add(new THREE.Points(g, mat()));
}

async function loadActivity(runId) {
  loading = true;
  try {
    const a = await (await fetch(`/api/brain/activity?run_id=${encodeURIComponent(runId)}`)).json();
    if (a.error) return;
    const role = new Map();
    for (const i of a.stim) role.set(i, ROLE.stim);
    for (const i of a.output) role.set(i, ROLE.output);
    for (const i of a.silenced) role.set(i, ROLE.silenced);
    const ids = Array.from(new Set([...a.n, ...role.keys()]));
    const local = new Map(ids.map((n, k) => [n, k]));
    const M = ids.length, pos = new Float32Array(M * 3), r = new Uint8Array(M);
    ids.forEach((n, k) => { pos.set(P.subarray(3 * n, 3 * n + 3), 3 * k); r[k] = role.get(n) || 0; });
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(M * 3), 3));
    g.setAttribute("size", new THREE.BufferAttribute(new Float32Array(M), 1));
    if (fg) { scene.remove(fg); fg.geometry.dispose(); }
    fg = new THREE.Points(g, mat()); scene.add(fg);
    // labels: centroid of each output group
    const groups = {};
    for (const [i, lab] of Object.entries(a.output_labels)) (groups[lab] = groups[lab] || []).push(+i);
    labelLayer.innerHTML = "";
    const labels = Object.entries(groups).map(([lab, idx]) => {
      const c = new THREE.Vector3();
      idx.forEach(i => c.add(new THREE.Vector3(P[3 * i], P[3 * i + 1], P[3 * i + 2])));
      c.divideScalar(idx.length);
      const el = document.createElement("div"); el.className = "b3label"; el.textContent = lab; labelLayer.appendChild(el);
      return { c, el, idx: idx.map(i => local.get(i)) };
    });
    act = { a, M, r, local, labels, sn: Int32Array.from(a.n, n => local.get(n)),
            st: Float64Array.from(a.t, s => s * a.step_ms / 1000), last: new Float64Array(M).fill(-1e9),
            ptr: 0, prevT: -1 };
    clock0 = performance.now();
  } finally { loading = false; }
}

function simTime() {
  if (!act) return 0;
  const a = act.a, v = document.getElementById("video");
  if (a.kind === "behavior" && window.flylabCarRun === a.run_id && window.flylabSceneTime) {
    const ct = window.flylabSceneTime(); if (ct != null) return Math.min(ct, a.duration_s);
  }
  if (a.kind === "behavior" && v && v.src.includes(a.run_id) && !isNaN(v.currentTime))
    return Math.min(v.currentTime * (a.play_speed || 0.25), a.duration_s);
  return ((performance.now() - clock0) / 1000 * 0.25) % a.duration_s;   // plain simulation: 0.25x loop
}

function update() {
  if (!act || !fg) return;
  const t = simTime();
  if (t < act.prevT) { act.ptr = 0; act.last.fill(-1e9); }          // looped / seeked back
  while (act.ptr < act.st.length && act.st[act.ptr] <= t) { act.last[act.sn[act.ptr]] = act.st[act.ptr]; act.ptr++; }
  act.prevT = t;
  const col = fg.geometry.attributes.color.array, size = fg.geometry.attributes.size.array;
  let firing = 0;
  for (let k = 0; k < act.M; k++) {
    const dt = t - act.last[k], glow = dt >= 0 ? Math.exp(-dt / TAU_S) : 0, ro = act.r[k];
    if (glow > 0.3) firing++;
    const b = BASE[ro] + 1.7 * glow, rgb = ROLE_RGB[ro];
    col[3 * k] = rgb[0] * b; col[3 * k + 1] = rgb[1] * b; col[3 * k + 2] = rgb[2] * b;
    size[k] = BASE_SIZE[ro] + 9 * glow;
  }
  fg.geometry.attributes.color.needsUpdate = true;
  fg.geometry.attributes.size.needsUpdate = true;
  const a = act.a;
  hud.innerHTML = `<b>${a.label.replace(/_/g, " ")}</b> · t = ${t.toFixed(2)} s · <b>${firing}</b> neurons firing now`
    + (a.contact_s != null ? (t >= a.contact_s ? " · <span style='color:#e66767'>sugar contact</span>" : "") : "")
    + `<br><span class="muted">${a.n_neurons_firing.toLocaleString()} of 138,639 neurons fire in this trial · one dot per neuron</span>`;
  // project labels
  const w = box.clientWidth, h = box.clientHeight;
  for (const L of act.labels) {
    const p = L.c.clone().project(camera);
    const on = L.idx.some(k => t - act.last[k] < 0.03);
    L.el.style.left = `${(p.x * 0.5 + 0.5) * w + 8}px`; L.el.style.top = `${(-p.y * 0.5 + 0.5) * h - 8}px`;
    L.el.classList.toggle("on", on);
  }
}

function resize() {
  const w = box.clientWidth, h = box.clientHeight; if (!w || !h) return;
  renderer.setSize(w, h, false); renderer.domElement.style.width = w + "px"; renderer.domElement.style.height = h + "px";
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(box);

// follow whichever of (latest simulation, latest behaviour) is newest
setInterval(() => {
  const s = window.flylabState || {}, sim = s.sim, beh = s.beh;
  const pick = (beh && (!sim || beh._mtime >= sim._mtime)) ? beh : sim;
  if (P && pick && !loading && pick.run_id !== currentRun) { currentRun = pick.run_id; loadActivity(pick.run_id); }
}, 500);

function loop() { controls.update(); update(); renderer.render(scene, camera); requestAnimationFrame(loop); }
loadPositions().then(() => { resize(); loop(); });
