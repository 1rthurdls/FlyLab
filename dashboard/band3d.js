// FlyLab "fly band at the piano": one fly per part (one whole-brain trial each) stands behind a grand piano.
// The proboscis extends in proportion to that fly's real MN9 rate (MN9 = the proboscis motor neuron in the model);
// when it reaches the key (MN9 > 20 Hz during the note, the singer rule) the key goes down and ONLY THEN the browser
// plays the note (louder for higher MN9). Notes whose MN9 stayed low (e.g. bitter) show a reach that falls short.
// Everything is timed by the simulation; the score (which key, when the sugar pulse comes) is ours.
import * as THREE from "three";

let B = null;
const COL = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"];
const BLACK = new Set([1, 3, 6, 8, 10]);
const isBlack = (m) => BLACK.has(((m % 12) + 12) % 12);
const W = 0.5, KL = 3.0, KH = 0.3;        // white key width, length, height
const THRESH = 20;                        // Hz: the singer rule (a note sounds if MN9 > 20 Hz)
const mat = (c, x = {}) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.5, metalness: 0.05, ...x });

// ------------------------------------------------------------------ models
function makeFly(color) {
  const g = new THREE.Group();
  const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.3, 20, 14), mat(0xb8782a)); thorax.scale.set(1, 1, 1.2); g.add(thorax);
  const abd = new THREE.Mesh(new THREE.SphereGeometry(0.3, 20, 14), mat(0xd9a441)); abd.scale.set(0.95, 0.85, 1.6); abd.position.set(0, -0.02, -0.55); g.add(abd);
  for (let k = 0; k < 3; k++) { const b = new THREE.Mesh(new THREE.TorusGeometry(0.26 - 0.03 * k, 0.03, 6, 20), mat(0x4a3216)); b.position.set(0, -0.02, -0.45 - 0.2 * k); g.add(b); }
  const head = new THREE.Group(); head.position.set(0, 0.08, 0.42); g.add(head);
  head.add(new THREE.Mesh(new THREE.SphereGeometry(0.22, 20, 14), mat(0xc9822b)));
  for (const s of [-1, 1]) { const e = new THREE.Mesh(new THREE.SphereGeometry(0.13, 16, 12), mat(0xc0201a, { roughness: 0.25 })); e.position.set(0.14 * s, 0.06, 0.06); head.add(e); }
  const band = new THREE.Mesh(new THREE.TorusGeometry(0.33, 0.035, 8, 24), mat(new THREE.Color(color).getHex(), { emissive: new THREE.Color(color).getHex(), emissiveIntensity: 0.4 }));
  band.rotation.x = Math.PI / 2; band.position.y = 0.05; g.add(band);   // coloured collar = which part this fly plays
  // proboscis: pivot under the head; rotation.x from tucked (-1.3) to straight down (0); length scales out
  const pivot = new THREE.Group(); pivot.position.set(0, -0.14, 0.12); head.add(pivot);
  const prob = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.06, 1, 10), mat(0x6b3a1a)); prob.position.y = -0.5; pivot.add(prob);
  const tip = new THREE.Mesh(new THREE.SphereGeometry(0.09, 12, 8), mat(0x8a4a22)); tip.position.y = -1.0; pivot.add(tip);
  const wings = [];
  for (const s of [-1, 1]) {
    const p = new THREE.Group(); p.position.set(0.1 * s, 0.25, -0.1);
    const w = new THREE.Mesh(new THREE.PlaneGeometry(0.36, 1.0), mat(0xe8f4ff, { transparent: true, opacity: 0.5, side: THREE.DoubleSide }));
    w.rotation.x = -Math.PI / 2; w.position.set(0.2 * s, 0, -0.35); w.rotation.z = -0.25 * s; p.add(w); g.add(p); wings.push({ p, s });
    for (const k of [-1, 0, 1]) { const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.018, 0.6), mat(0x3d2a12)); leg.position.set(0.32 * s, -0.3, 0.18 * k); leg.rotation.z = 0.5 * s; g.add(leg); }
  }
  g.scale.setScalar(1.15);
  return { g, head, pivot, wings };
}

function makeKeyboard(scene, lo, hi) {
  const keys = new Map(); let wi = 0; const whiteX = new Map();
  for (let m = lo; m <= hi; m++) if (!isBlack(m)) { whiteX.set(m, wi * W); wi++; }
  const width = wi * W;
  for (let m = lo; m <= hi; m++) {
    const black = isBlack(m);
    const x = black ? whiteX.get(m - 1) + W / 2 : whiteX.get(m);
    const pivot = new THREE.Group(); pivot.position.set(x - width / 2 + W / 2, 0, 0); scene.add(pivot);
    const geo = black ? new THREE.BoxGeometry(W * 0.58, KH, KL * 0.62) : new THREE.BoxGeometry(W * 0.94, KH, KL);
    const k = new THREE.Mesh(geo, mat(black ? 0x141414 : 0xf4f2ea, { roughness: black ? 0.35 : 0.6 }));
    k.position.set(0, black ? KH * 1.15 : KH / 2, black ? KL * 0.31 : KL / 2); pivot.add(k);
    keys.set(m, { pivot, k, black, x: pivot.position.x, base: k.material.color.clone(), top: black ? KH * 1.65 : KH });
  }
  // piano body: key bed, back rail (where the flies stand), fallboard, lid
  const body = mat(0x101012, { roughness: 0.25, metalness: 0.3 });
  const bed = new THREE.Mesh(new THREE.BoxGeometry(width + 1.2, 0.5, KL + 0.4), body); bed.position.set(0, -0.26, KL / 2); scene.add(bed);
  for (const s of [-1, 1]) { const cheek = new THREE.Mesh(new THREE.BoxGeometry(0.6, 1.0, KL + 0.4), body); cheek.position.set(s * (width / 2 + 0.3), 0.25, KL / 2); scene.add(cheek); }
  const rail = new THREE.Mesh(new THREE.BoxGeometry(width + 1.2, 0.9, 1.2), body); rail.position.set(0, 0.2, -0.6); scene.add(rail);
  const lid = new THREE.Mesh(new THREE.BoxGeometry(width + 1.2, 0.08, 6), body); lid.position.set(0, 2.6, -3.4); lid.rotation.x = -0.55; scene.add(lid);
  const back = new THREE.Mesh(new THREE.BoxGeometry(width + 1.2, 1.4, 5), body); back.position.set(0, 0.1, -3.7); scene.add(back);
  return { keys, width };
}

// ------------------------------------------------------------------ sound (WebAudio): played by the key press
function playNote(ctx, midi, vel, kind) {
  const f = 440 * Math.pow(2, (midi - 69) / 12), t0 = ctx.currentTime;
  const out = ctx.createGain(); out.connect(ctx.destination);
  const parts = kind === "organ" ? [[0.5, 0.3], [1, 1], [2, 0.5], [3, 0.25], [4, 0.15]]
              : kind === "bass" ? [[1, 1], [2, 0.3]] : [[1, 1], [2, 0.22], [3.98, 0.1]];
  const dur = kind === "organ" ? 1.4 : kind === "bass" ? 1.1 : 1.6;
  out.gain.setValueAtTime(0.0001, t0);
  out.gain.exponentialRampToValueAtTime(0.22 * vel, t0 + (kind === "organ" ? 0.08 : 0.006));
  out.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  for (const [h, a] of parts) {
    const o = ctx.createOscillator(), gn = ctx.createGain(); o.type = "sine"; o.frequency.value = f * h; gn.gain.value = a;
    o.connect(gn); gn.connect(out); o.start(t0); o.stop(t0 + dur + 0.05);
  }
}

// ------------------------------------------------------------------ data
function smooth(x, n) { const o = new Array(x.length).fill(0); let s = 0; for (let i = 0; i < x.length; i++) { s += x[i]; if (i >= n) s -= x[i - n]; o[i] = s / Math.min(i + 1, n); } return o; }

async function loadFlies(data, beh) {
  const parts = [{ name: "melody fly", seq: data.seq }];
  if (data.bass_seq) parts.push({ name: "bass fly", seq: data.bass_seq, bass: true });
  const sims = beh.brain_runs || [];
  const flies = [];
  for (let i = 0; i < parts.length; i++) {
    let mn9 = [], bin = 0.02;
    try { const r = await (await fetch(`/runs/${sims[i]}/result.json`)).json(); mn9 = r.timeseries.MN9; bin = r.timeseries_bin_s; } catch (e) {}
    const sm = smooth(mn9, 5);   // 100 ms window
    const notes = parts[i].seq.filter(e => e.midi != null).map(e => {
      const t0 = e.t, t1 = e.t + (e.dur_s || 0.6);
      let press = null;
      if (e.played) { for (let k = Math.floor(t0 / bin); k < Math.min(sm.length, Math.ceil(t1 / bin)); k++) if (sm[k] > THRESH) { press = k * bin; break; } if (press == null) press = t0 + 0.08; }
      return { ...e, t0, t1, press, pulseEnd: t0 + Math.min(0.8 * (t1 - t0), 0.35) };
    });
    flies.push({ ...parts[i], color: COL[i % COL.length], mn9: sm, bin, notes });
  }
  return flies;
}

// ------------------------------------------------------------------ build
function panel(box, fly, side) {
  const d = document.createElement("div");
  d.style.cssText = `position:absolute;top:6px;${side}:8px;width:196px;padding:5px 8px;border-radius:8px;background:rgba(20,20,19,.82);border:1px solid ${fly.color};font-size:12px;color:#f0efec;pointer-events:none`;
  d.innerHTML = `<div style="font-weight:700;color:${fly.color}">${fly.name}</div>
    <div style="margin-top:3px">taste input: <span class="lamp" style="padding:0 6px;border-radius:6px;background:#2c2c29">—</span></div>
    <div style="margin-top:3px">MN9 (proboscis motor neuron): <b class="hz">0</b> Hz</div>
    <div style="position:relative;height:8px;background:#2c2c29;border-radius:4px;margin-top:3px"><div class="bar" style="height:8px;width:0;border-radius:4px;background:${fly.color}"></div>
      <div style="position:absolute;left:${THRESH / 80 * 100}%;top:-3px;width:2px;height:14px;background:#fab219" title="20 Hz: key reached"></div></div>
    <canvas class="spark" width="178" height="30" style="margin-top:3px;width:178px;height:30px"></canvas>
    <div class="state" style="color:#c3c2b7;margin-top:2px">waiting</div>`;
  box.appendChild(d);
  return { d, lamp: d.querySelector(".lamp"), hz: d.querySelector(".hz"), bar: d.querySelector(".bar"), spark: d.querySelector(".spark"), state: d.querySelector(".state") };
}

async function build(box, data, beh) {
  const flies = await loadFlies(data, beh);
  const all = flies.flatMap(f => f.notes.map(n => n.midi));
  let lo = Math.min(...all), hi = Math.max(...all);
  lo = lo - 2; hi = hi + 2; while (isBlack(lo)) lo--; while (isBlack(hi)) hi++;
  box.innerHTML = ""; box.style.cssText = "position:absolute;inset:0";
  const renderer = new THREE.WebGLRenderer({ antialias: true }); renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.setClearColor(0x1b1a18); box.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x2a2622, 1.5));
  const spot = new THREE.DirectionalLight(0xfff1dc, 1.6); spot.position.set(3, 9, 7); scene.add(spot);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), mat(0x2a2420, { roughness: 0.9 })); floor.rotation.x = -Math.PI / 2; floor.position.y = -0.52; scene.add(floor);
  const kb = makeKeyboard(scene, lo, hi);
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
  const span = Math.max(kb.width, 6);
  camera.position.set(0, 3.4 + span * 0.12, KL + 2.2 + span * 0.42); camera.lookAt(0, 2.0, -0.2);
  const meanM = (f) => f.notes.reduce((a, n) => a + n.midi, 0) / Math.max(1, f.notes.length);
  const mid = flies.reduce((a, f) => a + meanM(f), 0) / flies.length;
  flies.forEach((f, i) => {
    const m = makeFly(f.color); scene.add(m.g); f.model = m;
    const first = f.notes[0]; f.x = first ? kb.keys.get(first.midi).x : 0; f.z = -0.15;
    m.g.position.set(f.x, 1.1, f.z);
    // panel on the side of the keyboard this fly plays (low parts left, high parts right)
    const side = flies.length === 1 ? "left" : (meanM(f) >= mid ? "right" : "left");
    f.ui = panel(box, f, side);
    const c = document.createElement("canvas"); c.width = 256; c.height = 64; const g = c.getContext("2d");
    g.fillStyle = f.color; g.beginPath(); g.roundRect(4, 8, 248, 48, 22); g.fill();
    g.fillStyle = "#fff"; g.font = "bold 28px Segoe UI, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(f.name, 128, 33);
    const tag = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
    tag.scale.set(1.6, 0.4, 1); tag.position.set(0, 1.25, 0); m.g.add(tag);
  });
  const ctrl = document.createElement("div");
  ctrl.style.cssText = "position:absolute;left:50%;bottom:8px;transform:translateX(-50%);display:flex;gap:8px;align-items:center;font-size:12.5px;color:#c3c2b7;background:rgba(20,20,19,.8);padding:4px 10px 4px 4px;border-radius:18px";
  ctrl.innerHTML = `<button style="padding:6px 14px;border-radius:16px;border:1px solid #d95926;background:#d95926;color:#fff;font-weight:700;cursor:pointer">Play</button><span class="lbl"></span>`;
  box.appendChild(ctrl);
  const dur = Math.max(...flies.flatMap(f => f.notes.map(n => n.t1))) + 1.0;
  B = { renderer, scene, camera, box, flies, kb, data, runId: beh.run_id, playing: false, t: 0, t0: 0, ctx: null, dur,
        btn: ctrl.querySelector("button"), lbl: ctrl.querySelector(".lbl"), kind: /organ|pulse/i.test(data.title) ? "organ" : "epiano" };
  B.btn.onclick = () => {
    if (!B.ctx) B.ctx = new (window.AudioContext || window.webkitAudioContext)();
    B.ctx.resume(); B.playing = true; B.t0 = performance.now(); B.t = 0;   // page clock (audio only plays sound)
    for (const f of flies) for (const n of f.notes) n.sounded = false;
    B.btn.textContent = "Replay";
  };
  window.flylabBandRun = beh.run_id; window.flylabCarRun = beh.run_id;   // 3D brain + readout chart follow this clock
  window.flylabSceneTime = () => (B ? Math.min(B.t, B.dur) : null);
  resize();
}

// ------------------------------------------------------------------ per frame
function noteAt(f, t) {   // current note (inside its window) or the next one
  let cur = null, next = null;
  for (const n of f.notes) { if (t >= n.t0 && t < n.t1) cur = n; if (!next && n.t0 >= t) next = n; }
  return { cur, next };
}

function frame() {
  if (B && B.box.offsetParent !== null) {
    if (B.playing) { B.t = (performance.now() - B.t0) / 1000; if (B.t > B.dur) { B.playing = false; B.btn.textContent = "Replay"; } }
    const t = B.t, now = performance.now();
    for (const k of B.kb.keys.values()) { k.pivot.rotation.x *= 0.8; k.k.material.color.lerp(k.base, 0.15); }
    let played = 0, total = 0;
    for (const f of B.flies) {
      const { cur, next } = noteAt(f, t);
      const i = Math.min(f.mn9.length - 1, Math.max(0, Math.floor(t / f.bin)));
      const hz = f.mn9.length ? f.mn9[i] : 0;
      // walk to the key of the current note, or (in the last 40% of a gap/note) toward the next one
      let target = cur || next; if (cur && next && t > cur.t0 + 0.6 * (cur.t1 - cur.t0)) target = next;
      if (target) { const tx = B.kb.keys.get(target.midi).x; const dx = tx - f.x; f.x += dx * 0.12; f.hop = Math.min(1, Math.abs(dx) * 2); }
      const m = f.model;
      m.g.position.set(f.x, 1.1 + 0.25 * (f.hop || 0) * Math.abs(Math.sin(now / 70)), f.z);
      // proboscis extension follows MN9; a reach without a press (MN9 too low) falls short of the key
      let ext = Math.min(1, hz / (2 * THRESH));
      if (cur && !cur.played) ext = Math.min(ext, 0.45);
      const pressing = cur && cur.played && t >= cur.press && t < Math.max(cur.press + 0.2, cur.t1 - 0.05);
      if (pressing) ext = Math.max(ext, 0.62);
      f.ext = (f.ext || 0) + (ext - (f.ext || 0)) * 0.35;
      m.pivot.rotation.x = -1.3 * (1 - f.ext); m.pivot.scale.y = 0.35 + 0.75 * f.ext;
      m.head.rotation.x = 0.35 * f.ext;
      for (const w of m.wings) w.p.rotation.z = 0.15 * w.s * Math.sin(now / 90);
      if (pressing) {
        const key = B.kb.keys.get(cur.midi); key.pivot.rotation.x = 0.06; key.k.material.color.set(f.color);
        if (B.playing && !cur.sounded) { cur.sounded = true; playNote(B.ctx, cur.midi, Math.min(1, 0.3 + cur.mn9_hz / 90), f.bass ? "bass" : B.kind); }
      }
      // activation panel
      const sugar = cur && t < cur.pulseEnd, bitter = sugar && cur.bitter;
      f.ui.lamp.textContent = bitter ? "SUGAR + BITTER" : sugar ? "SUGAR" : "—";
      f.ui.lamp.style.background = bitter ? "#6d28d9" : sugar ? "#b7791f" : "#2c2c29";
      f.ui.hz.textContent = hz.toFixed(0); f.ui.bar.style.width = Math.min(100, hz / 80 * 100) + "%";
      f.ui.state.textContent = pressing ? `TAP ${cur.name} (MN9 ${cur.mn9_hz} Hz)` : cur && !cur.played ? `reach falls short (MN9 ${cur.mn9_hz} Hz < 20)` : cur ? "reaching…" : "waiting";
      const g = f.ui.spark.getContext("2d"), Wd = 178, H = 30, span = 3.0; g.clearRect(0, 0, Wd, H);
      g.strokeStyle = "#fab219"; g.beginPath(); g.moveTo(0, H - THRESH / 80 * H); g.lineTo(Wd, H - THRESH / 80 * H); g.stroke();
      g.strokeStyle = f.color; g.beginPath();
      for (let x = 0; x < Wd; x++) { const tt = t - span + x / Wd * span; const k = Math.floor(tt / f.bin); const v = k >= 0 && k < f.mn9.length ? f.mn9[k] : 0; const y = H - Math.min(1, v / 80) * H; x ? g.lineTo(x, y) : g.moveTo(x, y); }
      g.stroke();
      for (const n of f.notes) { total++; if (n.played && t >= n.press) played++; }
    }
    B.lbl.textContent = B.playing || B.t > 0 ? `t = ${t.toFixed(1)} s · ${played} notes tapped · score is ours, every tap is the fly's MN9` : `${B.data.title} · press Play (sound starts on each tap)`;
    B.renderer.render(B.scene, B.camera);
  }
  requestAnimationFrame(frame);
}

function resize() {
  if (!B) return;
  const w = B.box.clientWidth, h = B.box.clientHeight; if (!w || !h) return;
  B.renderer.setSize(w, h, false); B.renderer.domElement.style.width = w + "px"; B.renderer.domElement.style.height = h + "px";
  B.camera.aspect = w / h; B.camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

window.flylabShowBand = async (box, data, beh) => {
  if (B && B.runId === beh.run_id && B.box === box) return;
  window.flylabHideBand();
  await build(box, data, beh);
};
window.flylabHideBand = () => {
  if (!B) return;
  try { B.renderer.dispose(); if (B.ctx) B.ctx.close(); } catch (e) {}
  if (window.flylabBandRun === B.runId) window.flylabBandRun = null;
  B = null;
};
requestAnimationFrame(frame);
