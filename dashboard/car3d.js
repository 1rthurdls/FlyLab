// FlyLab car scene: replays a drive_car trajectory (from the whole-brain simulation) at 1x speed.
// Exposes window.flylabSceneTime() so the readout chart and the 3D brain stay in sync.
import * as THREE from "three";

let state = null;   // {renderer, scene, camera, car, wheels, traj, t0, runId, ...}

function mat(c, extra = {}) { return new THREE.MeshStandardMaterial({ color: c, roughness: 0.55, metalness: 0.1, ...extra }); }

function makeFlyDriver() {
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.SphereGeometry(0.28, 20, 14), mat(0xd58a2e));
  body.scale.set(1.5, 1.0, 1.0); g.add(body);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.2, 20, 14), mat(0xc9822b)); head.position.set(0.42, 0.12, 0); g.add(head);
  for (const s of [-1, 1]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.11, 16, 12), mat(0xd12a1a, { roughness: 0.3 }));
    eye.position.set(0.5, 0.18, 0.13 * s); g.add(eye);
    const wing = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.28), mat(0xffffff, { transparent: true, opacity: 0.45, side: THREE.DoubleSide }));
    wing.rotation.x = -Math.PI / 2; wing.rotation.z = 0.35 * s; wing.position.set(-0.3, 0.22, 0.18 * s); g.add(wing);
    for (const k of [0, 1]) {   // two "arms" on the wheel
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.55), mat(0x8a5a22));
      leg.rotation.z = Math.PI / 2.4; leg.position.set(0.55, 0.0 - 0.05 * k, 0.15 * s); g.add(leg);
    }
  }
  return g;
}

function makeCar() {
  const car = new THREE.Group();
  const shell = new THREE.Mesh(new THREE.BoxGeometry(4.2, 0.7, 1.9), mat(0xd95926)); shell.position.y = 0.75; car.add(shell);
  const hood = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.25, 1.8), mat(0xe8703c)); hood.position.set(1.35, 1.2, 0); car.add(hood);
  const seat = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.7, 0.8), mat(0x2b2b28)); seat.position.set(-0.6, 1.35, 0.35); car.add(seat);
  const wheelRing = new THREE.Mesh(new THREE.TorusGeometry(0.28, 0.04, 8, 20), mat(0x222222));
  wheelRing.position.set(0.25, 1.55, 0.35); wheelRing.rotation.y = Math.PI / 2; wheelRing.rotation.x = 0.4; car.add(wheelRing);
  const wind = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.5, 1.7), mat(0x9fd3ff, { transparent: true, opacity: 0.35 }));
  wind.position.set(0.55, 1.45, 0); wind.rotation.z = -0.35; car.add(wind);
  const fly = makeFlyDriver(); fly.position.set(-0.35, 1.75, 0.35); car.add(fly);
  const wheels = [];
  for (const [x, z, front] of [[1.35, 0.95, 1], [1.35, -0.95, 1], [-1.35, 0.95, 0], [-1.35, -0.95, 0]]) {
    const pivot = new THREE.Group(); pivot.position.set(x, 0.42, z);
    const w = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.42, 0.3, 20), mat(0x1c1c1c));
    w.rotation.x = Math.PI / 2; pivot.add(w); car.add(pivot); wheels.push({ pivot, w, front });
  }
  return { car, wheels, fly };
}

function makeLabel(text, color = "#1a1a19", bg = "#fab219") {
  const c = document.createElement("canvas"); c.width = 512; c.height = 160;
  const g = c.getContext("2d"); g.fillStyle = bg; g.fillRect(0, 0, 512, 160);
  g.fillStyle = color; g.font = "bold 72px Segoe UI, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(text, 256, 84);
  const m = new THREE.Mesh(new THREE.PlaneGeometry(4, 1.25), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), side: THREE.DoubleSide }));
  return m;
}

function build(box, data) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x20201e);
  box.innerHTML = ""; box.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x20201e, 40, 110);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x333333, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(10, 25, 8); scene.add(sun);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), mat(0x3b3d40, { roughness: 0.95 }));
  ground.rotation.x = -Math.PI / 2; scene.add(ground);
  const grid = new THREE.GridHelper(400, 100, 0x55585c, 0x47494d); grid.position.y = 0.01; scene.add(grid);
  if (data.mode === "course") buildCourse(scene, data);
  else buildStation(scene, data);
  // full planned path (faint) + live trail
  const traj = data.trajectory;
  const pts = traj.map(r => new THREE.Vector3(r[1], 0.05, -r[2]));
  const ghost = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineDashedMaterial({ color: 0x8a897f, dashSize: 0.6, gapSize: 0.4 }));
  ghost.computeLineDistances(); if (data.mode !== "course") scene.add(ghost);
  const trailGeom = new THREE.BufferGeometry().setFromPoints(pts);
  const trail = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ color: 0x3987e5 })); trail.position.y = 0.04; scene.add(trail);
  const { car, wheels, fly } = makeCar(); scene.add(car);
  const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 300);
  const rays = data.mode === "course" ? makeRays(scene, data) : null;
  state = { renderer, scene, camera, car, wheels, fly, traj, trailGeom, t0: performance.now(), box,
            dur: traj[traj.length - 1][0], camPos: new THREE.Vector3(-8, 6, 0), runId: data.runId,
            hud: document.getElementById("carhud"), maneuvers: data.maneuvers || [], data, rays };
  resize();
}

function buildStation(scene, data) {
  // sugar station (world x, z = -y)
  const [sx, sy] = data.station, R = data.station_radius;
  const pad = new THREE.Mesh(new THREE.CylinderGeometry(R, R, 0.06, 40), mat(0xf6d36b, { emissive: 0x6b4f00, emissiveIntensity: 0.4 }));
  pad.position.set(sx, 0.03, -sy); scene.add(pad);
  const drop = new THREE.Mesh(new THREE.SphereGeometry(R * 0.55, 24, 16), mat(0xfff1c2, { transparent: true, opacity: 0.85, roughness: 0.15 }));
  drop.scale.y = 0.45; drop.position.set(sx, 0.05, -sy); scene.add(drop);
  const sign = makeLabel("SUGAR"); sign.position.set(sx, 3.6, -sy); scene.add(sign);
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 3), mat(0x888888)); pole.position.set(sx, 1.5, -sy); scene.add(pole);
  // cones along the start
  for (let k = 0; k < 8; k++) for (const s of [-1, 1]) {
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.25, 0.7, 12), mat(0xff7a1a)); cone.position.set(k * 4 - 6, 0.35, 4 * s); scene.add(cone);
  }
}

function buildCourse(scene, data) {
  const wallMat = mat(0xb9b4a6, { roughness: 0.8 }), capMat = mat(0xd95926);
  for (const [x1, y1, x2, y2] of data.walls) {
    const L = Math.hypot(x2 - x1, y2 - y1); if (L < 1e-3) continue;
    const m = new THREE.Mesh(new THREE.BoxGeometry(L + 0.12, 2.2, 0.35), wallMat);
    m.position.set((x1 + x2) / 2, 1.1, -(y1 + y2) / 2); m.rotation.y = Math.atan2(y2 - y1, x2 - x1); scene.add(m);
    const cap = new THREE.Mesh(new THREE.BoxGeometry(L + 0.12, 0.18, 0.4), capMat);
    cap.position.set((x1 + x2) / 2, 2.25, -(y1 + y2) / 2); cap.rotation.y = m.rotation.y; scene.add(cap);
  }
  const [[fx1, fy1], [fx2, fy2]] = data.finish;
  const fl = Math.hypot(fx2 - fx1, fy2 - fy1), fa = Math.atan2(fy2 - fy1, fx2 - fx1);
  const c = document.createElement("canvas"); c.width = 256; c.height = 32; const g = c.getContext("2d");
  for (let i = 0; i < 16; i++) for (let j = 0; j < 2; j++) { g.fillStyle = (i + j) % 2 ? "#111" : "#eee"; g.fillRect(i * 16, j * 16, 16, 16); }
  const line = new THREE.Mesh(new THREE.PlaneGeometry(fl, 1.2), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c) }));
  line.rotation.x = -Math.PI / 2; line.rotation.z = fa; line.position.set((fx1 + fx2) / 2, 0.03, -(fy1 + fy2) / 2); scene.add(line);
  const sign = makeLabel("FINISH"); sign.position.set((fx1 + fx2) / 2, 4.2, -(fy1 + fy2) / 2); sign.rotation.y = fa; scene.add(sign);
  for (const [px, py] of data.finish) { const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 4.2), mat(0x888888)); pole.position.set(px, 2.1, -py); scene.add(pole); }
  if (data.crash) {
    const [, cx, cy] = data.crash;
    const star = new THREE.Mesh(new THREE.SphereGeometry(1.4, 16, 12), new THREE.MeshBasicMaterial({ color: 0xe66767, transparent: true, opacity: 0 }));
    star.position.set(cx, 1.2, -cy); scene.add(star); data._crashMesh = star;
  }
}

function makeRays(scene, data) {
  const angs = data.ray_angles_deg.map(a => a * Math.PI / 180);
  const all = [...angs.map(a => [a, 0]), ...angs.map(a => [-a, 1])];
  const geom = new THREE.BufferGeometry(); geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(all.length * 6), 3));
  geom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(all.length * 6), 3));
  const lines = new THREE.LineSegments(geom, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85 }));
  scene.add(lines);
  return { all, geom, walls: data.walls, range: data.ray_range };
}

function castRay(ox, oy, a, walls, rmax) {
  const dx = Math.cos(a), dy = Math.sin(a); let best = rmax;
  for (const [x1, y1, x2, y2] of walls) {
    const ex = x2 - x1, ey = y2 - y1, den = dx * ey - dy * ex; if (Math.abs(den) < 1e-9) continue;
    const t = ((x1 - ox) * ey - (y1 - oy) * ex) / den, u = ((x1 - ox) * dy - (y1 - oy) * dx) / den;
    if (t > 0 && u >= 0 && u <= 1 && t < best) best = t;
  }
  return best;
}

function updateRays(x, y, th, eyeL, eyeR) {
  const R = state.rays; if (!R) return;
  const nx = x + 1.7 * Math.cos(th), ny = y + 1.7 * Math.sin(th);
  const P = R.geom.attributes.position.array, C = R.geom.attributes.color.array;
  R.all.forEach(([a, side], k) => {
    const d = castRay(nx, ny, th + a, R.walls, R.range);
    P.set([nx, 1.0, -ny, nx + d * Math.cos(th + a), 1.0, -(ny + d * Math.sin(th + a))], k * 6);
    const hz = side ? eyeR : eyeL, u = Math.min(1, hz / 70);
    const col = side ? [0.95, 0.55 + 0.1 * (1 - u), 0.15] : [0.22, 0.53, 0.9];
    const f = 0.15 + 0.85 * u * (1 - d / R.range * 0.6);
    C.set([col[0] * f, col[1] * f, col[2] * f, col[0] * f * 0.4, col[1] * f * 0.4, col[2] * f * 0.4], k * 6);
  });
  R.geom.attributes.position.needsUpdate = true; R.geom.attributes.color.needsUpdate = true;
}

function sample(traj, t) {
  const dt = traj[1][0] - traj[0][0];
  const i = Math.min(traj.length - 2, Math.max(0, Math.floor(t / dt)));
  const a = traj[i], b = traj[i + 1], u = Math.min(1, Math.max(0, (t - a[0]) / dt));
  return a.map((v, k) => v + (b[k] - v) * u);
}

const HOLD_S = 1.5;  // pause at the end before looping
window.flylabSceneTime = () => {
  if (!state) return null;
  const T = state.dur + HOLD_S;
  return Math.min(((performance.now() - state.t0) / 1000) % T, state.dur);
};

function frame() {
  if (state && state.box.offsetParent !== null) {
    const t = window.flylabSceneTime();
    const [, x, y, th, v, st] = sample(state.traj, t);
    state.car.position.set(x, 0, -y); state.car.rotation.y = th;
    for (const w of state.wheels) { if (w.front) w.pivot.rotation.y = st; w.w.rotation.y += 0; w.w.rotation.z = 0; }
    state.fly.rotation.z = Math.sin(performance.now() / 120) * 0.03;
    const n = Math.max(2, Math.floor(t / (state.traj[1][0] - state.traj[0][0])) + 1);
    state.trailGeom.setDrawRange(0, n);
    const back = new THREE.Vector3(-Math.cos(th) * 11, 6.5, Math.sin(th) * 11);
    const goal = new THREE.Vector3(x, 0, -y).add(back);
    state.camPos.lerp(goal, 0.06); state.camera.position.copy(state.camPos);
    state.camera.lookAt(x + Math.cos(th) * 4, 1.0, -y - Math.sin(th) * 4);
    if (state.data.mode === "course") {
      const D = state.data, i = Math.min(D.eyes_hz.length - 1, Math.floor(t / (state.traj[1][0] - state.traj[0][0])));
      const [eL, eR] = D.eyes_hz[Math.max(0, i)];
      updateRays(x, y, th, eL, eR);
      const done = t >= state.dur - 1e-3;
      let status = st > 0.03 ? "steering LEFT" : st < -0.03 ? "steering RIGHT" : "straight";
      if (D.crash && t >= D.crash[0]) status = "CRASH into the wall";
      else if (done && D.outcome === "finished") status = "FINISHED, no wall touched";
      if (D._crashMesh) D._crashMesh.material.opacity = (D.crash && t >= D.crash[0]) ? 0.35 + 0.25 * Math.sin(performance.now() / 90) : 0;
      if (state.hud) state.hud.innerHTML = `<b>${status}</b> · ${(v * 3.6).toFixed(0)} km/h · t = ${t.toFixed(2)} s`
        + `<br><span style="color:#5a9be8">left eye → LPLC1_L ${eL.toFixed(0)} Hz</span> · <span style="color:#f08c26">right eye → LPLC1_R ${eR.toFixed(0)} Hz</span>`
        + `<br><span class="muted">closed loop, no planner: walls seen by each eye drive LPLC1; the brain's DNa neurons steer (designed eye + car link)</span>`;
      state.renderer.render(state.scene, state.camera);
      requestAnimationFrame(frame); return;
    }
    let man = "", acc = 0;
    for (const m of state.maneuvers) { if (t >= acc && t < acc + m.seconds) man = m.action; acc += m.seconds; }
    if (t >= acc - 1e-6) man = "done";
    if (state.hud) state.hud.innerHTML = `<b>${man.replace("_", " ").toUpperCase()}</b> · ${(v * 3.6).toFixed(0)} km/h · steering ${(st * 57.3).toFixed(0)}° · t = ${t.toFixed(2)} s`
      + `<br><span class="muted">car driven by the brain's P9 / MDN / DNa firing (hand-designed car link)</span>`;
    state.renderer.render(state.scene, state.camera);
  }
  requestAnimationFrame(frame);
}

function resize() {
  if (!state) return;
  const w = state.box.clientWidth, h = state.box.clientHeight; if (!w || !h) return;
  state.renderer.setSize(w, h, false); state.renderer.domElement.style.width = w + "px"; state.renderer.domElement.style.height = h + "px";
  state.camera.aspect = w / h; state.camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

window.flylabShowCar = async (beh) => {
  if (state && state.runId === beh.run_id) return;
  const data = await (await fetch(`/${beh.car}?t=${beh._mtime}`)).json();
  data.runId = beh.run_id;
  build(document.getElementById("carbox"), data);
};
window.flylabHideCar = () => { if (state) { state.renderer.dispose(); state = null; } };
requestAnimationFrame(frame);
