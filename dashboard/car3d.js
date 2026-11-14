// FlyLab vehicle scenes: replays drive_car / run_course / fly_through / run_maze / run_olympics / ride_bike runs
// (trajectories computed from the whole-brain simulation) at 1x speed (bike: synced to the leg video).
// Exposes window.flylabSceneTime() so the readout chart and the 3D brain stay in sync.
import * as THREE from "three";

let state = null;

function mat(c, extra = {}) { return new THREE.MeshStandardMaterial({ color: c, roughness: 0.55, metalness: 0.1, ...extra }); }

const THEMES = {
  canyon: { bg: 0x20201e, fog: [40, 110], ground: 0x3b3d40, grid: true, hemi: 1.6, sun: 1.6 },
  cave: { bg: 0x0b0a09, fog: [10, 48], ground: 0x2e2722, grid: false, hemi: 0.55, sun: 0.35 },
  forest: { bg: 0x1a2a1f, fog: [18, 75], ground: 0x2c4424, grid: false, hemi: 1.2, sun: 1.3 },
  garden: { bg: 0x22301f, fog: [30, 95], ground: 0x3f6a2e, grid: false, hemi: 1.4, sun: 1.5 },
};

// ------------------------------------------------------------------ models
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
    for (const k of [0, 1]) {
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.55), mat(0x8a5a22));
      leg.rotation.z = Math.PI / 2.4; leg.position.set(0.55, 0.0 - 0.05 * k, 0.15 * s); g.add(leg);
    }
  }
  return g;
}

function makeFlyAvatar(scale = 3.0) {
  // stylised fly, x = forward; returns {group, wings, legs}
  const g = new THREE.Group(), wings = [], legs = [];
  const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.3, 20, 14), mat(0xb8782a)); thorax.scale.set(1.2, 1, 1); g.add(thorax);
  const abd = new THREE.Mesh(new THREE.SphereGeometry(0.3, 20, 14), mat(0xd9a441)); abd.scale.set(1.6, 0.85, 0.95); abd.position.set(-0.55, -0.03, 0); g.add(abd);
  for (let k = 0; k < 3; k++) { const band = new THREE.Mesh(new THREE.TorusGeometry(0.255 - 0.03 * k, 0.03, 6, 20), mat(0x4a3216)); band.rotation.y = Math.PI / 2; band.position.set(-0.45 - 0.2 * k, -0.03, 0); g.add(band); }
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.22, 20, 14), mat(0xc9822b)); head.position.set(0.42, 0.06, 0); g.add(head);
  for (const s of [-1, 1]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.13, 16, 12), mat(0xc0201a, { roughness: 0.25 })); eye.position.set(0.47, 0.12, 0.14 * s); g.add(eye);
    const pivot = new THREE.Group(); pivot.position.set(-0.05, 0.22, 0.1 * s);
    const wing = new THREE.Mesh(new THREE.PlaneGeometry(1.0, 0.36), mat(0xe8f4ff, { transparent: true, opacity: 0.5, side: THREE.DoubleSide }));
    wing.rotation.x = -Math.PI / 2; wing.position.set(-0.35, 0, 0.18 * s); wing.rotation.z = 0.25 * s; pivot.add(wing); g.add(pivot); wings.push({ pivot, s });
    for (const k of [-1, 0, 1]) {
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.018, 0.62), mat(0x3d2a12));
      const lp = new THREE.Group(); lp.position.set(0.18 * k, -0.12, 0.14 * s); leg.position.set(0, -0.22, 0.12 * s); leg.rotation.x = 0.55 * s; lp.add(leg); g.add(lp); legs.push({ lp, k, s });
    }
  }
  g.scale.setScalar(scale);
  return { group: g, wings, legs };
}

function makeCar(color = 0xd95926) {
  const car = new THREE.Group();
  const shell = new THREE.Mesh(new THREE.BoxGeometry(4.2, 0.7, 1.9), mat(color)); shell.position.y = 0.75; car.add(shell);
  const hood = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.25, 1.8), mat(new THREE.Color(color).offsetHSL(0, 0, 0.08))); hood.position.set(1.35, 1.2, 0); car.add(hood);
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

function makeBike() {
  const g = new THREE.Group(), frameM = mat(0x3987e5, { metalness: 0.4 }), dark = mat(0x1c1c1c);
  const wheel = (x) => {
    const p = new THREE.Group(); p.position.set(x, 0.36, 0);
    const tyre = new THREE.Mesh(new THREE.TorusGeometry(0.34, 0.035, 10, 32), dark); p.add(tyre);
    for (let k = 0; k < 6; k++) { const sp = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.66), mat(0xaaaaaa)); sp.rotation.z = k * Math.PI / 6; p.add(sp); }
    g.add(p); return p;
  };
  const rear = wheel(-0.52), frontPivot = new THREE.Group(); frontPivot.position.set(0.52, 0, 0); g.add(frontPivot);
  const front = new THREE.Group(); front.position.set(0, 0.36, 0); frontPivot.add(front);
  const tyre = new THREE.Mesh(new THREE.TorusGeometry(0.34, 0.035, 10, 32), dark); front.add(tyre);
  for (let k = 0; k < 6; k++) { const sp = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.66), mat(0xaaaaaa)); sp.rotation.z = k * Math.PI / 6; front.add(sp); }
  const tube = (a, b, m = frameM, r = 0.025) => {
    const A = new THREE.Vector3(...a), B = new THREE.Vector3(...b), L = A.distanceTo(B);
    const c = new THREE.Mesh(new THREE.CylinderGeometry(r, r, L, 8), m); c.position.copy(A).add(B).multiplyScalar(0.5);
    c.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), B.clone().sub(A).normalize()); return c;
  };
  g.add(tube([-0.52, 0.36, 0], [-0.05, 0.4, 0])); g.add(tube([-0.05, 0.4, 0], [-0.15, 0.85, 0])); g.add(tube([-0.15, 0.85, 0], [0.42, 0.85, 0]));
  g.add(tube([-0.05, 0.4, 0], [0.42, 0.85, 0])); g.add(tube([-0.52, 0.36, 0], [-0.15, 0.85, 0]));
  frontPivot.add(tube([0, 0.36, 0], [-0.1, 0.95, 0])); frontPivot.add(tube([-0.1, 0.95, -0.28], [-0.1, 0.95, 0.28], dark, 0.02));
  const seat = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.05, 0.12), dark); seat.position.set(-0.17, 0.9, 0); g.add(seat);
  const crank = new THREE.Group(); crank.position.set(-0.05, 0.4, 0); g.add(crank);
  for (const s of [-1, 1]) { const arm = tube([0, 0, 0.06 * s], [0, -0.16 * s, 0.06 * s], dark, 0.015); crank.add(arm);
    const pedal = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.02, 0.06), dark); pedal.position.set(0, -0.16 * s, 0.1 * s); crank.add(pedal); }
  const rider = makeFlyAvatar(0.55); rider.group.position.set(-0.08, 1.05, 0); rider.group.rotation.z = 0.25; g.add(rider.group);
  return { bike: g, rear, front, frontPivot, crank, rider };
}

function makeBat() {
  const g = new THREE.Group(), m = mat(0x1a1518, { roughness: 0.9 });
  const body = new THREE.Mesh(new THREE.SphereGeometry(0.35, 14, 10), m); body.scale.set(1.4, 0.8, 0.8); g.add(body);
  for (const s of [-1, 1]) {
    const ear = new THREE.Mesh(new THREE.ConeGeometry(0.08, 0.25, 6), m); ear.position.set(0.35, 0.3, 0.12 * s); g.add(ear);
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.05, 8, 6), new THREE.MeshBasicMaterial({ color: 0xff4030 })); eye.position.set(0.45, 0.12, 0.12 * s); g.add(eye);
  }
  const wings = [];
  for (const s of [-1, 1]) {
    const shape = new THREE.Shape(); shape.moveTo(0, 0); shape.lineTo(0.6, 1.6 * s); shape.lineTo(0.1, 1.2 * s); shape.lineTo(-0.1, 1.7 * s); shape.lineTo(-0.4, 1.0 * s); shape.lineTo(-0.5, 0.3 * s); shape.lineTo(0, 0);
    const w = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshStandardMaterial({ color: 0x2a2226, side: THREE.DoubleSide, roughness: 0.9 }));
    w.rotation.x = -Math.PI / 2; const p = new THREE.Group(); p.add(w); g.add(p); wings.push({ p, s });
  }
  g.scale.setScalar(1.3);
  return { group: g, wings };
}

function makeLabel(text, color = "#1a1a19", bg = "#fab219", w = 4) {
  const c = document.createElement("canvas"); c.width = 512; c.height = 160;
  const g = c.getContext("2d"); g.fillStyle = bg; g.fillRect(0, 0, 512, 160);
  g.fillStyle = color; g.font = "bold 64px Segoe UI, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(text, 256, 84);
  return new THREE.Mesh(new THREE.PlaneGeometry(w, w * 0.3125), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), side: THREE.DoubleSide }));
}

function makeSprite(text, bg) {
  const c = document.createElement("canvas"); c.width = 512; c.height = 96; const g = c.getContext("2d");
  g.fillStyle = bg; g.globalAlpha = 0.9; g.fillRect(0, 0, 512, 96); g.globalAlpha = 1;
  g.fillStyle = "#fff"; g.font = "bold 40px Segoe UI, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(text.slice(0, 26), 256, 50);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false })); s.scale.set(5.3, 1.0, 1); return s;
}

// ------------------------------------------------------------------ scene building
function build(box, data) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const theme = THEMES[data.theme] || THEMES.canyon;
  renderer.setClearColor(theme.bg);
  box.innerHTML = ""; box.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(theme.bg, theme.fog[0], theme.fog[1]);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x333333, theme.hemi));
  const sun = new THREE.DirectionalLight(0xffffff, theme.sun); sun.position.set(10, 25, 8); scene.add(sun);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(600, 600), mat(theme.ground, { roughness: 0.95 }));
  ground.rotation.x = -Math.PI / 2; scene.add(ground);
  if (theme.grid) { const grid = new THREE.GridHelper(400, 100, 0x55585c, 0x47494d); grid.position.y = 0.01; scene.add(grid); }
  const lamp = data.theme === "cave" ? new THREE.PointLight(0xffc98a, 60, 30, 1.6) : null; if (lamp) scene.add(lamp);
  const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 400);
  state = { renderer, scene, camera, box, data, runId: data.runId, t0: performance.now(), lamp,
            camPos: new THREE.Vector3(-8, 6, 0), hud: document.getElementById("carhud"), maneuvers: data.maneuvers || [] };
  if (data.mode === "race") return buildRace(scene, data);
  if (data.mode === "bike") return buildBike(scene, data);
  if (data.mode === "course") buildCourse(scene, data); else buildStation(scene, data);
  const traj = data.trajectory;
  const pts = traj.map(r => new THREE.Vector3(r[1], 0.05, -r[2]));
  if (data.mode !== "course") {
    const ghost = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineDashedMaterial({ color: 0x8a897f, dashSize: 0.6, gapSize: 0.4 }));
    ghost.computeLineDistances(); scene.add(ghost);
  }
  const trailGeom = new THREE.BufferGeometry().setFromPoints(pts);
  const trail = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ color: data.avatar === "fly" ? 0xfab219 : 0x3987e5 })); trail.position.y = 0.04; scene.add(trail);
  Object.assign(state, { traj, trailGeom, dur: traj[traj.length - 1][0] });
  if (data.avatar === "fly") {
    const av = makeFlyAvatar(data.theme === "garden" ? 2.4 : 3.0); scene.add(av.group);
    state.avatar = av; state.flying = data.theme !== "garden";
  } else {
    const { car, wheels, fly } = makeCar(); scene.add(car); Object.assign(state, { car, wheels, fly });
  }
  state.rays = data.mode === "course" ? makeRays(scene, data) : null;
  if (data.bat) { const b = makeBat(); b.group.visible = false; scene.add(b.group); state.bat = b; }
  resize();
}

function buildStation(scene, data) {
  const [sx, sy] = data.station, R = data.station_radius;
  const pad = new THREE.Mesh(new THREE.CylinderGeometry(R, R, 0.06, 40), mat(0xf6d36b, { emissive: 0x6b4f00, emissiveIntensity: 0.4 }));
  pad.position.set(sx, 0.03, -sy); scene.add(pad);
  const drop = new THREE.Mesh(new THREE.SphereGeometry(R * 0.55, 24, 16), mat(0xfff1c2, { transparent: true, opacity: 0.85, roughness: 0.15 }));
  drop.scale.y = 0.45; drop.position.set(sx, 0.05, -sy); scene.add(drop);
  const sign = makeLabel("SUGAR"); sign.position.set(sx, 3.6, -sy); scene.add(sign);
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 3), mat(0x888888)); pole.position.set(sx, 1.5, -sy); scene.add(pole);
  for (let k = 0; k < 8; k++) for (const s of [-1, 1]) {
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.25, 0.7, 12), mat(0xff7a1a)); cone.position.set(k * 4 - 6, 0.35, 4 * s); scene.add(cone);
  }
}

function rng(seed) { let s = seed; return () => (s = (s * 16807) % 2147483647) / 2147483647; }

function buildWalls(scene, walls, theme) {
  const R = rng(7);
  const wallMat = { canyon: mat(0xb9b4a6, { roughness: 0.8 }), cave: mat(0x4a3f36, { roughness: 1 }), forest: mat(0x23361d, { roughness: 1 }), garden: mat(0x2f5e25, { roughness: 1 }) }[theme] || mat(0xb9b4a6);
  const capMat = { canyon: mat(0xd95926), cave: mat(0x5a4c40, { roughness: 1 }), forest: mat(0x2d4a24, { roughness: 1 }), garden: mat(0x4c8a35, { roughness: 1 }) }[theme] || mat(0xd95926);
  const H = { canyon: 2.2, cave: 6.5, forest: 1.6, garden: 2.6 }[theme] || 2.2;
  for (const [x1, y1, x2, y2] of walls) {
    const L = Math.hypot(x2 - x1, y2 - y1); if (L < 1e-3) continue;
    const ang = Math.atan2(y2 - y1, x2 - x1), cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
    const m = new THREE.Mesh(new THREE.BoxGeometry(L + 0.12, H, theme === "cave" ? 1.2 : 0.5), wallMat);
    m.position.set(cx, H / 2, -cy); m.rotation.y = ang; scene.add(m);
    const cap = new THREE.Mesh(new THREE.BoxGeometry(L + 0.12, 0.18, theme === "cave" ? 1.3 : 0.6), capMat);
    cap.position.set(cx, H + 0.05, -cy); cap.rotation.y = ang; scene.add(cap);
    if (theme === "forest" && R() < 0.5) {   // trees behind the undergrowth
      const tx = cx + Math.sin(ang) * (1.5 + 2 * R()) * (R() < 0.5 ? 1 : -1), ty = cy;
      const tr = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.4, 7, 8), mat(0x5a3d22)); tr.position.set(tx, 3.5, -ty); scene.add(tr);
      const crown = new THREE.Mesh(new THREE.ConeGeometry(2.2, 5, 9), mat(0x1f4d22)); crown.position.set(tx, 8, -ty); scene.add(crown);
    }
    if (theme === "cave" && R() < 0.45) {   // stalactites hanging off the walls
      const st = new THREE.Mesh(new THREE.ConeGeometry(0.35 + 0.3 * R(), 1.5 + 2.5 * R(), 7), mat(0x6b5d50, { roughness: 1 }));
      st.rotation.x = Math.PI; st.position.set(cx, H - 0.8, -cy); scene.add(st);
    }
    if (theme === "garden" && R() < 0.25) {
      const fl = new THREE.Mesh(new THREE.SphereGeometry(0.18, 8, 6), mat([0xf2c4de, 0xfff3a8, 0xf08c8c][Math.floor(R() * 3)])); fl.position.set(cx, H + 0.2, -cy); scene.add(fl);
    }
  }
}

function buildCourse(scene, data) {
  const theme = data.theme || "canyon";
  buildWalls(scene, data.walls, theme);
  for (const [ox, oy, r] of data.obstacles || []) {
    if (theme === "forest") {
      const tr = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.9, r, 8, 12), mat(0x6b4a2b)); tr.position.set(ox, 4, -oy); scene.add(tr);
      const crown = new THREE.Mesh(new THREE.SphereGeometry(r * 3.2, 12, 10), mat(0x2a6b2c)); crown.position.set(ox, 9, -oy); scene.add(crown);
    } else if (theme === "cave") {
      const sg = new THREE.Mesh(new THREE.ConeGeometry(r * 1.15, 6, 9), mat(0x7a6a58, { roughness: 1 })); sg.position.set(ox, 3, -oy); scene.add(sg);
      const sc = new THREE.Mesh(new THREE.ConeGeometry(r * 0.9, 3.5, 9), mat(0x7a6a58, { roughness: 1 })); sc.rotation.x = Math.PI; sc.position.set(ox, 7.6, -oy); scene.add(sc);
    } else {
      const p = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 3, 14), mat(0xb9b4a6)); p.position.set(ox, 1.5, -oy); scene.add(p);
    }
  }
  data._pellets = (data.pellets || []).map((p, j) => {
    const g = new THREE.Group(), poison = p.kind === "poison", R = data.pellet_radius * 0.8;
    const puddle = new THREE.Mesh(new THREE.CylinderGeometry(R, R, 0.05, 32), mat(poison ? 0x8b5cf6 : 0xf6d36b, { emissive: poison ? 0x2e1065 : 0x6b4f00, emissiveIntensity: 0.4 }));
    const dome = new THREE.Mesh(new THREE.SphereGeometry(R * 0.55, 20, 12), mat(poison ? 0xc4b5fd : 0xfff1c2, { transparent: true, opacity: 0.85, roughness: 0.15 }));
    dome.scale.y = 0.45; g.add(puddle); g.add(dome);
    const sign = makeLabel(poison ? "BITTER!" : "SUGAR", "#1a1a19", poison ? "#c4b5fd" : "#fab219", 2.6); sign.position.set(0, 3.0, 0); g.add(sign);
    g.position.set(p.xy[0], 0.03, -p.xy[1]); scene.add(g);
    const ev = (name) => (data.events || []).find(e => e[1] === name && e[2] === j);
    return { g, sign, eatingT: ev("eating") ? ev("eating")[0] : null, eatenT: ev("eaten") ? ev("eaten")[0] : null };
  });
  const [[fx1, fy1], [fx2, fy2]] = data.finish;
  const fl = Math.hypot(fx2 - fx1, fy2 - fy1), fa = Math.atan2(fy2 - fy1, fx2 - fx1);
  const c = document.createElement("canvas"); c.width = 256; c.height = 32; const g = c.getContext("2d");
  for (let i = 0; i < 16; i++) for (let j = 0; j < 2; j++) { g.fillStyle = (i + j) % 2 ? "#111" : "#eee"; g.fillRect(i * 16, j * 16, 16, 16); }
  const line = new THREE.Mesh(new THREE.PlaneGeometry(fl, 1.2), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c) }));
  line.rotation.x = -Math.PI / 2; line.rotation.z = fa; line.position.set((fx1 + fx2) / 2, 0.03, -(fy1 + fy2) / 2); scene.add(line);
  const sign = makeLabel("FINISH"); sign.position.set((fx1 + fx2) / 2, theme === "cave" ? 5.5 : 4.2, -(fy1 + fy2) / 2); sign.rotation.y = fa; scene.add(sign);
  for (const [px, py] of data.finish) { const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 4.2), mat(0x888888)); pole.position.set(px, 2.1, -py); scene.add(pole); }
  if (data.crash) {
    const [, cx, cy] = data.crash;
    const star = new THREE.Mesh(new THREE.SphereGeometry(1.4, 16, 12), new THREE.MeshBasicMaterial({ color: 0xe66767, transparent: true, opacity: 0 }));
    star.position.set(cx, 1.2, -cy); scene.add(star); data._crashMesh = star;
  }
}

function buildRace(scene, data) {
  buildCourse(scene, { ...data, theme: "canyon", obstacles: [], pellets: [], crash: null });
  const racers = data.racers.map(r => {
    const { car, wheels } = makeCar(new THREE.Color(r.color).getHex()); scene.add(car);
    const tag = makeSprite(`#${r.rank} ${r.name}`, r.color); tag.position.set(0, 3.4, 0); car.add(tag);
    const pts = r.trajectory.map(q => new THREE.Vector3(q[1], 0.06, -q[2]));
    const tg = new THREE.BufferGeometry().setFromPoints(pts); const tl = new THREE.Line(tg, new THREE.LineBasicMaterial({ color: new THREE.Color(r.color) })); scene.add(tl);
    let crash = null;
    if (r.crash) { crash = new THREE.Mesh(new THREE.SphereGeometry(1.3, 14, 10), new THREE.MeshBasicMaterial({ color: 0xe66767, transparent: true, opacity: 0 })); crash.position.set(r.crash[1], 1.2, -r.crash[2]); scene.add(crash); }
    const cum = [0]; for (let k = 1; k < r.trajectory.length; k++) cum.push(cum[k - 1] + Math.hypot(r.trajectory[k][1] - r.trajectory[k - 1][1], r.trajectory[k][2] - r.trajectory[k - 1][2]));
    return { r, car, wheels, tg, crash, cum, dur: r.trajectory[r.trajectory.length - 1][0] };
  });
  Object.assign(state, { racers, dur: Math.max(...racers.map(x => x.dur)) });
  resize();
}

function buildBike(scene, data) {
  // park: path cones + trees around
  const R = rng(3);
  for (let k = 0; k < 60; k++) {
    const a = R() * Math.PI * 2, d = 25 + R() * 60, x = Math.cos(a) * d, z = Math.sin(a) * d;
    const tr = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.35, 4, 8), mat(0x5a3d22)); tr.position.set(x, 2, z); scene.add(tr);
    const cr = new THREE.Mesh(new THREE.SphereGeometry(1.8 + R(), 10, 8), mat(0x2f6b2c)); cr.position.set(x, 4.8, z); scene.add(cr);
  }
  for (let k = -3; k < 12; k++) for (const s of [-1, 1]) { const cone = new THREE.Mesh(new THREE.ConeGeometry(0.18, 0.5, 10), mat(0xff7a1a)); cone.position.set(k * 3, 0.25, 2.5 * s); scene.add(cone); }
  const traj = data.trajectory, pts = traj.map(r => new THREE.Vector3(r[1], 0.04, -r[2]));
  const trailGeom = new THREE.BufferGeometry().setFromPoints(pts); scene.add(new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ color: 0x3987e5 })));
  const b = makeBike(); scene.add(b.bike);
  const vid = document.createElement("video"); vid.src = `/${data.video}`; vid.muted = true; vid.loop = true; vid.autoplay = true; vid.playsInline = true;
  vid.style.cssText = "position:absolute;right:12px;bottom:12px;width:34%;border:2px solid #3987e5;border-radius:6px;background:#000";
  const cap = document.createElement("div"); cap.style.cssText = "position:absolute;right:14px;bottom:calc(12px + 34% * 0.75 + 6px);font-size:12px;color:#c3c2b7;";
  cap.textContent = "real legs (flygym physics) → ball read-out → bike";
  state.box.appendChild(vid); state.box.appendChild(cap); vid.play().catch(() => {});
  Object.assign(state, { traj, trailGeom, dur: traj[traj.length - 1][0], bike: b, vid, playSpeed: data.video_play_speed || 0.25, dist: 0, lastT: 0 });
  resize();
}

// ------------------------------------------------------------------ rays (course)
function makeRays(scene, data) {
  const angs = data.ray_angles_deg.map(a => a * Math.PI / 180);
  const all = [...angs.map(a => [a, 0]), ...angs.map(a => [-a, 1])];
  const geom = new THREE.BufferGeometry(); geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(all.length * 6), 3));
  geom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(all.length * 6), 3));
  const lines = new THREE.LineSegments(geom, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85 }));
  scene.add(lines);
  const walls = data.walls.slice();
  for (const [ox, oy, r] of data.obstacles || []) for (let k = 0; k < 10; k++) {
    const a1 = k / 10 * 2 * Math.PI, a2 = (k + 1) / 10 * 2 * Math.PI;
    walls.push([ox + r * Math.cos(a1), oy + r * Math.sin(a1), ox + r * Math.cos(a2), oy + r * Math.sin(a2)]);
  }
  return { all, geom, walls, range: data.ray_range };
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

function updateRays(x, y, th, eyeL, eyeR, h = 1.0) {
  const R = state.rays; if (!R) return;
  const nx = x + 1.7 * Math.cos(th), ny = y + 1.7 * Math.sin(th);
  const P = R.geom.attributes.position.array, C = R.geom.attributes.color.array;
  R.all.forEach(([a, side], k) => {
    const d = castRay(nx, ny, th + a, R.walls, R.range);
    P.set([nx, h, -ny, nx + d * Math.cos(th + a), h, -(ny + d * Math.sin(th + a))], k * 6);
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

const HOLD_S = 1.5;
window.flylabSceneTime = () => {
  if (!state) return null;
  if (state.vid && state.vid.duration) return Math.min(state.vid.currentTime * state.playSpeed, state.dur);
  const T = state.dur + HOLD_S;
  return Math.min(((performance.now() - state.t0) / 1000) % T, state.dur);
};

function follow(x, y, th, back = 11, up = 6.5, ahead = 4, lookY = 1.0, k = 0.06) {
  const goal = new THREE.Vector3(x - Math.cos(th) * back, up, -y + Math.sin(th) * back);
  state.camPos.lerp(goal, k); state.camera.position.copy(state.camPos);
  state.camera.lookAt(x + Math.cos(th) * ahead, lookY, -y - Math.sin(th) * ahead);
}

// ------------------------------------------------------------------ per-frame
function frameRace(t) {
  let lead = null, best = -1;
  const rows = [];
  for (const R of state.racers) {
    const tt = Math.min(t, R.dur), [, x, y, th, v, st] = sample(R.r.trajectory, tt);
    R.car.position.set(x, 0, -y); R.car.rotation.y = th; for (const w of R.wheels) if (w.front) w.pivot.rotation.y = st;
    R.tg.setDrawRange(0, Math.max(2, Math.floor(tt / (R.r.trajectory[1][0] - R.r.trajectory[0][0])) + 1));
    const crashed = R.r.crash && t >= R.r.crash[0], done = t >= R.dur - 1e-3;
    if (R.crash) R.crash.material.opacity = crashed ? 0.3 + 0.2 * Math.sin(performance.now() / 90) : 0;
    const prog = R.cum[Math.min(R.cum.length - 1, Math.floor(tt / (R.r.trajectory[1][0] - R.r.trajectory[0][0])))];
    if (!crashed && prog > best) { best = prog; lead = [x, y, th]; }
    rows.push(`<span style="color:${R.r.color}">■</span> #${R.r.rank} ${R.r.name}: ${crashed ? "<b>CRASH</b>" : done && R.r.outcome === "finished" ? "<b>FINISHED " + R.r.time_s.toFixed(1) + " s</b>" : (v * 3.6).toFixed(0) + " km/h"}`);
  }
  if (!lead) { const R = state.racers[0]; const s = sample(R.r.trajectory, Math.min(t, R.dur)); lead = [s[1], s[2], s[3]]; }
  follow(lead[0], lead[1], lead[2], 14, 9, 4);
  if (state.hud) state.hud.innerHTML = `<b>CONNECTOME OLYMPICS</b> · t = ${t.toFixed(1)} s<br>` + rows.join("<br>")
    + `<br><span class="muted">same connectome, different lesions, closed loop (eyes → LPLC1 → brain → DNa steering)</span>`;
}

function frameBike(t) {
  const B = state.bike, [, x, y, th, v, st] = sample(state.traj, t);
  B.bike.position.set(x, 0, -y); B.bike.rotation.y = th; B.frontPivot.rotation.y = st;
  const dt = t - state.lastT; state.lastT = t; if (dt > 0 && dt < 0.5) state.dist += v * dt;
  B.rear.rotation.z = -state.dist / 0.36; B.front.rotation.z = -state.dist / 0.36; B.crank.rotation.z = -state.dist / 0.9;
  B.bike.rotation.x = -st * 0.4;
  state.trailGeom.setDrawRange(0, Math.max(2, Math.floor(t / (state.traj[1][0] - state.traj[0][0])) + 1));
  follow(x, y, th, 5, 2.6, 2, 0.7, 0.08);
  let man = "", acc = 0;
  for (const m of state.maneuvers) { if (t >= acc && t < acc + m.seconds) man = m.action; acc += m.seconds; }
  const ball = state.data.ball, i = Math.min(ball.length - 1, Math.max(0, Math.floor(t / 0.02)));
  if (state.hud) state.hud.innerHTML = `<b>${(man || "done").replace("_", " ").toUpperCase()}</b> · bike ${(v * 3.6).toFixed(0)} km/h · t = ${t.toFixed(2)} s`
    + `<br>ball read-out: fly walks <b>${ball[i][1].toFixed(1)} mm/s</b>, turns <b>${ball[i][2] >= 0 ? "+" : ""}${ball[i][2].toFixed(0)}°/s</b>`
    + `<br><span class="muted">brain → legs (physics) → ball → bike (designed gear ${state.data.gear} m/s per mm/s)</span>`;
}

function frameCourse(t, x, y, th, v, st) {
  const D = state.data, dtj = state.traj[1][0] - state.traj[0][0];
  const i = Math.max(0, Math.min(D.eyes_hz.length - 1, Math.floor(t / dtj)));
  const [eL, eR] = D.eyes_hz[i];
  let alt = 0, roll = 0;
  const bat = D.bat;
  if (state.avatar) {
    const A = state.avatar, now = performance.now();
    alt = state.flying ? 2.4 : 0.75;
    if (bat && bat.t_dodge && t >= bat.t_dodge && t < bat.t_dodge + 0.7) { const u = (t - bat.t_dodge) / 0.7; alt += 3.2 * Math.sin(Math.PI * u); roll = 2 * Math.PI * u; }
    A.group.position.set(x, alt, -y); A.group.rotation.set(roll, th, state.flying ? 0.08 - st * 0.6 : 0, "YXZ");
    const flap = state.flying ? Math.sin(now / 18) * 0.7 : 0.05;
    for (const w of A.wings) w.pivot.rotation.x = flap * w.s;
    for (const l of A.legs) l.lp.rotation.z = state.flying ? -0.5 : Math.sin(now / 60 * (v > 0.3 ? 1 : 0) + l.k * 2 + (l.s > 0 ? Math.PI : 0)) * 0.35;
    if (state.lamp) state.lamp.position.set(x + Math.cos(th) * 3, alt + 2.5, -y - Math.sin(th) * 3);
  } else {
    state.car.position.set(x, 0, -y); state.car.rotation.y = th;
    for (const w of state.wheels) if (w.front) w.pivot.rotation.y = st;
  }
  updateRays(x, y, th, eL, eR, state.avatar ? alt : 1.0);
  if (state.bat && bat) {
    const tr = bat.track, B = state.bat;
    if (tr.length && t >= tr[0][0] && t <= tr[tr.length - 1][0] + (bat.state === "caught" ? 99 : 0)) {
      const k = Math.min(tr.length - 1, Math.max(0, Math.round((t - tr[0][0]) / 0.02)));
      const caught = bat.state === "caught" && t >= bat.t_caught;
      B.group.visible = true;
      B.group.position.set(caught ? x : tr[k][1], (state.flying ? 2.6 : 1.5) + (caught ? 0.4 : 0), caught ? -y : -tr[k][2]);
      B.group.rotation.y = th + Math.PI; const f = Math.sin(performance.now() / 45) * 0.8; for (const w of B.wings) w.p.rotation.x = f * w.s;
    } else B.group.visible = false;
  }
  for (const P of D._pellets || []) {
    if (P.eatenT != null && t >= P.eatenT) P.g.visible = false;
    else { P.g.visible = true; const sc = (P.eatingT != null && t >= P.eatingT) ? Math.max(0.15, 1 - (t - P.eatingT) / 0.6) : 1; P.g.scale.set(sc, 1, sc); }
    P.sign.lookAt(state.camera.position);
  }
  const lastEv = (D.events || []).filter(e => e[0] <= t).slice(-1)[0];
  const done = t >= state.dur - 1e-3;
  let status = st > 0.03 ? "steering LEFT" : st < -0.03 ? "steering RIGHT" : "straight";
  if (D.crash && t >= D.crash[0]) status = "CRASH";
  else if (bat && bat.state === "caught" && t >= bat.t_caught) status = "CAUGHT by the bat (giant fiber silent)";
  else if (bat && bat.t_dodge && t >= bat.t_dodge && t < bat.t_dodge + 1.2) status = `DODGE! giant fiber ${bat.gf_hz.toFixed(0)} Hz (scripted roll)`;
  else if (bat && t >= bat.t0 && !bat.t_dodge && bat.state !== "caught" && t < bat.t0 + 3) status = "BAT incoming";
  else if (lastEv && lastEv[1] === "eating" && t < lastEv[0] + 0.7) status = `EATING ${lastEv[3] === "poison" ? "POISON" : "sugar"} (MN9 above 30 Hz)`;
  else if (lastEv && lastEv[1] === "passed" && t < lastEv[0] + 1.2) status = lastEv[3] === "poison" ? "REFUSED the bitter pellet" : "walked past the sugar";
  else if (done && D.outcome === "finished") status = "FINISHED, no wall touched";
  if (D._crashMesh) D._crashMesh.material.opacity = (D.crash && t >= D.crash[0]) ? 0.35 + 0.25 * Math.sin(performance.now() / 90) : 0;
  const unit = state.avatar && !state.flying ? "m/s" : "km/h", vv = unit === "m/s" ? v.toFixed(1) : (v * 3.6).toFixed(0);
  const what = D.avatar === "fly" ? (state.flying ? "fly-through (flight not simulated: fixed-height glide)" : "walking fly") : "car";
  if (state.hud) state.hud.innerHTML = `<b>${status}</b> · ${vv} ${unit} · t = ${t.toFixed(2)} s`
    + `<br><span style="color:#5a9be8">left eye → LPLC1_L ${eL.toFixed(0)} Hz</span> · <span style="color:#f08c26">right eye → LPLC1_R ${eR.toFixed(0)} Hz</span>`
    + `<br><span class="muted">closed loop, no planner: walls seen by each eye drive LPLC1; the brain's DNa neurons steer the ${what}</span>`;
  // walking fly between hedges: higher, closer camera so the hedges never hide it in bends
  if (state.avatar) follow(x, y, th, state.flying ? 9 : 5.5, state.flying ? 5.5 : 9, 3, alt * 0.6, 0.06);
  else follow(x, y, th);
}

function frame() {
  if (state && state.box.offsetParent !== null) {
    const t = window.flylabSceneTime();
    if (state.racers) frameRace(t);
    else if (state.bike) frameBike(t);
    else {
      const [, x, y, th, v, st] = sample(state.traj, t);
      const n = Math.max(2, Math.floor(t / (state.traj[1][0] - state.traj[0][0])) + 1);
      state.trailGeom.setDrawRange(0, n);
      if (state.data.mode === "course") frameCourse(t, x, y, th, v, st);
      else {
        state.car.position.set(x, 0, -y); state.car.rotation.y = th;
        for (const w of state.wheels) if (w.front) w.pivot.rotation.y = st;
        state.fly.rotation.z = Math.sin(performance.now() / 120) * 0.03;
        follow(x, y, th);
        let man = "", acc = 0;
        for (const m of state.maneuvers) { if (t >= acc && t < acc + m.seconds) man = m.action; acc += m.seconds; }
        if (t >= acc - 1e-6) man = "done";
        if (state.hud) state.hud.innerHTML = `<b>${man.replace("_", " ").toUpperCase()}</b> · ${(v * 3.6).toFixed(0)} km/h · steering ${(st * 57.3).toFixed(0)}° · t = ${t.toFixed(2)} s`
          + `<br><span class="muted">car driven by the brain's P9 / MDN / DNa firing (hand-designed car link)</span>`;
      }
    }
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
  if (state) { state.renderer.dispose(); if (state.vid) state.vid.pause(); }
  build(document.getElementById("carbox"), data);
};
window.flylabHideCar = () => { if (state) { state.renderer.dispose(); if (state.vid) state.vid.pause(); state = null; } };
requestAnimationFrame(frame);
