// FlyLab piano view: keyboard mode (flies walking on a keyboard, footfalls = notes) and orchestra mode (piano roll).
// wrapped in its own scope: index.html defines globals with the same names (loop, el, ...)
(() => {
let P = null;
const FLY_COL = ["#3987e5", "#199e70", "#d95926"];
const LANE = { drums: ["#d55181", "drums · looming → giant fiber"], melody: ["#199e70", "melody · P9 + DNa steering"],
               bass: ["#fab219", "bass · sugar → MN9 (bitter mutes)"] };

function el(tag, css, html) { const e = document.createElement(tag); if (css) e.style.cssText = css; if (html) e.innerHTML = html; return e; }

function sampleTraj(tr, t) {
  if (!tr.length) return null;
  let i = 0; while (i < tr.length - 1 && tr[i + 1][0] <= t) i++;
  return tr[Math.min(i, tr.length - 1)];
}

function drawKeyboard() {
  const { cv, data, media } = P, g = cv.getContext("2d"), W = cv.width, H = cv.height;
  const t = media.currentTime / data.time_scale;
  const x0 = data.key0_x, kw = data.key_w, n = data.n_keys, L = data.key_len;
  const sx = W / (n * kw), sy = H / (L + 4);
  const X = (x) => (x - x0) * sx, Y = (y) => H / 2 - y * sy;
  g.fillStyle = "#151514"; g.fillRect(0, 0, W, H);
  const lit = new Map();
  for (const nt of data.notes) if (nt.t <= t && t - nt.t < 0.035) lit.set(nt.key, nt.player);
  for (let k = 0; k < n; k++) {
    const p = lit.get(k);
    g.fillStyle = p != null ? FLY_COL[p % 3] : (k % 5 ? "#efeee8" : "#d8d6cc");
    g.fillRect(X(x0 + k * kw) + 1, Y(L / 2), kw * sx - 2, L * sy);
    if (k % 5 === 0) { g.fillStyle = "#555"; g.font = "11px Segoe UI"; g.fillText("C" + (2 + k / 5), X(x0 + k * kw) + 3, Y(-L / 2) - 4); }
  }
  data.flies.forEach((f, i) => {
    const s = sampleTraj(f.trajectory, t); if (!s) return;
    const [, x, y, hd] = s, a = -hd * Math.PI / 180;
    g.save(); g.translate(X(x), Y(y)); g.rotate(a);
    g.fillStyle = FLY_COL[i % 3]; g.beginPath(); g.ellipse(0, 0, 13, 7, 0, 0, 2 * Math.PI); g.fill();
    g.fillStyle = "#b3261e"; g.beginPath(); g.arc(11, -4, 3, 0, 7); g.arc(11, 4, 3, 0, 7); g.fill();
    g.restore();
    g.fillStyle = FLY_COL[i % 3]; g.font = "bold 12px Segoe UI"; g.fillText(`fly ${i + 1}`, X(x) - 14, Y(y) - 14);
  });
  const recent = data.notes.filter(nt => nt.t <= t).length;
  P.info.innerHTML = `t = ${t.toFixed(3)} s (8x slowed) · ${recent}/${data.notes.length} notes · `
    + data.flies.map((f, i) => `<span style="color:${FLY_COL[i % 3]}">■ fly ${i + 1}: ${f.behavior.replace("_", " ")} (${f.n_notes})</span>`).join(" · ");
}

function drawRoll() {
  const { cv, data, media } = P, g = cv.getContext("2d"), W = cv.width, H = cv.height;
  const t = media.currentTime, D = data.duration_s, lanes = ["drums", "melody", "bass"], lh = (H - 30) / 3;
  const X = (s) => 120 + (s / D) * (W - 130);
  g.fillStyle = "#151514"; g.fillRect(0, 0, W, H);
  const bar = data.eighth_s * 8;
  for (let b = 0; b < data.bars; b++) {
    if (data.bitter_bars.includes(b)) { g.fillStyle = "rgba(139,92,246,0.18)"; g.fillRect(X(b * bar), 0, X(bar) - X(0), H - 30); }
    g.strokeStyle = "#2c2c29"; g.beginPath(); g.moveTo(X(b * bar), 0); g.lineTo(X(b * bar), H - 30); g.stroke();
    g.fillStyle = "#8a897f"; g.font = "11px Segoe UI"; g.fillText(`bar ${b + 1}` + (data.bitter_bars.includes(b) ? " + bitter" : ""), X(b * bar) + 3, H - 14);
  }
  lanes.forEach((ln, i) => {
    g.fillStyle = LANE[ln][0]; g.font = "bold 12px Segoe UI"; g.fillText(LANE[ln][1].split(" · ")[0], 8, i * lh + 18);
    g.fillStyle = "#8a897f"; g.font = "10px Segoe UI"; g.fillText(LANE[ln][1].split(" · ")[1], 8, i * lh + 32);
    const ns = data.notes.filter(n => n.musician === ln); if (!ns.length) return;
    const lo = Math.min(...ns.map(n => n.midi)), hi = Math.max(...ns.map(n => n.midi)) + 1;
    for (const n of ns) {
      const y = i * lh + 6 + (1 - (n.midi - lo) / Math.max(1, hi - lo)) * (lh - 16);
      const on = t >= n.t && t < n.t + 0.2;
      g.globalAlpha = 0.35 + 0.65 * n.vel; g.fillStyle = on ? "#fff" : LANE[ln][0];
      g.fillRect(X(n.t), y, Math.max(3, X(data.eighth_s) - X(0) - 2), 7); g.globalAlpha = 1;
    }
  });
  g.strokeStyle = "#f0efec"; g.lineWidth = 2; g.beginPath(); g.moveTo(X(t), 0); g.lineTo(X(t), H - 30); g.stroke(); g.lineWidth = 1;
  const i = Math.min(data.slots.bass_MN9.length - 1, Math.floor(t / data.eighth_s));
  P.info.innerHTML = `t = ${t.toFixed(2)} s · MN9 <b>${data.slots.bass_MN9[i]} Hz</b> · turn signal <b>${data.slots.melody_turn[i]}</b> · giant fiber <b>${data.slots.drums_GF[i]} Hz</b>`
    + ` <span style="color:#8a897f">· score designed by us, notes decided by the brain</span>`;
}

function drawSong() {
  const { cv, data, media } = P, g = cv.getContext("2d"), W = cv.width, H = cv.height;
  const t = media.currentTime / data.time_scale, seq = data.seq;
  const total = seq.reduce((a, e) => a + e.beats, 0), ms = seq.filter(e => e.midi != null).map(e => e.midi);
  const lo = Math.min(...ms) - 1, hi = Math.max(...ms) + 1, X = (b) => 10 + b / total * (W - 20), Y = (m) => 14 + (1 - (m - lo) / (hi - lo)) * (H - 40);
  g.fillStyle = "#151514"; g.fillRect(0, 0, W, H);
  let acc = 0, cur = null, nPlayed = 0;
  for (const e of seq) {
    const x0 = X(acc), w = Math.max(2, X(acc + e.beats) - x0 - 2); acc += e.beats;
    if (e.midi == null) continue;
    const done = e.played && t >= e.t, y = Y(e.midi);
    if (done) nPlayed++;
    if (e.played && t >= e.t && (!cur || e.t > cur.t)) cur = e;
    const hue = ((e.midi - 60) % 12 + 12) % 12 * 30;
    g.fillStyle = done ? `hsl(${hue},75%,62%)` : "#2c2c29"; g.fillRect(x0, y - 7, w, 14);
    if (!e.played) { g.strokeStyle = "#d03b3b"; g.lineWidth = 2; g.strokeRect(x0, y - 7, w, 14); g.lineWidth = 1; }
  }
  if (cur) { g.fillStyle = "#fff"; g.font = "bold 16px Segoe UI"; g.fillText(cur.name, 14, H - 8); }
  const missed = seq.filter(e => e.midi != null && !e.played).length;
  P.info.innerHTML = `<b>${data.title}</b> · ${nPlayed}/${seq.filter(e => e.midi != null).length} notes so far` + (missed ? ` · <span style="color:#d03b3b">${missed} missed (red)</span>` : "")
    + ` <span style="color:#8a897f">· melody = our score; ` + (data.video ? "rhythm = the fly's real footsteps (8x slowed)" : "each note sounds only if the singer's MN9 fires > 20 Hz") + `</span>`;
}

function pianoLoop() {
  if (P && P.box.offsetParent !== null) {
    const r = P.cv.getBoundingClientRect();
    if (P.cv.width !== Math.round(r.width) || P.cv.height !== Math.round(r.height)) { P.cv.width = Math.round(r.width); P.cv.height = Math.round(r.height); }
    (P.data.mode === "keyboard" ? drawKeyboard : P.data.mode === "song" ? drawSong : drawRoll)();
  }
  requestAnimationFrame(pianoLoop);
}

window.flylabShowPiano = async (beh) => {
  if (P && P.runId === beh.run_id) return;
  const data = await (await fetch(`/${beh.piano}?t=${beh._mtime}`)).json();
  const box = document.getElementById("pianobox"); box.innerHTML = "";
  if (data.mode === "song" && !data.video && window.flylabShowBand) {   // singer flies: the 3D fly band at the piano
    if (P) { try { P.media.pause(); } catch (e) {} }
    P = null; await window.flylabShowBand(box, data, beh); return;
  }
  if (window.flylabHideBand) window.flylabHideBand();
  box.style.cssText = "position:absolute;inset:0;display:flex;flex-direction:column;gap:6px;padding:8px;";
  let media;
  if (data.mode === "keyboard" || (data.mode === "song" && data.video)) {
    media = el("video"); media.src = `/${data.video}?t=${beh._mtime}`; media.controls = true; media.loop = true; media.playsInline = true;
    media.style.cssText = "flex:1.4;min-height:0;width:100%;object-fit:contain;background:#000";
    box.appendChild(el("div", "font-size:12px;color:#c3c2b7", data.mode === "song" ? `▶ press play for sound · musical road: ${data.title}` : "▶ press play for sound · fly 1 = real physics video; the keyboard below shows the whole gang (each fly simulated separately)"));
    box.appendChild(media);
  } else {
    media = el("audio"); media.src = `/${data.audio}?t=${beh._mtime}`; media.controls = true; media.loop = true;
    media.style.cssText = "width:100%";
    box.appendChild(el("div", "font-size:13px;color:#c3c2b7", data.mode === "song" ? `<b>Singer fly</b> · ${data.title} · ▶ press play` : "<b>Brain orchestra</b> · three musician flies · ▶ press play"));
    box.appendChild(media);
  }
  const cv = el("canvas", "flex:1;min-height:0;width:100%;border-radius:6px");
  const info = el("div", "font-size:12.5px;color:#f0efec;min-height:18px");
  box.appendChild(cv); box.appendChild(info);
  P = { box, cv, info, data, media, runId: beh.run_id };
};
window.flylabHidePiano = () => { if (P) { P.media.pause(); P = null; } if (window.flylabHideBand) window.flylabHideBand(); };
requestAnimationFrame(pianoLoop);
})();
