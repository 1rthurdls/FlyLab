"""Physics-simulated fly BODY for FlyLab (NeuroMechFly v2 via flygym 1.2.1).

The brain model (``flylab.brain``) produces a *MotorProgram*: a few coarse,
per-time-bin descending "drives" (walk forward, turn, walk backward, feed).
This module plays such a program on the NeuroMechFly v2 body in MuJoCo and
films it.

IMPORTANT - what is and is not simulated
----------------------------------------
* The body, its legs, ground contact, adhesion and the CPG/reflex walking
  controller are flygym's own published components (NeuroMechFly v2,
  Wang-Chen et al., Nature Methods 2024).  We did not write any locomotion.
* The link between brain and body (``MotorProgram`` -> controller inputs) is
  a HAND-DESIGNED interface, documented exactly below.  There is no
  connectome-derived VNC / motor neuron model in between.
* The proboscis is NOT articulated in the stock NeuroMechFly v2 model (the
  ``Rostrum`` and ``Haustellum`` bodies are welded to the head).  We add two
  hinge joints (pitch axis) to those bodies plus position actuators, so the
  proboscis can physically swing out ("proboscis extension").  The extension
  posture (angles below) was chosen by eye; it is a visual stand-in for PER,
  not a biomechanical model of the labellum.

Mapping MotorProgram -> body (per physics step, dt = 1e-4 s)
------------------------------------------------------------
flygym's ``HybridTurningController`` takes a 2-vector descending drive
``[d_left, d_right]``.  For each side, the CPG oscillators of that side's 3
legs get amplitude ``|d|`` and frequency ``sign(d) * 12 Hz``: a positive drive
walks forward, a negative drive runs the step cycle in reverse (backward
walking), a lower drive on one side shortens its strides (turning toward that
side).  Stepping frequency is fixed (12 Hz); the drive scales stride
amplitude, hence speed.  Sensory stumbling/retraction reflexes stay active.
(flygym declares the action box as [-0.5, 1.5] but does not clip; backward
drive of -1 was stable in our tests.)

For bin ``i`` (raw program values, clipped to their ranges)::

    gate      = 0 if feeding[i] > 0.5 else 1         # feeding suppresses walking
    net       = forward[i] - backward[i]              # in [-1, 1]
    d_left    = gate * (net - TURN_GAIN * turn[i])   # TURN_GAIN = 0.6
    d_right   = gate * (net + TURN_GAIN * turn[i])
    d_*       clipped to [-1.2, 1.4]

    turn > 0  -> left legs slower than right legs -> counter-clockwise (LEFT)
    turn alone (net = 0) -> left/right legs step in opposite directions
                            (turning on the spot)

Then ``d_left``, ``d_right`` and ``feeding[i]`` each pass through a first-order
low-pass filter (time constant ``drive_tau`` = 50 ms) so bin-to-bin jumps do
not jerk the gait.  The filtered feeding level ``f`` drives the added
proboscis actuators (targets ``f * ROSTRUM_EXT`` and ``f * HAUSTELLUM_EXT``
rad) and a neck actuator that pitches the head down by ``f * HEAD_PITCH`` rad.

Measured with the defaults (flat floor, seed 0, 1.5 s runs):
    forward=1              ~13.5 mm/s, heading drifts ~+20 deg (CPG asymmetry)
    forward=0.8, turn=+0.6 ~140 deg/s counter-clockwise (left), ~10 mm/s
    backward=1             ~8 mm/s backward, heading drift ~-5 deg
    feeding=1 after walking: stops within ~0.1 s (drifts < 1 mm),
                             proboscis reaches ~95% extension

Sugar contact
-------------
If ``sugar_pos`` is given, a sugar drop is drawn on the floor (visual only, no
collision): an amber puddle whose radius equals ``sugar_radius`` plus a glossy
dome.  "Contact" = the horizontal (x, y) distance from ``sugar_pos`` to a part
listed in ``CONTACT_PARTS`` is < ``sugar_radius`` mm, checked every 1 ms of
simulated time.  Default: the haustellum (proboscis tip body) only, so the fly
stops with its mouth over the drop.  The front tarsal tips (LFTarsus5 /
RFTarsus5) are also tracked and can be enabled.

Speed (i7-1360P laptop, CPU only, one process): ~60-80 s wall per simulated
second without rendering, ~75-125 s with 960x720 rendering at 0.25x play
speed (~0.13 s per frame), plus ~13 s to build the model.  MuJoCo itself is
~3-5 ms per 0.1 ms step (2220 leg collision pairs); the rest is flygym Python.

Coordinates: mm, seconds.  The fly spawns at (0, 0) facing +x (heading 0 deg);
heading is counter-clockwise from +x seen from above.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# flygym / mujoco imports are deferred to FlyBody.__init__ so that importing
# MotorProgram (e.g. from the brain side) stays cheap.

# --- interface constants (see module docstring) ---------------------------
TURN_GAIN = 0.6
DRIVE_MIN, DRIVE_MAX = -1.2, 1.4
FEEDING_GATE = 0.5
ROSTRUM_EXT = -1.0      # rad, rostrum swings forward/down
HAUSTELLUM_EXT = 1.1    # rad, haustellum unfolds down
HEAD_PITCH = 0.3        # rad, head pitches down while feeding
CONTACT_CHECK_S = 1e-3
# Body parts whose (x, y) position inside the sugar drop counts as "contact" (taste).
# Proboscis only (FlyLab choice): the fly then stops with its mouth over the drop.
# The front tarsi (also taste organs in real flies) are tracked but not used by default.
CONTACT_PARTS = ("proboscis",)

_LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]
_CONTACT_SENSORS = [
    f"{leg}{seg}"
    for leg in _LEGS
    for seg in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]


@dataclass
class MotorProgram:
    """Per-bin descending drives produced by the brain.

    bin_s    : duration of each bin in seconds (e.g. 0.02)
    forward  : per-bin forward walking drive in [0, 1]
    turn     : per-bin steering in [-1, 1]; positive = turn LEFT
               (counter-clockwise seen from above)
    backward : per-bin backward walking drive in [0, 1]
    feeding  : per-bin feeding drive in [0, 1] (proboscis extension;
               > 0.5 also stops walking)
    """

    bin_s: float
    forward: np.ndarray
    turn: np.ndarray
    backward: np.ndarray
    feeding: np.ndarray

    def __post_init__(self):
        self.bin_s = float(self.bin_s)
        if self.bin_s <= 0:
            raise ValueError("bin_s must be > 0")
        self.forward = np.clip(np.asarray(self.forward, float).ravel(), 0, 1)
        self.turn = np.clip(np.asarray(self.turn, float).ravel(), -1, 1)
        self.backward = np.clip(np.asarray(self.backward, float).ravel(), 0, 1)
        self.feeding = np.clip(np.asarray(self.feeding, float).ravel(), 0, 1)
        n = {len(self.forward), len(self.turn), len(self.backward), len(self.feeding)}
        if len(n) != 1:
            raise ValueError("forward/turn/backward/feeding must have equal length")

    @property
    def n_bins(self) -> int:
        return len(self.forward)

    @property
    def duration_s(self) -> float:
        return self.n_bins * self.bin_s

    @classmethod
    def constant(cls, duration_s: float, bin_s: float = 0.02, forward=0.0,
                 turn=0.0, backward=0.0, feeding=0.0) -> "MotorProgram":
        """Program with constant drives for ``duration_s`` seconds."""
        n = max(1, int(round(duration_s / bin_s)))
        return cls(bin_s, np.full(n, forward), np.full(n, turn),
                   np.full(n, backward), np.full(n, feeding))

    @classmethod
    def concat(cls, programs: Sequence["MotorProgram"]) -> "MotorProgram":
        bins = {p.bin_s for p in programs}
        if len(bins) != 1:
            raise ValueError("all programs must share bin_s")
        return cls(bins.pop(),
                   np.concatenate([p.forward for p in programs]),
                   np.concatenate([p.turn for p in programs]),
                   np.concatenate([p.backward for p in programs]),
                   np.concatenate([p.feeding for p in programs]))


def drives_for_bin(forward: float, turn: float, backward: float,
                   feeding: float) -> tuple[float, float]:
    """Raw (unfiltered) [d_left, d_right] descending drive for one bin."""
    gate = 0.0 if feeding > FEEDING_GATE else 1.0
    net = forward - backward
    d_left = gate * (net - TURN_GAIN * turn)
    d_right = gate * (net + TURN_GAIN * turn)
    return (float(np.clip(d_left, DRIVE_MIN, DRIVE_MAX)),
            float(np.clip(d_right, DRIVE_MIN, DRIVE_MAX)))


def _cached_obs_fly_class():
    """flygym ``Fly`` whose observation is cached per physics time.

    ``HybridTurningController.step`` computes the observation once at the
    start of a step and the base Simulation computes it again after the
    physics step; the first call always sees exactly the state the previous
    post-step call saw.  Caching on ``physics.data.time`` removes that
    duplicate (pure Python) work - results are bit-identical.
    """
    from flygym import Fly

    class _CachedObsFly(Fly):
        _obs_key = None
        _obs_val = None

        def get_observation(self, sim):
            key = float(sim.physics.data.time)
            if self._obs_key is not None and key == self._obs_key:
                return self._obs_val
            obs = super().get_observation(sim)
            self._obs_key, self._obs_val = key, obs
            return obs

        def reset(self, sim, **kwargs):
            self._obs_key = self._obs_val = None
            return super().reset(sim, **kwargs)

    return _CachedObsFly


def _look_at_xmat(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """MuJoCo camera rotation (row-major 3x3, flattened) looking at target."""
    f = target - cam_pos
    f = f / np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, f)
    return np.column_stack([right, up, -f]).ravel()


class FlyBody:
    """Resumable NeuroMechFly v2 body driven by MotorPrograms.

    Parameters
    ----------
    sugar_pos : optional (x, y) in mm of a sugar drop on the floor.
    sugar_radius : contact radius in mm (see module docstring).
    render : record video frames during ``run``.
    fps : video frame rate.
    play_speed : video speed relative to simulated time (0.25 = 4x slow
        motion; at 1.0 the ~12 Hz stepping is barely resolvable at 30 fps).
    window_size : (width, height) of the video in pixels.
    cam_distance, cam_elevation_deg, cam_azimuth_deg : follow camera placed at
        that distance / elevation / azimuth (deg, CCW from +x, world-fixed) from
        the fly; it tracks the (smoothed) fly position but does not rotate
        with the fly, so turns are visible.  When a sugar drop is near
        (< 15 mm) the camera aims part-way toward it so both stay in frame.
    hud : draw a small text overlay (time, drives) on the frames.
    drive_tau : low-pass time constant (s) applied to the drives.
    self_collisions : flygym leg self-collision preset ("legs", "legs-no-coxa",
        "tarsi" or "none"); "legs" is flygym's default.
    settle_time : seconds of zero drive simulated at construction so the fly
        lands on its feet (not recorded, not counted in ``time``).
    seed : seed of the CPG initial phases.
    """

    def __init__(self, sugar_pos: Optional[Sequence[float]] = None,
                 sugar_radius: float = 1.5, render: bool = True, fps: int = 30,
                 play_speed: float = 0.25, window_size=(960, 720),
                 cam_distance: float = 11.0, cam_elevation_deg: float = 30.0,
                 cam_azimuth_deg: float = -65.0, hud: bool = True,
                 drive_tau: float = 0.05, timestep: float = 1e-4,
                 self_collisions: str = "legs", settle_time: float = 0.05,
                 seed: int = 0):
        from flygym.arena import FlatTerrain
        from flygym.examples.locomotion import HybridTurningController

        self.sugar_pos = None if sugar_pos is None else np.asarray(sugar_pos, float)[:2]
        self.sugar_radius = float(sugar_radius)
        self.render_enabled = bool(render)
        self.fps = int(fps)
        self.play_speed = float(play_speed)
        self.window_size = (int(window_size[0]), int(window_size[1]))
        self.hud = hud
        self.drive_tau = float(drive_tau)
        self.timestep = float(timestep)

        # ---- arena: flat floor with a finer checkerboard so motion is visible
        arena = FlatTerrain(size=(300, 300))
        mat = arena.root_element.find("material", "grid")
        mat.texrepeat = (300, 300)          # 2 mm repeat -> 1 mm checks
        tex = mat.texture
        tex.rgb1 = (0.36, 0.38, 0.42)
        tex.rgb2 = (0.44, 0.46, 0.50)
        mat.reflectance = 0.0               # floor reflection doubles render cost
        arena.root_element.worldbody.add(
            "light", name="key_light", mode="trackcom", directional=True,
            castshadow=True, pos=(0, 0, 50), dir=(0.35, 0.25, -1),
            diffuse=(0.55, 0.55, 0.55), specular=(0.1, 0.1, 0.1))
        if self.sugar_pos is not None:
            sx, sy = self.sugar_pos
            r = self.sugar_radius
            wb = arena.root_element.worldbody
            # flat puddle = exactly the contact zone; glossy dome = the drop
            wb.add("geom", name="sugar_puddle", type="cylinder",
                   size=(r, 0.01), pos=(sx, sy, 0.01),
                   rgba=(0.98, 0.80, 0.38, 0.9), contype=0, conaffinity=0)
            wb.add("geom", name="sugar_dome", type="ellipsoid",
                   size=(0.55 * r, 0.55 * r, 0.25 * r), pos=(sx, sy, 0.0),
                   rgba=(1.0, 0.93, 0.70, 0.85), contype=0, conaffinity=0)
        # camera (pose overwritten every frame)
        self._cam = arena.root_element.worldbody.add(
            "camera", name="flylab_follow", mode="fixed", pos=(-5, -5, 5),
            fovy=35)

        # ---- fly: stock NMF v2 + proboscis joints/actuators + neck actuator
        Fly = _cached_obs_fly_class()
        fly = Fly(enable_adhesion=True, draw_adhesion=False,
                  contact_sensor_placements=_CONTACT_SENSORS,
                  self_collisions=self_collisions, spawn_pos=(0.0, 0.0, 0.5))
        # offscreen buffer must fit the video and agree between fly and arena
        fly_glob = getattr(fly.model.visual, "global")
        ow = max(self.window_size[0], fly_glob.offwidth or 640)
        oh = max(self.window_size[1], fly_glob.offheight or 480)
        for root in (fly.model, arena.root_element):
            getattr(root.visual, "global").offwidth = ow
            getattr(root.visual, "global").offheight = oh
            root.visual.quality.shadowsize = 2048
        for body_name in ("Rostrum", "Haustellum"):
            body = fly.model.find("body", body_name)
            geom = fly.model.find("geom", body_name)   # darker so PER reads on video
            geom.material = None
            geom.rgba = (0.40, 0.20, 0.10, 1.0)
            body.add("joint", name=f"joint_{body_name}", type="hinge",
                     axis=(0, 1, 0), stiffness=0.0, damping=0.005,
                     limited=True, range=(-1.6, 1.6))
        for wing in ("LWing", "RWing"):
            fly.model.find("body", wing).add(
                "joint", name=f"joint_{wing}_flap", type="hinge", axis=(1, 0, 0), stiffness=0.0,
                damping=0.001, armature=1e-4, limited=True, range=(-0.01, 0.01))
        self._prob_act = [
            fly.model.actuator.add("position", name=f"actuator_position_joint_{b}",
                                   joint=f"joint_{b}", kp=2.0,
                                   ctrlrange="-3 3", ctrllimited=True)
            for b in ("Rostrum", "Haustellum")
        ]
        self._neck_kp = 10.0
        self._neck_act = fly.model.actuator.add(
            "position", name="actuator_position_joint_Head_pitch",
            joint="joint_Head", kp=self._neck_kp, ctrlrange="-3 3",
            ctrllimited=True)
        # neck has a passive spring (flygym neck_stiffness); compensate so the
        # head settles at the requested pitch
        self._neck_gain = (self._neck_kp + (fly.neck_stiffness or 0.0)) / self._neck_kp
        self.fly = fly

        self.sim = HybridTurningController(fly=fly, cameras=[], arena=arena,
                                           timestep=self.timestep, seed=seed)
        self.obs, _ = self.sim.reset(seed=seed)
        ph = self.sim.physics
        self._ph = ph
        n = fly.name
        self._body_ids = {
            "proboscis": ph.model.name2id(f"{n}/Haustellum", "body"),
            "LF_leg": ph.model.name2id(f"{n}/LFTarsus5", "body"),
            "RF_leg": ph.model.name2id(f"{n}/RFTarsus5", "body"),
        }
        self._thorax_id = ph.model.name2id(f"{n}/Thorax", "body")
        self._rostrum_qadr = ph.model.jnt_qposadr[ph.model.name2id(f"{n}/joint_Rostrum", "joint")]
        self._cam_id = ph.model.name2id("flylab_follow", "camera")
        self._wing_qadr = [ph.model.jnt_qposadr[ph.model.name2id(f"{n}/joint_{w}_flap", "joint")]
                           for w in ("LWing", "RWing")]
        free = [j for j in range(ph.model.njnt) if ph.model.jnt_type[j] == 0]
        self._free_qadr = int(ph.model.jnt_qposadr[free[0]]) if free else None
        self._cam_z = 0.6
        self.airborne = False
        self.cam_distance = float(cam_distance)
        self._cam_dir = np.array([
            np.cos(np.radians(cam_elevation_deg)) * np.cos(np.radians(cam_azimuth_deg)),
            np.cos(np.radians(cam_elevation_deg)) * np.sin(np.radians(cam_azimuth_deg)),
            np.sin(np.radians(cam_elevation_deg))])

        # controller state (filtered drives) persists between run() calls
        self._d = np.zeros(2)       # filtered [d_left, d_right]
        self._feed = 0.0            # filtered feeding level
        self.time = 0.0             # body time (s), excludes settle phase
        self.frames: list[np.ndarray] = []
        self._next_frame_t = 0.0
        self._cam_xy = None
        self._heading_unwrapped = None
        self.flipped = False
        self.physics_error: Optional[str] = None
        self.trajectory: list[tuple] = []   # (t, x, y, heading_deg, dL, dR, feed)
        self._hud_text = ""

        # settle: let the fly land with zero drive (not recorded)
        for _ in range(int(round(settle_time / self.timestep))):
            self._physics_step()
        self._heading_unwrapped = self._raw_heading()
        self._cam_xy, self._cam_dist = self._camera_goal()
        self._log()

    # ------------------------------------------------------------------ state
    @property
    def position(self) -> np.ndarray:
        """Thorax (x, y) in mm."""
        return np.array(self.obs["fly"][0][:2], dtype=float)

    def _raw_heading(self) -> float:
        fwd = self.obs["cardinal_vectors"][0]
        return float(np.arctan2(fwd[1], fwd[0]))

    def _update_heading(self):
        h = self._raw_heading()
        if self._heading_unwrapped is None:
            self._heading_unwrapped = h
        else:
            dh = (h - self._heading_unwrapped + np.pi) % (2 * np.pi) - np.pi
            self._heading_unwrapped += dh

    @property
    def heading(self) -> float:
        """Heading in degrees, CCW from +x, unwrapped (can exceed +-180)."""
        return float(np.degrees(self._heading_unwrapped))

    @property
    def proboscis_extension(self) -> float:
        """Measured rostrum angle as a fraction of full extension, in [0, 1]."""
        q = float(self._ph.data.qpos[self._rostrum_qadr])
        return float(np.clip(q / ROSTRUM_EXT, 0.0, 1.0))

    def _contact_part(self) -> Optional[str]:
        if self.sugar_pos is None:
            return None
        xpos = self._ph.data.xpos
        for part, bid in self._body_ids.items():
            if part not in CONTACT_PARTS:
                continue
            if np.hypot(*(xpos[bid, :2] - self.sugar_pos)) < self.sugar_radius:
                return part
        return None

    def _log(self):
        p = self.position
        self.trajectory.append((self.time, p[0], p[1], self.heading,
                                self._d[0], self._d[1], self._feed))

    # ---------------------------------------------------------------- physics
    def _physics_step(self):
        f = self._feed
        self._ph.bind(self._prob_act).ctrl = [f * ROSTRUM_EXT, f * HAUSTELLUM_EXT]
        self._ph.bind(self._neck_act).ctrl = f * HEAD_PITCH * self._neck_gain
        self.obs, _, _, _, info = self.sim.step(self._d.copy())
        if info.get("flip", False):
            self.flipped = True

    # -------------------------------------------------------------- rendering
    def _camera_goal(self) -> tuple[np.ndarray, float]:
        """Camera aim point (x, y) and distance.

        Normally the fly itself at ``cam_distance``.  When a sugar drop is
        within 15 mm the aim moves toward the fly-sugar midpoint and the camera
        backs off just enough to keep both in frame (zooming back in as the
        fly arrives).
        """
        goal = self._ph.data.xpos[self._thorax_id, :2].copy()
        dist = self.cam_distance
        if self.sugar_pos is not None:
            d = float(np.hypot(*(self.sugar_pos - goal)))
            w = float(np.clip((15.0 - d) / 5.0, 0, 1))
            goal += 0.5 * w * (self.sugar_pos - goal)
            dist = max(dist, w * (0.5 * d + 2.5) / 0.42)
        return goal, dist

    def _render_frame(self):
        ph = self._ph
        # smooth camera target (tau 0.08 s of sim time)
        alpha = min(1.0, (self.play_speed / self.fps) / 0.08)
        goal_xy, goal_dist = self._camera_goal()
        self._cam_xy = (1 - alpha) * self._cam_xy + alpha * goal_xy
        self._cam_dist = (1 - alpha) * self._cam_dist + alpha * goal_dist
        target = np.array([self._cam_xy[0], self._cam_xy[1], self._cam_z])
        cam_pos = target + self._cam_dist * self._cam_dir
        ph.data.cam_xpos[self._cam_id] = cam_pos
        ph.data.cam_xmat[self._cam_id] = _look_at_xmat(cam_pos, target)
        w, h = self.window_size
        img = ph.render(width=w, height=h, camera_id=self._cam_id).copy()
        if self.hud:
            import cv2
            lines = [f"t = {self.time:5.2f} s  ({self.play_speed:g}x speed)",
                     self._hud_text]
            if self._feed > 0.05:
                lines.append(f"proboscis extension {self.proboscis_extension:.0%}")
            for k, line in enumerate(lines):
                y = 32 + 28 * k
                cv2.putText(img, line, (18, y), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                            (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(img, line, (18, y), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                            (30, 30, 30), 1, cv2.LINE_AA)
        self.frames.append(img)

    # -------------------------------------------------------------------- run
    def run(self, program: MotorProgram, stop_on_sugar_contact: bool = False) -> dict:
        """Play ``program`` on the body (state persists across calls).

        Body pose, CPG phases, filtered drives, clock, recorded frames and
        trajectory all carry over between calls, so a caller can run the
        approach with ``stop_on_sugar_contact=True``, switch the brain
        stimulus, and call ``run()`` again with the next program.

        ``sugar_contact`` refers to the first contact detected during THIS
        call (checked every 1 ms).  With ``stop_on_sugar_contact`` the call
        returns at that moment (mid-bin); if the fly is already in contact
        when the call starts it returns after ~1 ms.

        Returns a summary dict:
          sim_time_s, t_start_s, t_end_s (body clock), start_xy, end_xy,
          heading_start_deg, heading_end_deg, heading_change_deg (CCW +),
          distance_walked_mm (path length of the 50 ms low-passed thorax
          position), net_displacement_mm,
          sugar_contact (bool), sugar_contact_time_s (body clock) and
          sugar_contact_part ('proboscis' / 'LF_leg' / 'RF_leg'),
          stopped_on_contact, mean_proboscis_extension (measured, 0..1),
          mean_drive_left/right, flipped, physics_error, n_frames_total,
          wall_time_s.
        """
        from dm_control.rl.control import PhysicsError

        wall0 = time.perf_counter()
        t0 = self.time
        p0 = self.position
        h0 = self.heading
        path = 0.0
        # path length is measured on a low-passed (tau 50 ms) thorax position
        # so the lateral sway of each stride is not counted as distance
        p_lp = p0.copy()
        last_p = p0
        lp_a = CONTACT_CHECK_S / 0.05
        contact_time = None
        contact_part = None
        stopped = False
        prob_sum, prob_n = 0.0, 0
        dsum = np.zeros(2)
        n_steps_bin = max(1, int(round(program.bin_s / self.timestep)))
        check_every = max(1, int(round(CONTACT_CHECK_S / self.timestep)))
        a = min(1.0, self.timestep / self.drive_tau) if self.drive_tau > 0 else 1.0
        step_count = 0

        try:
            for i in range(program.n_bins):
                fw, tn, bw, fd = (program.forward[i], program.turn[i],
                                  program.backward[i], program.feeding[i])
                target = np.array(drives_for_bin(fw, tn, bw, fd))
                self._hud_text = (f"fwd {fw:.2f}  turn {tn:+.2f}  back {bw:.2f}"
                                  f"  feed {fd:.2f}")
                for _ in range(n_steps_bin):
                    self._d += a * (target - self._d)
                    self._feed += a * (fd - self._feed)
                    self._physics_step()
                    self.time += self.timestep
                    step_count += 1
                    dsum += self._d
                    if step_count % check_every == 0:
                        self._update_heading()
                        p_lp += lp_a * (self.position - p_lp)
                        path += float(np.hypot(*(p_lp - last_p)))
                        last_p = p_lp.copy()
                        prob_sum += self.proboscis_extension
                        prob_n += 1
                        if step_count % (10 * check_every) == 0:
                            self._log()
                        part = self._contact_part()
                        if part is not None and contact_time is None:
                            contact_time, contact_part = self.time, part
                            if stop_on_sugar_contact:
                                stopped = True
                    if self.render_enabled and self.time >= self._next_frame_t:
                        self._render_frame()
                        self._next_frame_t += self.play_speed / self.fps
                    if stopped:
                        break
                if stopped:
                    break
        except PhysicsError as e:   # numerical blow-up: keep what we have
            self.physics_error = str(e)

        self._update_heading()
        p1 = self.position
        path += float(np.hypot(*(p1 - last_p)))   # close the lag of the filter
        self._log()
        return {
            "sim_time_s": round(self.time - t0, 6),
            "t_start_s": round(t0, 6),
            "t_end_s": round(self.time, 6),
            "start_xy": [round(float(v), 4) for v in p0],
            "end_xy": [round(float(v), 4) for v in p1],
            "heading_start_deg": round(h0, 2),
            "heading_end_deg": round(self.heading, 2),
            "heading_change_deg": round(self.heading - h0, 2),
            "distance_walked_mm": round(path, 4),
            "net_displacement_mm": round(float(np.hypot(*(p1 - p0))), 4),
            "sugar_contact": contact_time is not None,
            "sugar_contact_time_s": None if contact_time is None else round(contact_time, 4),
            "sugar_contact_part": contact_part,
            "stopped_on_contact": stopped,
            "mean_proboscis_extension": round(prob_sum / prob_n, 4) if prob_n else 0.0,
            "mean_drive_left": round(float(dsum[0] / max(step_count, 1)), 4),
            "mean_drive_right": round(float(dsum[1] / max(step_count, 1)), 4),
            "flipped": self.flipped,
            "physics_error": self.physics_error,
            "n_frames_total": len(self.frames),
            "wall_time_s": round(time.perf_counter() - wall0, 2),
        }

    def scripted_takeoff(self, duration_s: float = 0.9, climb_mm: float = 9.0, forward_mm: float = 14.0,
                         flap_hz_video: float = 9.0) -> dict:
        """SCRIPTED take-off and short flight (kinematic animation, no aerodynamics).

        NeuroMechFly v2 in flygym is a walking model and cannot fly. When the brain's giant fiber
        fires above threshold (see interface.py), FlyLab plays this fixed animation: the fly
        lifts off along a single pre-defined climbing path in the direction it is facing, with the
        wings flapping (wing-beat slowed down so it is visible on video). Physics is not stepped
        during the animation; the video ends with the fly airborne.
        """
        if self._free_qadr is None:
            return {"takeoff": False, "reason": "no free joint"}
        ph = self._ph
        a = self._free_qadr
        q0 = ph.data.qpos[a:a + 3].copy()
        h = np.radians(self.heading)
        frame_dt = self.play_speed / self.fps
        n = max(1, int(round(duration_s / frame_dt)))
        t0 = self.time
        self.airborne = True
        for k in range(n):
            s_ = (k + 1) / n
            z = q0[2] + climb_mm * (1 - (1 - s_) ** 2.2)
            d = forward_mm * s_ ** 1.5
            ph.data.qpos[a:a + 3] = [q0[0] + d * np.cos(h), q0[1] + d * np.sin(h), z]
            ang = 1.0 * np.sin(2 * np.pi * flap_hz_video * (k / self.fps))
            ph.data.qpos[self._wing_qadr[0]] = ang
            ph.data.qpos[self._wing_qadr[1]] = -ang
            ph.forward()
            self.time += frame_dt
            self._cam_z = 0.6 + 0.85 * (z - q0[2])
            self._hud_text = "TAKE-OFF: giant fiber fired -> scripted flight animation (no aerodynamics)"
            if self.render_enabled:
                self._render_frame()
            self._log()
        return {"takeoff": True, "t_start_s": round(t0, 3), "t_end_s": round(self.time, 3),
                "climb_mm": climb_mm, "forward_mm": forward_mm, "scripted": True}

    def save_video(self, path) -> str:
        """Write all frames recorded so far (across run() calls) to an mp4."""
        import imageio

        if not self.frames:
            raise RuntimeError("no frames recorded (render=False or nothing run yet)")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(str(path), fps=self.fps, codec="libx264",
                                quality=8, macro_block_size=8,
                                ffmpeg_log_level="error") as w:
            for fr in self.frames:
                w.append_data(fr)
        return str(path)

    def close(self):
        try:
            self.sim.close()
        except Exception:
            pass


def render_program(program: MotorProgram, out_mp4, sugar_pos=None,
                   stop_on_sugar_contact: bool = False, **body_kwargs) -> dict:
    """One-shot: build a FlyBody, run ``program``, save ``out_mp4``.

    Returns the run() summary plus 'video' (path) and 'trajectory'
    (list of (t, x, y, heading_deg, d_left, d_right, feeding) every 10 ms).
    """
    body = FlyBody(sugar_pos=sugar_pos, **body_kwargs)
    summary = body.run(program, stop_on_sugar_contact=stop_on_sugar_contact)
    if body.render_enabled and body.frames:
        summary["video"] = body.save_video(out_mp4)
    summary["trajectory"] = [tuple(round(float(v), 4) for v in row)
                             for row in body.trajectory]
    body.close()
    return summary
