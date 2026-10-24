"""Plane teleop: drive the TCP in the plane PARALLEL to the camera image, ignoring MediaPipe depth (z) entirely.

Why: the first real teleop run showed that motion TOWARD/AWAY from the camera (monocular depth, MediaPipe z) is the
noisy axis: ~28 mm std while holding still, versus ~3 mm in the image plane. So every control here is read from
IMAGE-PLANE landmarks (x, y) only, never z, and normalized by the shoulder width (in image units) with the shoulder
midpoint as the origin, so the mapping is independent of how far the operator stands from the camera.

Mapping (relative / clutched; the hand positions and the current TCP are captured on engage so there is no jump):
    robot y (lateral)  <- RIGHT hand horizontal   hand to the operator's RIGHT -> gripper to their right (robot -y)
    robot z (vertical) <- RIGHT hand vertical      raise -> up
    robot x (reach)    <- LEFT hand height, as a JOYSTICK RATE (reach_mode "rate", the dev_arm_real default): the
                          hand ABOVE ~shoulder height drives reach forward, BELOW ~mid torso drives it back, in
                          between holds; speed scales with how far past the (body-absolute) threshold the hand is,
                          up to a max, integrated onto tcp0's x and clamped to the box (no wind-up at the wall).
                          reach_mode "position" keeps the old height->x position map (raise -> forward, lower -> back).
    lateral/vertical:  target = tcp0 + ramp * gain * (hand_now - hand0)      (gain in metres per shoulder-width)
then a One-Euro filter on the normalized hand coordinates, a small deadband, a 2 s engage ramp, and the SafetyLayer
workspace box (targets outside the box are clamped and `.clipped` is flagged, so the ghost shows amber). The clamped
target still runs through the normal IK path in the app (orient from the profile's teleop_orient).

The image points come from `BodyPose.skel2d` (already mirror-corrected in the body client, so pressing M / --mirror
keeps this consistent). skel2d order is the body client's SKEL list:
    [0]=L shoulder [1]=R shoulder [2]=L elbow [3]=R elbow [4]=L wrist [5]=R wrist [6]=L hip [7]=R hip [8]=nose.
In that mirror-corrected frame the operator's RIGHT side sits at the SMALLER x (verified on the 2026-10-23 session:
R shoulder x ~0.41 < L shoulder x ~0.74), so moving the right hand to the operator's right DECREASES x; with a
positive lateral gain that drives robot -y, i.e. the gripper to the operator's right. See docs/PLANE_TELEOP.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..types import BodyPose
from .retarget import OneEuro

# skel2d (body client SKEL) indices
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, NOSE = range(9)

# Live-tuning clamp bounds (shared by PlaneMapper.adjust/load and fit_reach_thresholds). A sign flip only changes a
# sign, never a magnitude; these bound the magnitudes so no live tweak can leave a safe range.
RR_MAX_LO, RR_MAX_HI = 0.01, 0.08     # reach speed cap, m/s (0.08 is still well inside the real arm's limits)
LAT_LO, LAT_HI = 0.05, 0.35           # lateral gain, m per shoulder-width
VRT_LO, VRT_HI = 0.05, 0.30           # vertical gain, m per shoulder-width
FRAC_LO, FRAC_HI = 0.03, 0.60         # reach threshold fractions of the shoulder->hip torso span
FRAC_MIN_GAP = 0.05                   # forward_frac must stay this far below back_frac (a non-zero dead zone)
DB_LO, DB_HI = 0.0, 0.05              # position deadband, shoulder-widths
STEP_SPEED = 1.25                     # reach_faster/slower multiply/divide
STEP_GAIN = 1.2                       # more/less sensitive multiply/divide
STEP_FRAC = 0.05                      # reach_*_sooner step
STEP_DB = 0.005                       # dead-zone step


def _clamp(v, lo, hi):
    return min(max(float(v), lo), hi)


def _sgn(v):
    return 1 if v >= 0 else -1

DEFAULTS = {
    "side": "right",                                  # the operator's CONTROL hand (lateral + vertical); other = reach
    "gain": {"lateral": 0.20, "vertical": 0.15, "reach": 0.15},  # metres of TCP per shoulder-width of hand motion
    "sign": {"lateral": 1.0, "vertical": 1.0, "reach": 1.0},     # flip one if a control drives the wrong way
    "one_euro": {"min_cutoff": 1.0, "beta": 0.05},    # on the NORMALIZED hand coords (steady when still, low lag)
    "deadband": 0.015,                                # shoulder-widths: hold a coord until it drifts this far
    "ramp_s": 2.0,                                    # ease the gain in from 0 over this long on engage (no jump)
    "drop_below_hip_s": 0.7,                          # right wrist below the hip line this long -> "right arm dropped"
    "conf_gate": True,                                # skip a pose that is not visible (the TCP holds)
    # reach_mode: "rate" = JOYSTICK reach from the LEFT-hand height (default on dev_arm_real); "position" = the old
    # height->x position map (reach = tcp0 + gain * height-delta). Rate reach fixes the live finding that height->x
    # as a position map gave almost no forward room and was engage-relative (docs/PLANE_TELEOP.md 2026-10-23).
    "reach_mode": "rate",
    # Joystick reach thresholds, ABSOLUTE to the operator's own body (image plane only, never engage-relative). The
    # left-wrist height is read in shoulder-widths above the shoulder midpoint; the shoulder (0) -> hip-midpoint line
    # defines the torso, and the two thresholds are fractions of that torso span measured DOWN from the shoulders:
    #   reach_up ABOVE the forward threshold (around shoulder height)       -> reach moves FORWARD, speed scales
    #   reach_up BELOW the back threshold    (around lower chest/mid torso) -> reach moves BACK
    #   in between                                                          -> dead zone, reach HOLDS
    # Speed grows linearly past a threshold and saturates at max_speed once full_scale shoulder-widths past it. A
    # little hysteresis stops chatter at a threshold. If the reach hand is not tracked the rate is 0 (reach holds).
    "reach_rate": {
        "forward_frac": 0.10,   # fraction of the shoulder->hip span below the shoulders where FORWARD begins
        "back_frac": 0.45,      # fraction below the shoulders where BACK begins (~lower chest / mid torso)
        "max_speed": 0.04,      # m/s cap on the reach speed (well inside the real arm's limits, REAL_DATA 4a)
        "full_scale": 0.5,      # shoulder-widths past a threshold at which max_speed is reached
        "hysteresis": 0.08,     # shoulder-widths of hysteresis at each threshold (anti-chatter)
    },
}

REACH_MAX_DT = 0.2  # s: cap the integration step so a resumed track after a gap can't jump the reach


def _cfg(cfg, key):
    d = DEFAULTS[key]
    v = (cfg or {}).get(key, d)
    if isinstance(d, dict) and isinstance(v, dict):
        out = dict(d)
        out.update(v)
        return out
    return v


def plane_coords(pose: BodyPose, side: str = "right"):
    """Normalized image-plane hand coordinates, or None if the pose is unusable. x, y only; z is never read.

    Returns dict:
      ctrl_x    control hand horizontal, shoulder-widths from the shoulder midpoint (+ = toward operator's LEFT).
      ctrl_up   control hand height, shoulder-widths above the shoulder midpoint (+ = up; image y is flipped).
      reach_up  reach (other) hand height, same units (drives forward/back reach).
      reach_ok  the reach hand is actually tracked (BodyPose.wrist_other visible), so reach_up is trustworthy.
      hip_drop  the hip-midpoint height in the same units (negative: hips sit below the shoulders). The shoulder
                line (0) and this set the operator's torso span, so the rate-reach thresholds can be body-relative.
      below_hip the control wrist is below the hip line (image y) -> used for the "right arm dropped" release.
    """
    sk = pose.skel2d
    if not sk or len(sk) < 8:
        return None
    pts = np.asarray(sk, float)
    l_sh, r_sh = pts[L_SH], pts[R_SH]
    sh_mid = (l_sh + r_sh) / 2.0
    sw = float(np.linalg.norm(l_sh - r_sh))
    if sw < 1e-6:
        return None
    ctrl_wr = pts[R_WR] if side == "right" else pts[L_WR]
    reach_wr = pts[L_WR] if side == "right" else pts[R_WR]
    hip_mid = (pts[L_HIP] + pts[R_HIP]) / 2.0
    hip_y = float(hip_mid[1])

    def nx(p):
        return float((p[0] - sh_mid[0]) / sw)

    def nup(p):
        return float(-(p[1] - sh_mid[1]) / sw)   # image y points DOWN; flip so raising the hand is +

    return {"ctrl_x": nx(ctrl_wr), "ctrl_up": nup(ctrl_wr), "reach_up": nup(reach_wr),
            "reach_ok": pose.wrist_other is not None, "hip_drop": nup(hip_mid),
            "below_hip": float(ctrl_wr[1]) > hip_y}


class PlaneMapper:
    """BodyPose -> a 3-vector TCP target in the arm base frame, for the app's IK path. Drop-in for `retarget`:
    `engage(tcp_now)` on clutch-engage, `pose -> p` each frame, `reset()`, and `.clipped` for the amber ghost."""

    def __init__(self, arm, home, cfg: dict | None = None, ws_min=None, ws_max=None):
        self.arm = arm
        self.home = np.asarray(home, float).copy()
        self.ws_min = None if ws_min is None else np.asarray(ws_min, float)
        self.ws_max = None if ws_max is None else np.asarray(ws_max, float)
        self.side = _cfg(cfg, "side")
        g = _cfg(cfg, "gain")
        s = _cfg(cfg, "sign")
        # per robot axis (x reach, y lateral, z vertical), already carrying the sign
        self.gain = np.array([g["reach"] * s["reach"], g["lateral"] * s["lateral"], g["vertical"] * s["vertical"]])
        self.reach_sign = float(s["reach"])
        oe = _cfg(cfg, "one_euro")
        self.filter = OneEuro(oe.get("min_cutoff", 1.0), oe.get("beta", 0.05))
        self.deadband = float(_cfg(cfg, "deadband"))
        self.ramp_s = float(_cfg(cfg, "ramp_s"))
        self.conf_gate = bool(_cfg(cfg, "conf_gate"))
        self.reach_mode = str(_cfg(cfg, "reach_mode"))
        rr = _cfg(cfg, "reach_rate")
        self.rr_fwd_frac = float(rr["forward_frac"])
        self.rr_back_frac = float(rr["back_frac"])
        self.rr_max = float(rr["max_speed"])
        self.rr_full = max(1e-6, float(rr["full_scale"]))
        self.rr_hyst = float(rr["hysteresis"])
        self.clipped = False
        self.reach_rate = 0.0        # last signed reach speed, m/s (status: HUD shows >>/<< from the sign)
        self.reach_state = "hold"    # ">>" forward, "<<" back, "hold"
        self._last_p = None          # last emitted target (so a sign flip can re-anchor without a jump)
        self._defaults = self._persist_dict()   # the config baseline, for reset_tuning()
        self.reset()

    def reset(self):
        self._tcp0 = None
        self._n0 = None            # captured [ctrl_x, ctrl_up, reach_up] at engage
        self._held = None          # last emitted normalized coords (deadband)
        self._reach_hold = 0.0     # last trustworthy reach_up (held while the reach hand is not tracked)
        self._t_engage = None
        self._reach_off = 0.0      # rate mode: integrated reach offset (m) added to tcp0 x, anti-wind-up at the box
        self._reach_zone = "hold"  # rate mode: latched zone for hysteresis ("fwd"/"back"/"hold")
        self._t_last = None        # rate mode: previous frame time for dt
        self.reach_rate = 0.0
        self.reach_state = "hold"
        self._last_p = None
        self.filter.reset()
        self.clipped = False

    def engage(self, tcp_now, neutral=None, *args, **kwargs):
        """Anchor at the current TCP. `neutral` / extra args are accepted and ignored (the app passes the clutch
        neutral and sometimes recentre=...); plane mode captures its own neutral from the first sample after engage,
        so engaging never jumps (target == tcp0 on the first frame)."""
        self._tcp0 = np.asarray(tcp_now, float)[:3].copy()
        self._n0 = None
        self._held = None
        self._reach_hold = 0.0
        self._t_engage = None
        self._reach_off = 0.0
        self._reach_zone = "hold"
        self._t_last = None
        self.reach_rate = 0.0
        self.reach_state = "hold"
        self._last_p = None
        self.filter.reset()
        self.clipped = False

    def _reach_rate_speed(self, reach_up, hip_drop):
        """Joystick reach speed (m/s, BEFORE the reach sign) + a HUD label, from the reach-hand height vs the body.
        Thresholds are fractions of the shoulder(0) -> hip(hip_drop) torso span, measured DOWN from the shoulders, so
        they are absolute to the operator's body and never engage-relative. Hysteresis latches the zone to stop
        chatter at a threshold; the speed grows linearly past the threshold and saturates at max_speed."""
        span = abs(hip_drop)                    # torso height, shoulder-widths (hips sit below -> hip_drop < 0)
        fwd_t = -self.rr_fwd_frac * span         # reach_up >= fwd_t (near the shoulders)        -> forward
        back_t = -self.rr_back_frac * span       # reach_up <= back_t (near lower chest/mid torso) -> back
        h = self.rr_hyst
        z = self._reach_zone
        if z == "fwd" and reach_up < fwd_t - h:          # only leave a zone once past its threshold by hysteresis
            z = "hold"
        elif z == "back" and reach_up > back_t + h:
            z = "hold"
        if z == "hold":
            if reach_up >= fwd_t:
                z = "fwd"
            elif reach_up <= back_t:
                z = "back"
        self._reach_zone = z
        gain = self.rr_max / self.rr_full        # m/s per shoulder-width past the threshold
        if z == "fwd":
            return min(self.rr_max, gain * max(0.0, reach_up - fwd_t)), ">>"
        if z == "back":
            return -min(self.rr_max, gain * max(0.0, back_t - reach_up)), "<<"
        return 0.0, "hold"

    def __call__(self, pose: BodyPose):
        if self._tcp0 is None:
            return None
        if self.conf_gate and not pose.visible:
            return None
        c = plane_coords(pose, self.side)
        if c is None:
            return None
        reach = c["reach_up"] if c["reach_ok"] else self._reach_hold   # hold reach when the left hand isn't tracked
        if c["reach_ok"]:
            self._reach_hold = c["reach_up"]
        n = np.array([reach, c["ctrl_x"], c["ctrl_up"]], float)   # robot-axis order: x reach, y lateral, z vertical
        n = self.filter(n, pose.t)
        if self._held is None:
            self._held = n.copy()
        else:
            move = np.abs(n - self._held) >= self.deadband
            self._held = np.where(move, n, self._held)
        if self._n0 is None:
            self._n0 = self._held.copy()
            self._t_engage = pose.t
            self._t_last = pose.t
        s = 1.0 if self.ramp_s <= 0 else max(0.0, min(1.0, (pose.t - self._t_engage) / self.ramp_s))
        delta = self._held - self._n0
        p = self._tcp0 + s * self.gain * delta                    # position map on all three axes

        if self.reach_mode == "rate":
            # x (reach) is a joystick RATE, not a position: integrate a body-relative speed onto tcp0's x.
            dt = 0.0 if self._t_last is None else max(0.0, min(REACH_MAX_DT, pose.t - self._t_last))
            if c["reach_ok"]:
                speed, state = self._reach_rate_speed(self._held[0], c["hip_drop"])
            else:                                                 # reach hand not tracked -> hold (rate 0)
                self._reach_zone = "hold"
                speed, state = 0.0, "hold"
            self.reach_rate = self.reach_sign * speed * s         # the 2 s engage ramp eases the rate in too
            self.reach_state = state
            self._reach_off += self.reach_rate * dt
            p[0] = self._tcp0[0] + self._reach_off                # raw; the box clamp below pins it with anti-wind-up
        else:
            self.reach_rate, self.reach_state = 0.0, "hold"
        self._t_last = pose.t

        if self.ws_min is not None:
            if self.reach_mode == "rate":
                # reach x: clamp with ANTI-WIND-UP (pin the offset at the wall so reversing frees it at once) and
                # flag the amber ghost whenever the rate is held off by a wall, even though the per-frame overshoot
                # is tiny. y and z keep the position-map 5 mm out-of-box test.
                x_raw = self._tcp0[0] + self._reach_off
                x_cl = min(max(x_raw, self.ws_min[0]), self.ws_max[0])
                at_wall = (x_cl != x_raw
                           or (x_cl >= self.ws_max[0] - 1e-9 and self.reach_rate > 0)
                           or (x_cl <= self.ws_min[0] + 1e-9 and self.reach_rate < 0))
                self._reach_off = x_cl - self._tcp0[0]
                p[0] = x_cl
                cl = np.clip(p, self.ws_min, self.ws_max)
                yz_out = bool(max(abs(cl[1] - p[1]), abs(cl[2] - p[2])) > 0.005)
                self.clipped = bool(at_wall or yz_out)
                p = cl
            else:
                cl = np.clip(p, self.ws_min, self.ws_max)
                self.clipped = bool(np.max(np.abs(cl - p)) > 0.005)   # > 5 mm outside the box: really out of reach
                p = cl
        self._last_p = np.asarray(p, float).copy()
        return p

    # ---- live tuning (adjust / snapshot / save / load / reset) -------------------------------------------------
    def _persist_dict(self) -> dict:
        """The knobs that live-tuning changes and save()/load() persist (signed gains; magnitudes stay in range)."""
        return {"reach_max_speed": float(self.rr_max),
                "reach_gain": float(self.gain[0]), "lateral": float(self.gain[1]), "vertical": float(self.gain[2]),
                "reach_sign": float(self.reach_sign),
                "forward_frac": float(self.rr_fwd_frac), "back_frac": float(self.rr_back_frac),
                "deadband": float(self.deadband)}

    def snapshot(self) -> dict:
        """Compact read-only view for the status line / HUD (magnitudes + current reach state)."""
        return {"reach_state": self.reach_state, "reach_rate": round(float(self.reach_rate), 4),
                "max_speed": round(float(self.rr_max), 4),
                "lateral": round(abs(float(self.gain[1])), 3), "vertical": round(abs(float(self.gain[2])), 3),
                "deadband": round(float(self.deadband), 4),
                "forward_frac": round(float(self.rr_fwd_frac), 3), "back_frac": round(float(self.rr_back_frac), 3),
                "sign_lateral": _sgn(self.gain[1]), "sign_vertical": _sgn(self.gain[2]),
                "sign_reach": _sgn(self.reach_sign)}

    def _set_mag(self, idx, mag, lo, hi):
        """Set gain[idx] to a clamped MAGNITUDE, keeping its current sign (so a sensitivity change never flips)."""
        s = 1.0 if self.gain[idx] == 0 else math.copysign(1.0, self.gain[idx])
        self.gain[idx] = s * _clamp(mag, lo, hi)

    def _fix_fracs(self):
        """Clamp both reach-threshold fractions into range and keep a non-zero dead zone (forward < back)."""
        self.rr_fwd_frac = _clamp(self.rr_fwd_frac, FRAC_LO, FRAC_HI)
        self.rr_back_frac = _clamp(self.rr_back_frac, FRAC_LO, FRAC_HI)
        if self.rr_back_frac - self.rr_fwd_frac < FRAC_MIN_GAP:
            # push them apart around their midpoint, then re-clamp so both stay in [FRAC_LO, FRAC_HI]
            mid = 0.5 * (self.rr_fwd_frac + self.rr_back_frac)
            self.rr_fwd_frac = _clamp(mid - FRAC_MIN_GAP / 2, FRAC_LO, FRAC_HI - FRAC_MIN_GAP)
            self.rr_back_frac = _clamp(self.rr_fwd_frac + FRAC_MIN_GAP, FRAC_LO + FRAC_MIN_GAP, FRAC_HI)

    def _reanchor(self):
        """Re-capture the current target as the anchor so a sign flip is continuous (no jump mid-follow)."""
        if self._tcp0 is None:
            return
        if self._last_p is not None:
            self._tcp0 = np.asarray(self._last_p, float).copy()
        if self._held is not None:
            self._n0 = self._held.copy()
        self._reach_off = 0.0
        self._reach_zone = "hold"

    def adjust(self, op: str) -> str | None:
        """Apply one bounded, clamped tuning step. Returns a spoken confirmation, or None for an unknown op. Does
        NOT persist (save() does). A sign flip re-anchors so the arm never jumps."""
        if op == "reach_faster":
            self.rr_max = _clamp(self.rr_max * STEP_SPEED, RR_MAX_LO, RR_MAX_HI)
            return f"Reach faster, {self.rr_max * 100:.1f} centimetres a second."
        if op == "reach_slower":
            self.rr_max = _clamp(self.rr_max / STEP_SPEED, RR_MAX_LO, RR_MAX_HI)
            return f"Reach slower, {self.rr_max * 100:.1f} centimetres a second."
        if op == "more_sensitive":
            self._set_mag(1, abs(self.gain[1]) * STEP_GAIN, LAT_LO, LAT_HI)
            self._set_mag(2, abs(self.gain[2]) * STEP_GAIN, VRT_LO, VRT_HI)
            return f"More sensitive. Lateral {abs(self.gain[1]):.2f}, vertical {abs(self.gain[2]):.2f}."
        if op == "less_sensitive":
            self._set_mag(1, abs(self.gain[1]) / STEP_GAIN, LAT_LO, LAT_HI)
            self._set_mag(2, abs(self.gain[2]) / STEP_GAIN, VRT_LO, VRT_HI)
            return f"Less sensitive. Lateral {abs(self.gain[1]):.2f}, vertical {abs(self.gain[2]):.2f}."
        if op == "reach_fwd_sooner":        # forward begins with less raising -> lower the forward threshold
            self.rr_fwd_frac += STEP_FRAC
            self._fix_fracs()
            return f"Reach goes forward sooner (forward at {self.rr_fwd_frac:.2f})."
        if op == "reach_back_sooner":       # back begins with less lowering -> raise the back threshold
            self.rr_back_frac -= STEP_FRAC
            self._fix_fracs()
            return f"Reach goes back sooner (back at {self.rr_back_frac:.2f})."
        if op == "bigger_deadzone":
            self.deadband = _clamp(self.deadband + STEP_DB, DB_LO, DB_HI)
            return f"Bigger dead zone, {self.deadband:.3f}."
        if op == "smaller_deadzone":
            self.deadband = _clamp(self.deadband - STEP_DB, DB_LO, DB_HI)
            return f"Smaller dead zone, {self.deadband:.3f}."
        if op == "flip_lateral":
            self.gain[1] = -self.gain[1]
            self._reanchor()
            return "Flipped left and right."
        if op == "flip_vertical":
            self.gain[2] = -self.gain[2]
            self._reanchor()
            return "Flipped up and down."
        if op == "flip_reach":
            self.gain[0] = -self.gain[0]
            self.reach_sign = -self.reach_sign
            self._reanchor()
            return "Flipped reach direction."
        return None

    def set_reach_thresholds(self, forward_frac, back_frac):
        """Set the reach forward/back thresholds (used by the guided tune check), clamped with a live dead zone."""
        self.rr_fwd_frac = float(forward_frac)
        self.rr_back_frac = float(back_frac)
        self._fix_fracs()

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._persist_dict(), indent=2))

    def load(self, path) -> bool:
        """Load a saved tuning override; every field is re-clamped defensively. Returns True if a file was read."""
        path = Path(path)
        if not path.exists():
            return False
        try:
            d = json.loads(path.read_text())
        except Exception:
            return False
        if "reach_max_speed" in d:
            self.rr_max = _clamp(d["reach_max_speed"], RR_MAX_LO, RR_MAX_HI)
        if "lateral" in d:
            self.gain[1] = _sgn(d["lateral"]) * _clamp(abs(d["lateral"]), LAT_LO, LAT_HI)
        if "vertical" in d:
            self.gain[2] = _sgn(d["vertical"]) * _clamp(abs(d["vertical"]), VRT_LO, VRT_HI)
        if "reach_sign" in d:
            self.reach_sign = float(_sgn(d["reach_sign"]))
        if "reach_gain" in d:
            self.gain[0] = _sgn(d["reach_gain"]) * _clamp(abs(d["reach_gain"]), 0.01, 0.5)
        if "forward_frac" in d:
            self.rr_fwd_frac = float(d["forward_frac"])
        if "back_frac" in d:
            self.rr_back_frac = float(d["back_frac"])
        self._fix_fracs()
        if "deadband" in d:
            self.deadband = _clamp(d["deadband"], DB_LO, DB_HI)
        self._reanchor()
        return True

    def reset_tuning(self) -> None:
        """Restore the config baseline captured at construction (discards any live tweaks / loaded override)."""
        d = self._defaults
        self.rr_max = d["reach_max_speed"]
        self.gain = np.array([d["reach_gain"], d["lateral"], d["vertical"]], float)
        self.reach_sign = float(d["reach_sign"])
        self.rr_fwd_frac = d["forward_frac"]
        self.rr_back_frac = d["back_frac"]
        self.deadband = d["deadband"]
        self._reanchor()


def fit_reach_thresholds(samples: dict) -> dict:
    """Pure fit: from the operator's measured left-hand heights at shoulder and waist (reach_up, shoulder-widths)
    and their torso span (shoulder->hip, `span`), set the forward/back reach thresholds as fractions of that span.
    The dead zone sits in the middle third of the measured shoulder->waist range, so holding near shoulder reads
    FORWARD and near waist reads BACK. Clamped to [FRAC_LO, FRAC_HI] with a non-zero dead zone. Returns the chosen
    fracs plus the measured ranges (for the report)."""
    span = abs(float(samples.get("span", -1.0)))
    if span < 1e-6:
        span = 1.0
    h_f = float(samples.get("shoulder", 0.0))      # reach_up near shoulder height (~0)
    h_b = float(samples.get("waist", -1.0))        # reach_up near waist (more negative)
    f_f = max(0.0, -h_f / span)                    # fraction of the span below the shoulders, shoulder sample
    f_b = max(f_f, -h_b / span)                    # ... waist sample (never above the shoulder sample)
    fwd = f_f + (f_b - f_f) / 3.0
    back = f_f + 2.0 * (f_b - f_f) / 3.0
    fwd = _clamp(fwd, FRAC_LO, FRAC_HI)
    back = _clamp(back, FRAC_LO, FRAC_HI)
    if back - fwd < FRAC_MIN_GAP:
        back = _clamp(fwd + FRAC_MIN_GAP, FRAC_LO + FRAC_MIN_GAP, FRAC_HI)
        if back - fwd < FRAC_MIN_GAP:
            fwd = _clamp(back - FRAC_MIN_GAP, FRAC_LO, FRAC_HI - FRAC_MIN_GAP)
    rx = samples.get("right_x")
    return {"forward_frac": round(fwd, 3), "back_frac": round(back, 3),
            "measured": {"shoulder_up": round(h_f, 3), "waist_up": round(h_b, 3), "span": round(span, 3),
                         "right_x": (round(float(rx), 3) if rx is not None else None)}}


class PlaneTuneCheck:
    """Guided ~20 s reach-threshold check. Prompts the operator through three held poses (left hand to shoulder,
    left hand to waist, right hand out), samples the in-plane coordinates, and fits forward/back fracs with
    fit_reach_thresholds. Time + pose driven (tick it with plane_coords each frame); it only measures, it never
    commands the arm. The app applies the fit and persists it on 'save tuning'."""

    # (key, prompt, hold seconds, capture tag). Samples are taken in the latter part of each hold (after settling).
    STAGES = [
        ("ready", "Tune check: stand still with both hands down.", 2.5, None),
        ("shoulder", "Raise your LEFT hand to shoulder height and hold.", 4.5, "reach_shoulder"),
        ("waist", "Lower your LEFT hand to your waist and hold.", 4.5, "reach_waist"),
        ("right", "Reach your RIGHT hand out to the side and hold.", 4.5, "ctrl_right"),
    ]

    def __init__(self):
        self.active = False
        self.finished = False
        self.prompt = ""
        self.result = None
        self._said = ""
        self._stage = 0
        self._t0 = 0.0
        self._reach = []
        self._ctrl = []
        self._span = None
        self._store = {}

    def start(self, now: float):
        self.active = True
        self.finished = False
        self.result = None
        self._stage = 0
        self._t0 = now
        self._reach = []
        self._ctrl = []
        self._span = None
        self._store = {}
        self.prompt = self.STAGES[0][1]
        self._said = ""

    def cancel(self):
        self.active = False

    def update(self, now: float, pc) -> None:
        """Advance the state machine. `pc` is plane_coords(...) (or None if the pose is unusable this frame)."""
        if not self.active:
            return
        _key, prompt, dur, cap = self.STAGES[self._stage]
        self.prompt = prompt
        if pc is not None and (now - self._t0) > dur * 0.4:        # collect only after the operator has settled
            if cap in ("reach_shoulder", "reach_waist"):
                self._reach.append(pc["reach_up"])
            elif cap == "ctrl_right":
                self._ctrl.append(pc["ctrl_x"])
            if pc.get("hip_drop") is not None:
                self._span = pc["hip_drop"]
        if now - self._t0 >= dur:                                  # stage over: capture its sample, advance
            if cap in ("reach_shoulder", "reach_waist") and self._reach:
                self._store[cap] = float(np.median(self._reach))
            elif cap == "ctrl_right" and self._ctrl:
                self._store[cap] = float(np.median(self._ctrl))
            self._reach, self._ctrl = [], []
            self._stage += 1
            self._t0 = now
            if self._stage >= len(self.STAGES):
                self._finish()

    def _finish(self):
        self.active = False
        self.finished = True
        samples = {"shoulder": self._store.get("reach_shoulder", 0.0),
                   "waist": self._store.get("reach_waist", -1.0),
                   "right_x": self._store.get("ctrl_right"),
                   "span": self._span if self._span is not None else -1.0}
        self.result = fit_reach_thresholds(samples)
        self.prompt = "Tune check done."

    def pop_prompt(self) -> str | None:
        """Return the current prompt once each time it changes (so the app says it only on a stage boundary)."""
        if self.prompt != self._said:
            self._said = self.prompt
            return self.prompt
        return None

    @property
    def progress(self) -> float:
        total = sum(s[2] for s in self.STAGES)
        done = sum(self.STAGES[i][2] for i in range(min(self._stage, len(self.STAGES))))
        return min(1.0, done / total) if total > 0 else 1.0


class DropWatch:
    """Latches the 'right arm dropped' release: the control wrist stays below the hip line for `hold_s`. Reused by
    the app in plane mode (the left-hand-raise clutch is disabled there, so a lowered control arm is the stow cue)."""

    def __init__(self, hold_s: float = 0.7):
        self.hold_s = float(hold_s)
        self._t_down = None

    def update(self, below_hip: bool, now: float) -> bool:
        if not below_hip:
            self._t_down = None
            return False
        if self._t_down is None:
            self._t_down = now
        return now - self._t_down >= self.hold_s

    def reset(self):
        self._t_down = None
