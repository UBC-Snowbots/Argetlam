"""Full-arm body teleop ("arm mirror"): your whole arm's pose drives the robot's joints directly, no IK.

Why not the old position box? The first real teleop run mapped the WRIST POSITION (relative to the shoulder) into a
small workspace box and ran frozen-wrist IK. Webcam depth jitter (28 mm even while still) pushed the robot forward,
the target was pinned on a box wall 46 % of the time and below the floor 39 %, and IK near the home singularity
spun the wrist. So instead we read the operator's arm as JOINT ANGLES and send them straight to the robot joints:

    your arm                         robot joint
    swing the upper arm L/R   ->  J1 base yaw      (azimuth of the upper arm in the torso frame)
    raise / lower the arm     ->  J2 shoulder pitch (elevation of the upper arm)
    bend / straighten elbow   ->  J3 elbow pitch   (angle between upper arm and forearm)
    bend the wrist            ->  J5 wrist pitch    (forearm vs hand; needs the hand landmark, else held)
    J4 (forearm roll) and J6 (tool roll) are held at the profile's base pose.

Robustness:
- Angles come from a TORSO frame built from both shoulders + the hip midpoint, so the mapping does not depend on
  where the camera is (only the az/el reference axes rotate with the torso, not the room).
- Elevation is asin(vertical component / segment length): the vertical axis is the clean image-y axis for a camera
  either in front of or beside the operator, so J2 barely depends on depth. Azimuth is the one angle that needs the
  forward/back (depth) split; put the camera BESIDE the operator and even J1 moves in the image plane (see the doc).
- J1 is HELD whenever the upper arm is within `j1_hold_vert_deg` of vertical (azimuth is undefined when the arm hangs).
- One-Euro filter on the JOINT ANGLES (smooth when still, low lag when moving), a small deadband, and a per-joint
  rate limit (= the profile max_vel) before the SafetyLayer. On engage we ramp from the current pose to the mirrored
  pose over `ramp_s` so the arm never jumps. Confidence gate: a non-visible pose is skipped (the arm holds).

The absolute map per joint is  q = clip(home + sign * gain * (raw_angle - neutral), joint_limits).  `neutral` is the
raw arm angle that should map to the base pose; the defaults are Pranav's 2026-10-23 session medians (camera in
front). Retune `arm_mirror.neutral/gain/sign` in the arm profile (see docs/ARM_MIRROR.md).
"""
from __future__ import annotations

import math

import numpy as np

from ..types import BodyPose
from .retarget import OneEuro

# landmark-id reminder (MediaPipe pose): right = shoulder 12 / elbow 14 / wrist 16 / index 20 / pinky 18;
# left = 11 / 13 / 15 / 19 / 17. body_client already sends shoulder/elbow/wrist/shoulder_other/hips and (new) hand
# (= index/pinky world midpoint) for the chosen side, so arm_mirror never indexes landmarks by id itself.

DEFAULTS = {
    "side": "right",
    "neutral": {"az": -0.925, "el": -0.419, "elbow": 1.641, "wrist": 0.0},   # rad (-53, -24, 94, 0 deg)
    "gain": {"j1": 1.0, "j2": 1.0, "j3": 0.8, "j5": 0.8},                     # rad of robot per rad of human
    "sign": {"j1": 1.0, "j2": 1.0, "j3": 1.0, "j5": 1.0},
    "one_euro": {"min_cutoff": 1.0, "beta": 0.05},
    "deadband_deg": 1.5,
    "j1_hold_vert_deg": 20.0,
    "ramp_s": 2.0,
    "conf_gate": True,
}


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def arm_angles(pose: BodyPose, side: str = "right"):
    """Raw human arm angles in the torso frame, or None if the pose is unusable.

    Returns dict(az, el, elbow, wrist, horizfrac) in radians:
      az       azimuth of the UPPER ARM (shoulder->elbow) about the torso vertical; 0 = straight ahead,
               + toward the operator's LEFT (sign is tuned per joint in the profile).
      el       elevation of the upper arm: asin(vertical / |upper arm|); + up, - down (arm hanging ~ -pi/2).
      elbow    included angle between the upper arm and the forearm; ~pi straight... no: acos of the unit dot,
               so 0 = straight arm, grows as you bend (fully folded -> pi).
      wrist    forearm-vs-hand bend (0 = hand inline with the forearm); None when no hand landmark is sent.
      horizfrac |horizontal part of the upper arm| / |upper arm|: used to hold J1 when the arm is near vertical.
    """
    if pose.elbow is None or pose.shoulder_other is None or pose.hips is None:
        return None
    S = np.asarray(pose.shoulder, float)
    E = np.asarray(pose.elbow, float)
    W = np.asarray(pose.wrist, float)
    O = np.asarray(pose.shoulder_other, float)
    H = np.asarray(pose.hips, float)
    up = _unit((S + O) / 2.0 - H)                       # torso up (shoulder midpoint over the hips)
    medial = O - S if side == "right" else S - O        # across the chest toward the operator's LEFT
    left = _unit(medial - np.dot(medial, up) * up)
    fwd = _unit(np.cross(left, up))                     # anterior (out of the chest)
    u = E - S                                            # upper arm drives the shoulder joints
    un = float(np.linalg.norm(u))
    if un < 1e-6:
        return None
    uh = u - np.dot(u, up) * up                          # horizontal part of the upper arm
    horiz = float(np.linalg.norm(uh))
    az = math.atan2(float(np.dot(uh, left)), float(np.dot(uh, fwd)))
    el = math.asin(max(-1.0, min(1.0, float(np.dot(u, up)) / un)))
    fa = W - E
    fn = float(np.linalg.norm(fa))
    elbow = math.acos(max(-1.0, min(1.0, float(np.dot(_unit(u), _unit(fa))))))
    wrist = None
    hand = getattr(pose, "hand", None)
    if hand is not None and fn > 1e-6:
        hv = np.asarray(hand, float) - W
        if float(np.linalg.norm(hv)) > 1e-6:
            wrist = math.acos(max(-1.0, min(1.0, float(np.dot(_unit(fa), _unit(hv))))))
    return {"az": az, "el": el, "elbow": elbow, "wrist": wrist, "horizfrac": horiz / un}


def _cfg(cfg, key):
    d = dict(DEFAULTS[key]) if isinstance(DEFAULTS.get(key), dict) else DEFAULTS.get(key)
    v = (cfg or {}).get(key, d)
    if isinstance(d, dict) and isinstance(v, dict):
        out = dict(d)
        out.update(v)
        return out
    return v


class ArmMirror:
    """Maps a BodyPose to a 6-vector of robot joint targets (URDF radians). Drop-in for the app's `retarget`:
    `engage(q_now)` on clutch-engage, `pose -> q6 or None` each frame, `reset()`, and `.clipped` (a joint hit a
    limit this frame, for the amber ghost). Output already filtered, deadbanded, rate-limited and ramped."""

    def __init__(self, arm, home, max_vel, cfg: dict | None = None):
        self.arm = arm
        self.home = np.asarray(home, float).copy()
        self.lower = np.asarray(arm.lower, float)
        self.upper = np.asarray(arm.upper, float)
        self.rate = np.abs(np.asarray(max_vel, float))              # rad/s per joint (= profile max_vel)
        self.side = _cfg(cfg, "side")
        self.neutral = _cfg(cfg, "neutral")
        self.gain = _cfg(cfg, "gain")
        self.sign = _cfg(cfg, "sign")
        oe = _cfg(cfg, "one_euro")
        self.filter = OneEuro(oe.get("min_cutoff", 1.0), oe.get("beta", 0.05))
        self.deadband = math.radians(float(_cfg(cfg, "deadband_deg")))
        self.j1_hold_sin = math.sin(math.radians(float(_cfg(cfg, "j1_hold_vert_deg"))))
        self.ramp_s = float(_cfg(cfg, "ramp_s"))
        self.conf_gate = bool(_cfg(cfg, "conf_gate"))
        self.clipped = False
        self.reset()

    def reset(self):
        self._q0 = None
        self._out = None
        self._t = None
        self._t_engage = None
        self._j1 = None
        self.filter.reset()
        self.clipped = False

    def engage(self, q_now=None, *args, **kwargs):
        """Anchor the engage ramp at the current joints (q_now, a 6-vector). Extra positional/keyword args are
        accepted and ignored so the app's generic `retarget.engage(tcp, neutral[, recentre=...])` call still works."""
        q0 = self.home if q_now is None else np.asarray(q_now, float)
        q0 = q0[:6] if q0.shape == (6,) or q0.size >= 6 else self.home
        self._q0 = np.asarray(q0, float)[:6].copy()
        self._out = self._q0.copy()
        self._j1 = float(self._q0[0])
        self._t = None
        self._t_engage = None
        self.filter.reset()
        self.clipped = False

    def _raw_joints(self, feats) -> np.ndarray:
        """Absolute per-joint map (before filtering/ramp/limits). J4, J6 stay at the base pose."""
        q = self.home.copy()
        if feats["horizfrac"] >= self.j1_hold_sin:                  # arm has a usable horizontal direction
            self._j1 = self.home[0] + self.sign["j1"] * self.gain["j1"] * (feats["az"] - self.neutral["az"])
        q[0] = self._j1 if self._j1 is not None else self.home[0]   # else hold J1 (azimuth undefined near vertical)
        q[1] = self.home[1] + self.sign["j2"] * self.gain["j2"] * (feats["el"] - self.neutral["el"])
        q[2] = self.home[2] + self.sign["j3"] * self.gain["j3"] * (feats["elbow"] - self.neutral["elbow"])
        if feats["wrist"] is not None:
            q[4] = self.home[4] + self.sign["j5"] * self.gain["j5"] * (feats["wrist"] - self.neutral["wrist"])
        return q

    def __call__(self, pose: BodyPose):
        if self._q0 is None:
            return None
        if self.conf_gate and not pose.visible:
            return None
        feats = arm_angles(pose, self.side)
        if feats is None:
            return None
        q_raw = self._raw_joints(feats)
        q_clip = np.clip(q_raw, self.lower, self.upper)
        self.clipped = bool(np.max(np.abs(q_clip - q_raw)) > 1e-3)   # a joint target hit a limit (amber ghost)
        q_f = self.filter(q_clip, pose.t)                           # One-Euro on the joint angles
        if self._t_engage is None:
            self._t_engage = pose.t
        s = 1.0 if self.ramp_s <= 0 else max(0.0, min(1.0, (pose.t - self._t_engage) / self.ramp_s))
        q_t = self._q0 + s * (q_f - self._q0)                       # ramp from the engage pose (no jump)
        out = self._out.copy()
        dt = None if self._t is None else max(1e-3, pose.t - self._t)
        for i in range(6):
            d = q_t[i] - out[i]
            if s >= 1.0 and abs(d) < self.deadband:                 # deadband (only once the ramp is done)
                continue
            if dt is not None:
                lim = self.rate[i] * dt
                d = max(-lim, min(lim, d))                          # per-joint rate limit (= profile max_vel)
            out[i] += d
        out = np.clip(out, self.lower, self.upper)
        self._out = out
        self._t = pose.t
        return out.copy()
