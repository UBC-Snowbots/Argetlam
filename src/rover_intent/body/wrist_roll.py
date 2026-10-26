"""Optional WRIST-ROLL follow for PLANE body teleop.

When enabled, turning the operator's CONTROL hand (the lateral/vertical hand) about the forearm
(pronation / supination, i.e. the palm turning) rotates the gripper about its own axis = the last
joint (joint 6, the wrist roll) of the dev_arm / dev_arm_real profiles. Default OFF.

Signal: the roll angle is read from the control hand's 21 MediaPipe Hands landmarks as the angle of
the palm ACROSS-vector (index MCP -> pinky MCP) in the IMAGE plane (x, y only; the hand z is never
read, like the rest of plane teleop). The angle at clutch engage / toggle-on is captured as 0, so the
operator's natural pose maps to the current joint-6 value and turning the follow on never jumps.

`palm_roll_deg` is a PURE function of the landmarks (testable without a camera). `WristRoll` adds the
anchor, a One-Euro smoother (continuous-unwrapped, so +-180 wrap never spikes it), a deadband, a gain
(hand degrees -> joint degrees, default 1.0) and a clamp to the joint-6 limits with a margin. A
missing / low-confidence hand holds the current roll.

Why joint 6 can be set straight after the IK: on this arm joint 6's axis runs through the TCP (the
tool offset is purely along that axis), so rolling joint 6 spins the tool about its own axis WITHOUT
moving the TCP position. The frozen-wrist IK solves the position on joints 1-3 and holds 4-6, so the
app overrides joint 6 with this target and the TCP position is unchanged (verified by FK).
"""
from __future__ import annotations

import math

import numpy as np

from .retarget import OneEuro

# MediaPipe Hands landmark indices (21 points).
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17

# the across-vector must be at least this fraction of the palm scale to be a trustworthy roll read
MIN_SPAN_FRAC = 0.2


def _wrap180(d: float) -> float:
    """Wrap a degree difference into (-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


def palm_roll_deg(landmarks) -> float | None:
    """Roll angle (degrees) of the palm across-vector (index MCP -> pinky MCP) in the image plane, or None.

    `landmarks` is a sequence of 21 points, each (x, y) or (x, y, z) in normalized image coords
    (y points DOWN), or None. Missing / too-short / degenerate input returns None (the caller holds).
    The angle is atan2(dy, dx) of the vector from the index MCP to the pinky MCP; as the hand pronates
    or supinates that vector sweeps in the image plane, so the angle tracks the palm turning. Pure: no
    state, no camera, x and y only."""
    if landmarks is None or len(landmarks) < 21:
        return None
    P = [(float(p[0]), float(p[1])) for p in landmarks]
    scale = math.hypot(P[WRIST][0] - P[MIDDLE_MCP][0], P[WRIST][1] - P[MIDDLE_MCP][1])
    if scale < 1e-6:
        return None
    dx = P[PINKY_MCP][0] - P[INDEX_MCP][0]
    dy = P[PINKY_MCP][1] - P[INDEX_MCP][1]
    if math.hypot(dx, dy) < MIN_SPAN_FRAC * scale:   # fingers edge-on / degenerate: not a trustworthy roll
        return None
    return math.degrees(math.atan2(dy, dx))


class WristRoll:
    """Anchor + smooth + deadband + gain + clamp from a palm roll angle to an absolute joint-6 target (radians).

    engage(joint6): capture the current joint-6 value as the zero (and re-anchor the hand on the next sample).
    update(landmarks, now): return the joint-6 target. A missing / low-confidence hand holds the last target.
    Disabled -> always returns the engage value (hold)."""

    def __init__(self, enabled: bool = False, gain: float = 1.0, deadband_deg: float = 10.0,
                 lo: float = -math.pi, hi: float = math.pi, margin_deg: float = 5.0, sign: float = 1.0,
                 min_cutoff: float = 1.0, beta: float = 0.05):
        self.enabled = bool(enabled)
        self.gain = float(gain)
        self.deadband_deg = max(0.0, float(deadband_deg))
        self.sign = 1.0 if float(sign) >= 0 else -1.0
        margin = math.radians(max(0.0, float(margin_deg)))
        lo, hi = float(lo), float(hi)
        self.lo, self.hi = (lo + margin, hi - margin) if hi - margin > lo + margin else (lo, hi)
        self.filter = OneEuro(min_cutoff, beta)
        self.roll0 = 0.0          # joint-6 value captured at engage (radians)
        self.target = 0.0         # last emitted joint-6 target (radians)
        self.roll_deg = 0.0       # applied roll offset from the anchor, in JOINT degrees (for the HUD)
        self._hand0 = None        # filtered hand angle at engage (degrees), captured on the first sample
        self._unwrapped = None    # continuous (unwrapped) hand angle (degrees)
        self._prev_raw = None     # previous raw hand angle (degrees), for the unwrap
        self.reset()

    def _clamp(self, v: float) -> float:
        return min(max(float(v), self.lo), self.hi)

    def reset(self):
        self.roll0 = self._clamp(0.0)
        self.target = self.roll0
        self.roll_deg = 0.0
        self._hand0 = None
        self._unwrapped = None
        self._prev_raw = None
        self.filter.reset()

    def engage(self, joint6_rad: float):
        """Anchor at the current joint-6 value; the hand angle is captured on the first update after this."""
        self.roll0 = self._clamp(joint6_rad)
        self.target = self.roll0
        self.roll_deg = 0.0
        self._hand0 = None
        self._unwrapped = None
        self._prev_raw = None
        self.filter.reset()

    def set_enabled(self, on: bool, joint6_rad: float | None = None):
        """Enable / disable the follow. Turning it ON re-anchors (so it never jumps) when a joint-6 value is given."""
        self.enabled = bool(on)
        if self.enabled and joint6_rad is not None:
            self.engage(joint6_rad)

    def update(self, landmarks, now: float) -> float:
        """One frame: return the joint-6 target (radians). Holds on a missing / low-confidence hand or when disabled."""
        if not self.enabled:
            return self.target
        raw = palm_roll_deg(landmarks)
        if raw is None:                                   # no trustworthy hand this frame: hold
            return self.target
        if self._prev_raw is None:
            self._unwrapped = raw
        else:
            self._unwrapped += _wrap180(raw - self._prev_raw)   # continuous angle, so +-180 wrap never spikes
        self._prev_raw = raw
        filt = float(self.filter(np.array([self._unwrapped], float), now)[0])
        if self._hand0 is None:                           # first sample = the zero (natural pose -> current joint 6)
            self._hand0 = filt
            return self.target
        delta = filt - self._hand0                        # hand roll since engage, degrees
        if abs(delta) <= self.deadband_deg:
            eff = 0.0
        else:
            eff = delta - math.copysign(self.deadband_deg, delta)   # continuous past the dead zone
        self.roll_deg = self.sign * self.gain * eff
        self.target = self._clamp(self.roll0 + math.radians(self.roll_deg))
        return self.target

    def snapshot(self) -> dict:
        """Compact read-only view for the HUD / body client."""
        return {"enabled": bool(self.enabled), "roll_deg": round(float(self.roll_deg), 1),
                "target_deg": round(math.degrees(self.target), 1), "gain": round(float(self.gain), 2),
                "deadband_deg": round(float(self.deadband_deg), 1), "sign": int(self.sign)}
