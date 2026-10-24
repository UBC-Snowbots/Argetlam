"""End-effector (EE / gripper, firmware axis 7 'ee_prismatic') limits + mapping for the real dev_arm.

The EE is a lead-screw LINEAR axis. It is driven through Rowan's stepper_arm_driver as the
rover_msgs/ArmCommand.end_effector velocity (firmware deg/s) in a cmd_type 86 'V' command; feedback is
/arm/feedback positions[6] (firmware degrees). TWO things make its numbers tricky:

  * The firmware converts EE steps with AXIS 6's gear ratio (a known bug), so the reported EE "degrees" are a
    mis-scaled, arbitrary unit. ONLY RELATIVE positions are meaningful; treat the unit as "fw deg" but never as a
    real angle or length.
  * There is NO limit switch, and home-all (cmd_value 170 = 0xAA) skips the EE, so its zero is ARBITRARY after
    each power-on. Every limit here is RELATIVE TO THIS POWER-ON and must be re-marked or re-referenced each
    session (scripts/bench/ee_limits.py).

One end has a mechanical hard stop; the other can squeeze/stall. BOTH ends are dangerous until marked.

The app's gripper value `grip` is 0 = open .. 1 = closed. We map:
    ee_fw_target = open_fw + grip * (closed_fw - open_fw)      (grip 0 -> open_fw, grip 1 -> closed_fw)
clamped to the SOFT LIMITS = the [open, closed] travel contracted by `margin_fw` at each end (so a full-open or
full-close command still stops `margin_fw` short of the marked stop). open_fw and closed_fw may be in either
numeric order (which firmware direction opens depends on the lead-screw wiring -> open_sign), so every comparison
here sorts the two ends rather than assuming open < closed.

Pure stdlib (math + dataclasses + a tiny flat-YAML reader/writer) so it imports on the system python the ROS
nodes use (the sim_to_real follower, real_to_sim) as well as inside the venv app. No numpy, no PyYAML.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

# Repo root = .../rover-intent (this file is src/rover_intent/control/ee_limits.py)
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIMITS_PATH = ROOT / "configs" / "arms" / "dev_arm_real_ee.yaml"

# EE speeds are in the mis-scaled firmware "deg/s". Start low: the EE max speed is unknown and both ends bite.
EE_DEFAULT_SPEED_DEG_S = 10.0    # follower / bench default EE speed
EE_SPEED_CAP_DEG_S = 30.0        # hard cap: no EE command ever exceeds this |deg/s|, whatever the argument
EE_DEFAULT_MARGIN_FW = 5.0       # fw deg kept clear of each marked stop
EE_DEFAULT_KP = 3.0              # 1/s, P-loop gain (same shape as the joint follower)
EE_DEFAULT_DEADBAND = 0.5        # fw deg
REFERENCE_FACTOR = 2.0           # refuse EE control if feedback is outside [min-2*margin, max+2*margin]


def cap_ee_speed(deg_s: float) -> float:
    """Clamp a requested EE speed to [0, EE_SPEED_CAP_DEG_S]; a wrong argument can never exceed the cap."""
    return min(max(float(deg_s), 0.0), EE_SPEED_CAP_DEG_S)


@dataclass(frozen=True)
class EELimits:
    """Marked EE limits for ONE power-on of the real dev_arm (see module docstring)."""
    open_fw: float          # EE firmware deg at the marked fully-OPEN position
    closed_fw: float        # EE firmware deg at the marked fully-CLOSED position
    open_sign: int = 1      # which fw-velocity sign OPENS the fingers (+1 or -1); recorded at the bench
    margin_fw: float = EE_DEFAULT_MARGIN_FW
    marked_at: str = ""     # when/how these were marked; a reminder the reference is per-power-on
    notes: str = "relative to this power-on; re-mark or re-reference each session"

    # ---- geometry ----------------------------------------------------------------------------------
    @property
    def span(self) -> float:
        """Signed travel from open to closed (closed_fw - open_fw)."""
        return self.closed_fw - self.open_fw

    def soft_limits(self) -> tuple[float, float]:
        """(lo, hi) firmware-deg interval the EE may be commanded into: the [open, closed] travel contracted by
        margin_fw at each end, ALWAYS returned lo <= hi. If the margin would cross, both collapse to the midpoint."""
        lo, hi = sorted((self.open_fw, self.closed_fw))
        lo += self.margin_fw
        hi -= self.margin_fw
        if lo > hi:
            mid = 0.5 * (self.open_fw + self.closed_fw)
            return mid, mid
        return lo, hi

    def clamp(self, fw: float) -> float:
        lo, hi = self.soft_limits()
        return min(max(float(fw), lo), hi)

    def grip_to_fw(self, grip: float) -> float:
        """app grip (0 open .. 1 closed) -> EE firmware-deg target, clamped into the soft limits."""
        raw = self.open_fw + float(grip) * self.span
        return self.clamp(raw)

    def fw_to_grip(self, fw: float) -> float:
        """EE firmware deg -> app grip (0 open .. 1 closed), clamped to [0, 1]. Inverse of grip_to_fw (the mirror)."""
        if self.span == 0:
            return 0.0
        g = (float(fw) - self.open_fw) / self.span
        return min(max(g, 0.0), 1.0)

    def reference_ok(self, cur_ee_fw: float, factor: float = REFERENCE_FACTOR) -> bool:
        """Is the current EE feedback inside [min(open,closed) - factor*margin, max + factor*margin]? Outside means
        the per-power-on reference is stale (a fresh power-on, or the EE crept): refuse EE control and re-reference."""
        lo, hi = sorted((self.open_fw, self.closed_fw))
        return (lo - factor * self.margin_fw) <= float(cur_ee_fw) <= (hi + factor * self.margin_fw)

    def shifted_to_open(self, measured_open_fw: float, marked_at: str = "") -> "EELimits":
        """Re-reference: move the OPEN mark to measured_open_fw, keeping the span (so closed follows). open_sign,
        margin and notes are kept. Used when a later session jogs to the open position by eye and presses m."""
        span = self.span
        return EELimits(open_fw=float(measured_open_fw), closed_fw=float(measured_open_fw) + span,
                        open_sign=self.open_sign, margin_fw=self.margin_fw,
                        marked_at=marked_at or self.marked_at, notes=self.notes)

    # ---- serialization -----------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"open_fw": float(self.open_fw), "closed_fw": float(self.closed_fw),
                "open_sign": int(self.open_sign), "margin_fw": float(self.margin_fw),
                "marked_at": str(self.marked_at), "notes": str(self.notes)}

    @classmethod
    def from_dict(cls, d: dict) -> "EELimits":
        return cls(open_fw=float(d["open_fw"]), closed_fw=float(d["closed_fw"]),
                   open_sign=int(d.get("open_sign", 1)), margin_fw=float(d.get("margin_fw", EE_DEFAULT_MARGIN_FW)),
                   marked_at=str(d.get("marked_at", "")), notes=str(d.get("notes", "")))


# ---- EE velocity command (the single gate point) ---------------------------------------------------

def ee_velocity(grip: float, cur_ee_fw: float, limits: EELimits, cap_deg_s: float,
                kp: float = EE_DEFAULT_KP, deadband: float = EE_DEFAULT_DEADBAND) -> float:
    """One capped P-loop EE velocity (fw deg/s) toward grip's soft-clamped target; 0 inside the deadband.
    Never drives further OUTSIDE the soft limits, so a stale/odd feedback value cannot push into a stop."""
    cap = cap_ee_speed(cap_deg_s)
    target = limits.grip_to_fw(grip)
    e = target - float(cur_ee_fw)
    if abs(e) <= deadband:
        return 0.0
    v = max(-cap, min(cap, kp * e))
    lo, hi = limits.soft_limits()
    if (cur_ee_fw <= lo and v < 0) or (cur_ee_fw >= hi and v > 0):
        return 0.0
    return v


def ee_command(ee_on: bool, grip, cur_ee_fw, limits, cap_deg_s: float,
               kp: float = EE_DEFAULT_KP, deadband: float = EE_DEFAULT_DEADBAND) -> float:
    """The EE velocity the follower should publish in ArmCommand.end_effector. Returns 0.0 whenever EE control is
    gated OFF, there is no limits file, or there is no EE feedback yet -- so end_effector stays 0 unless everything
    required is present. This is the one place the gate is enforced."""
    if not ee_on or limits is None or cur_ee_fw is None:
        return 0.0
    return ee_velocity(grip, cur_ee_fw, limits, cap_deg_s, kp, deadband)


def ee_enabled(ee_flag: bool, path=DEFAULT_LIMITS_PATH, config_ee: bool = False) -> bool:
    """EE control is active only when the limits file EXISTS and (the --ee flag is set OR real_arm.ee is true)."""
    return exists(path) and (bool(ee_flag) or bool(config_ee))


# ---- flat-YAML persistence (stdlib only; the file is also valid YAML for yaml.safe_load) -----------

def exists(path=DEFAULT_LIMITS_PATH) -> bool:
    return os.path.exists(str(path))


def _parse_scalar(s: str):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def load(path=DEFAULT_LIMITS_PATH) -> EELimits:
    """Read a limits file written by save() (a flat key: value YAML)."""
    d: dict = {}
    with open(str(path)) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, _, val = line.partition(":")
            d[key.strip()] = _parse_scalar(val)
    return EELimits.from_dict(d)


def save(limits: EELimits, path=DEFAULT_LIMITS_PATH) -> str:
    """Write a limits file (flat key: value YAML with a reminder header). Returns the path written."""
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d = limits.to_dict()
    lo, hi = limits.soft_limits()
    header = (
        "# dev_arm_real EE (gripper, firmware axis 7 'ee_prismatic') limits.\n"
        "# RELATIVE to this power-on (no limit switch; home-all skips the EE; firmware mis-scales EE steps with\n"
        "# axis 6's ratio) -- re-mark or re-reference each session with scripts/bench/ee_limits.py.\n"
        "# grip 0 = open (open_fw), grip 1 = closed (closed_fw); open_sign = fw-velocity sign that OPENS.\n"
        f"# soft limits (fw deg, what the follower may command): [{lo:.3f}, {hi:.3f}]\n"
    )

    def _fmt(k, v):
        if isinstance(v, str):
            return f'{k}: "{v}"'
        if isinstance(v, int):
            return f"{k}: {v}"
        return f"{k}: {v:.6g}"

    body = "\n".join(_fmt(k, d[k]) for k in ("open_fw", "closed_fw", "open_sign", "margin_fw", "marked_at", "notes"))
    with open(path, "w") as f:
        f.write(header + body + "\n")
    return path


def now_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
