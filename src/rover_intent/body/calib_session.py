"""Calibration session (positioning -> capture -> fit -> touch-the-target verification -> save) and the mapper
that uses the resulting profile for teleop. The maths lives in body/calibration.py."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .calibration import DIRS, Capture, Profile, command_frame, fit, hand_in_body
from .retarget import OneEuro

log = logging.getLogger("rover_intent")


class CalibratedMapper:
    """Teleop mapping from a calibration Profile. Clutch-in = quick re-centre: your pose at the end of the clutch's
    hold-still becomes neutral (maps to the box centre); the arm then GLIDES there at `glide_mps` before following."""

    def __init__(self, profile: Profile, lo, hi, camera_provider, glide_mps=0.12, min_cutoff=1.0, beta=0.7):
        self.profile, self.lo, self.hi = profile, np.asarray(lo, float), np.asarray(hi, float)
        self.camera = camera_provider          # () -> {"eye", "target"} of the current Isaac view
        self.glide_mps = glide_mps
        self.filter = OneEuro(min_cutoff, beta)
        self.clipped = False
        self._recentre = False
        self._carrot = None
        self._gliding = False
        self._t = None

    def frame(self):
        cam = self.camera() or self.profile.camera
        return command_frame(cam["eye"], cam["target"])

    def engage(self, tcp_now, neutral=None, recentre=True):
        """recentre=False (after a grab assist): keep neutral, just glide from where the arm is to your hand's point."""
        self._recentre, self._carrot, self._gliding, self._t = recentre, np.asarray(tcp_now, float).copy(), True, None
        self.filter.reset()

    def reset(self):
        self._recentre, self._carrot, self._gliding, self._t = False, None, False, None
        self.filter.reset()

    def __call__(self, pose):
        h = hand_in_body(pose)
        if h is None:
            return None
        if self._recentre:
            self.profile.neutral = h.tolist()   # quick re-centre (R and scales unchanged)
            self._recentre = False
        target, self.clipped = self.profile.map(h, self.frame(), self.lo, self.hi)
        target = self.filter(target, pose.t)
        if self._gliding and self._carrot is not None:
            dt = 0.0 if self._t is None else max(0.0, min(0.1, pose.t - self._t))
            d = target - self._carrot
            n = np.linalg.norm(d)
            step = self.glide_mps * dt
            if n <= max(step, 0.01):
                self._gliding = False
            else:
                self._carrot = self._carrot + d / n * step
                target = self._carrot
        self._t = pose.t
        return target


class CalibrationSession:
    """positioning -> capturing -> verifying -> done. Driven by the app with every body sample."""

    TARGET_TOL, TARGET_HOLD, TARGET_TIMEOUT = 0.03, 1.0, 20.0

    def __init__(self, lo, hi, profile_path, arm_profile, default_camera, steps=None):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.path, self.arm_profile, self.default_camera = Path(profile_path), arm_profile, default_camera
        self.steps = steps
        self.state, self.note, self.result = "idle", "", None
        self.capture, self.profile = None, None
        self._t_inpos = None
        self.targets, self.ti, self.errors = [], 0, []
        self._t_target = self._t_near = None
        self._near = []

    # ------------------------------------------------------------------ control
    AXIS_STEPS = {"right/left": ["right", "left"], "forward/back": ["forward"], "up/down": ["up", "down"]}

    def start(self, now, redo_axis=None):
        """redo_axis: re-capture only that axis (plus neutral) and keep the other holds from the last capture."""
        keep = {}
        if redo_axis and self.capture is not None and self.capture.holds:
            keep = {k: v for k, v in self.capture.holds.items()
                    if k != "neutral" and k not in self.AXIS_STEPS.get(redo_axis, [])}
            steps = ["neutral"] + self.AXIS_STEPS[redo_axis]
        else:
            steps = self.steps
        self.state, self.note, self.result, self._t_inpos = "positioning", "", None, None
        self.capture = Capture(**({"steps": steps} if steps else {}))
        self._keep = keep
        return self._say("Stand on the guide in the webcam window: it turns green when you're in position.")

    def cancel(self):
        self.state = "idle"

    @property
    def active(self):
        return self.state in ("positioning", "capturing", "verifying")

    def vtarget(self):
        return self.targets[self.ti].tolist() if self.state == "verifying" and self.ti < len(self.targets) else None

    def view(self):
        if self.state == "idle" and not self.result:
            return None
        v = {"state": self.state, "note": self.note, "result": self.result}
        if self.state == "capturing":
            v.update(step=self.capture.step, prompt=self.capture.prompt, progress=round(self.capture.progress, 2),
                     n=self.capture.i + 1, of=len(self.capture.steps))
        elif self.state == "positioning":
            v.update(prompt="Get into position (webcam window guide turns green)")
        elif self.state == "verifying":
            v.update(prompt=f"Touch the MAGENTA target with the gripper ({self.ti + 1}/{len(self.targets)})",
                     n=self.ti + 1, of=len(self.targets), errors=[round(100 * e, 1) for e in self.errors])
        return v

    # ------------------------------------------------------------------ per sample
    def update(self, now, pose, in_position, tcp_meas=None, camera=None):
        """Returns a dict of side effects for the app: {"say": str, "engage": bool, "release": bool, "profile": Profile}"""
        out = {}
        if self.state == "positioning":
            if in_position:
                self._t_inpos = now if self._t_inpos is None else self._t_inpos
                if now - self._t_inpos >= 1.0:
                    self.state = "capturing"
                    out.update(self._say(self.capture.prompt), release=True)
            else:
                self._t_inpos = None
        elif self.state == "capturing":
            if not in_position:
                self.note = "step back onto the guide - capture paused"
                return out
            prev = self.capture.step
            ev = self.capture.update(now, hand_in_body(pose) if pose is not None else None)
            self.note = self.capture.note
            if ev and ev.startswith("held:"):
                out.update(self._say("Good. " + self.capture.prompt))
            elif ev and ev.startswith("timeout:"):
                out.update(self._say(f"Skipped {prev} (no clear hold). " + self.capture.prompt))
            elif ev == "done":
                out.update(self._finish_capture(camera))
        elif self.state == "verifying":
            out.update(self._verify_step(now, tcp_meas))
        return out

    def _finish_capture(self, camera):
        holds = {**getattr(self, "_keep", {}), **self.capture.holds}
        self.capture.holds = holds
        need = {"neutral", "right", "up", "forward"}
        if not need <= set(holds):
            self.state, self.result = "failed", {"accepted": False, "why": f"missing holds: {sorted(need - set(holds))}"}
            return self._say("Calibration failed: some directions weren't captured. Press C to try again.")
        cam = camera or self.default_camera
        self.profile = fit(holds, command_frame(cam["eye"], cam["target"]), self.lo, self.hi,
                           arm_profile=self.arm_profile, camera=cam)
        log.info("calibration fit: residual %.1f deg, per direction %s, reach %s, scale %s",
                 self.profile.residual_deg, {k: round(v, 1) for k, v in self.profile.per_dir_deg.items()},
                 {k: round(v, 3) for k, v in self.profile.reach.items()},
                 {k: round(v, 2) for k, v in self.profile.scale.items()})
        C = command_frame(cam["eye"], cam["target"])
        c = (self.lo + self.hi) / 2
        half = (self.hi - self.lo) / 2 - 0.02
        self.targets = []
        for r, f, u in [(1, 1, 0.3), (-1, 1, -0.3), (1, -1, -0.3), (-1, -1, 0.3)]:   # 4 quadrants, mixed heights
            p = c + C @ np.array([0.6 * r, 0.6 * f, u]) * np.array([half.max(), half.max(), half[2]])
            self.targets.append(np.clip(p, self.lo + 0.04, self.hi - 0.04))  # >= 4 cm inside: no clamping near targets
        self.ti, self.errors, self._t_target, self._t_near, self._near = 0, [], None, None, []
        self.state = "verifying"
        return {**self._say(f"Fit done ({self.profile.residual_deg:.1f} degrees off). Now touch each magenta target "
                            f"with the gripper and hold it there."), "profile": self.profile, "engage": True}

    def _verify_step(self, now, tcp):
        if tcp is None:
            return {}
        self._t_target = now if self._t_target is None else self._t_target
        tgt = self.targets[self.ti]
        d = float(np.linalg.norm(np.asarray(tcp) - tgt))
        if d < self.TARGET_TOL:
            self._t_near = now if self._t_near is None else self._t_near
            self._near.append(d)
            done = now - self._t_near >= self.TARGET_HOLD
        else:
            self._t_near, self._near = None, []
            done = False
        timeout = now - self._t_target > self.TARGET_TIMEOUT
        if not (done or timeout):
            self.note = f"{100 * d:.0f} cm away"
            return {}
        err = float(np.mean(self._near)) if done else d
        self.errors.append(err)
        self.err_vecs = getattr(self, "err_vecs", []) + [np.asarray(tcp) - tgt]
        self.ti += 1
        self._t_target, self._t_near, self._near = None, None, []
        if self.ti < len(self.targets):
            return self._say(f"Target {self.ti}: {100 * err:.1f} cm. Next target.")
        return self._finish_verify()

    def _finish_verify(self):
        ok = all(e < self.TARGET_TOL for e in self.errors)
        bad_axis = None
        if not ok:
            C = command_frame(self.profile.camera["eye"], self.profile.camera["target"])
            comp = np.mean([np.abs(C.T @ v) for v, e in zip(self.err_vecs, self.errors) if e >= self.TARGET_TOL], 0)
            bad_axis = ["right/left", "forward/back", "up/down"][int(np.argmax(comp))]
        self.profile.verify = {"errors_cm": [round(100 * e, 1) for e in self.errors], "accepted": ok,
                               "bad_axis": bad_axis}
        self.profile.save(self.path)
        log.info("calibration verify: %s -> saved %s", self.profile.verify, self.path)
        self.state = "done"
        self.result = {"accepted": ok, "errors_cm": self.profile.verify["errors_cm"],
                       "residual_deg": round(self.profile.residual_deg, 1), "bad_axis": bad_axis}
        msg = (f"Calibration accepted: all targets within 3 cm ({', '.join(f'{100 * e:.1f}' for e in self.errors)} cm)."
               if ok else f"Not accepted: worst axis is {bad_axis}. Press X to redo just that axis (C = everything).")
        return {**self._say(msg), "profile": self.profile}

    def _say(self, text):
        return {"say": text}
