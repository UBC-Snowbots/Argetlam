"""Safety layer between any intent source and the arm. Everything the arm executes passes through here.

- workspace box clip on the TCP target
- joint position limits (from the URDF)
- joint velocity + acceleration limits (PLACEHOLDERS until the motor datasheets are in)
- input watchdog: if the commanding source goes stale, brake at max decel to a stop and hold
- e-stop latch: once tripped, only an explicit reset() clears it
"""
from __future__ import annotations

import numpy as np


class SafetyLayer:
    def __init__(self, lower, upper, max_vel, max_acc, workspace_min, workspace_max,
                 watchdog_s: float = 0.25):
        self.lower, self.upper = np.asarray(lower, float), np.asarray(upper, float)
        self.max_vel, self.max_acc = np.asarray(max_vel, float), np.asarray(max_acc, float)
        self.ws_min, self.ws_max = np.asarray(workspace_min, float), np.asarray(workspace_max, float)
        self.watchdog_s = watchdog_s
        self.estopped = False
        self.reason = ""
        self._vel = np.zeros_like(self.lower)

    def estop(self, reason: str = "manual"):
        self.estopped, self.reason = True, reason
        self._vel[:] = 0

    def reset(self):
        self.estopped, self.reason = False, ""
        self._vel[:] = 0

    def clip_workspace(self, p: np.ndarray) -> np.ndarray:
        return np.clip(p, self.ws_min, self.ws_max)

    def step(self, q_now: np.ndarray, q_des: np.ndarray, dt: float, input_age_s: float) -> np.ndarray:
        """Return the next joint command given the current joints and the desired joints."""
        if self.estopped:
            return q_now.copy()
        dt_acc = self.max_acc * dt
        if input_age_s > self.watchdog_s:
            v_want = np.zeros_like(self._vel)  # stale input: brake at max decel and hold
        else:
            q_des = np.clip(q_des, self.lower, self.upper)
            d = q_des - q_now
            v_want = np.clip(d / dt, -self.max_vel, self.max_vel)
            # discrete braking curve: largest speed from which we can still stop within |d|
            d_safe = np.maximum(np.abs(d) - self.max_acc * dt * dt, 0.0)  # one-step margin for discretisation
            v_brake = self.max_acc * (-dt / 2 + np.sqrt(dt * dt / 4 + 2 * d_safe / self.max_acc))
            v_want = np.clip(v_want, -v_brake, v_brake)
        self._vel = self._vel + np.clip(v_want - self._vel, -dt_acc, dt_acc)
        q_out = np.clip(q_now + self._vel * dt, self.lower, self.upper)
        self._vel = (q_out - q_now) / dt  # stay consistent if a limit clipped the step
        return q_out
