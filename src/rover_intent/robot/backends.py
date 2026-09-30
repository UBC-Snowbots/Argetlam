"""Arm backends. The control loop only talks to this interface.

- MockArm:   perfect position tracking in-process. Works today; used by tests and the dry-run loop.
- IsaacArm:  sends joint targets over UDP to sim/isaac_bridge.py running in ~/projects/ubc-rover-arm/isaac/isaacenv6. (TODO)
- RealArm:   the team's old arm. Control path unknown until the arm is revived; see docs/REAL_ARM.md.
"""
from __future__ import annotations

import abc
import json
import socket
import time

import numpy as np


class ArmBackend(abc.ABC):
    @abc.abstractmethod
    def read_joints(self) -> np.ndarray: ...

    @abc.abstractmethod
    def command(self, q: np.ndarray, gripper: float, ghost=None, ghost_bad: bool = False,
                ghost_state: str | None = None, vtarget=None) -> None: ...

    def reset(self, q_home) -> None:
        """Put the arm at q_home with the gripper open (sim: also restore the objects)."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class MockArm(ArmBackend):
    def __init__(self, q0):
        self.q = np.array(q0, float)
        self.gripper = 0.0

    def read_joints(self):
        return self.q.copy()

    def command(self, q, gripper, ghost=None, ghost_bad=False, ghost_state=None, vtarget=None):
        self.q = np.array(q, float)
        self.gripper = float(gripper)

    def reset(self, q_home):
        self.q, self.gripper = np.array(q_home, float), 0.0


class IsaacArm(ArmBackend):
    """Joint targets -> Isaac over UDP; state <- Isaac.
    cmd:   {"q": [6], "g": 0..1, "ghost": [x,y,z]?, "ghost_bad": bool?}   or   {"reset": true}   or   {"view": name}
    state: {"q": [6], "objects": {name: [x, y, z_bottom]}}   (objects: sim ground truth, stands in for perception)
    """

    def __init__(self, host: str, cmd_port: int, state_port: int, q0):
        self.addr = (host, cmd_port)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("0.0.0.0", state_port))
        self.rx.setblocking(False)
        self.q = np.array(q0, float)
        self.objects: dict[str, list[float]] = {}
        self.camera: dict | None = None      # Isaac viewport camera {"eye", "target"} (for the calibration frame)
        self.connected = False
        self._ignore_until = 0.0

    def read_joints(self):
        while True:
            try:
                msg = json.loads(self.rx.recv(65536))
            except BlockingIOError:
                return self.q.copy()
            if time.time() < self._ignore_until:
                continue  # pre-reset states still in flight: drop them
            self.q = np.array(msg["q"], float)
            self.objects = msg.get("objects", self.objects)
            if msg.get("cam", {}).get("eye"):
                self.camera = msg["cam"]
            self.connected = True

    def command(self, q, gripper, ghost=None, ghost_bad=False, ghost_state=None, vtarget=None):
        msg = {"q": list(map(float, q)), "g": float(gripper), "vtarget": None if vtarget is None else
               list(map(float, vtarget))}
        if ghost is not None:
            msg["ghost"], msg["ghost_bad"] = list(map(float, ghost)), bool(ghost_bad)
            msg["ghost_state"] = ghost_state or ("clamped" if ghost_bad else "ok")
        self.tx.sendto(json.dumps(msg).encode(), self.addr)

    def reset(self, q_home):
        """The bridge teleports the arm home and restores the objects; then we adopt home as the state."""
        self.tx.sendto(json.dumps({"reset": True}).encode(), self.addr)
        self.q = np.array(q_home, float)
        self._ignore_until = time.time() + 0.3  # the bridge resets within one physics step (~8 ms)


class RealArm(ArmBackend):
    def __init__(self, **kw):
        raise NotImplementedError("Real arm control path not known yet - see docs/REAL_ARM.md")

    def read_joints(self):
        raise NotImplementedError

    def command(self, q, gripper, ghost=None, ghost_bad=False, ghost_state=None, vtarget=None):
        raise NotImplementedError
