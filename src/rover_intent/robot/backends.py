"""Arm backends. The control loop only talks to this interface.

- MockArm:   perfect position tracking in-process. Works today; used by tests and the dry-run loop.
- IsaacArm:  sends joint targets over UDP to sim/isaac_bridge.py running in ~/isaacenv6. (TODO)
- RealArm:   the team's old arm. Control path unknown until the arm is revived; see docs/REAL_ARM.md.
"""
from __future__ import annotations

import abc
import json
import socket

import numpy as np


class ArmBackend(abc.ABC):
    @abc.abstractmethod
    def read_joints(self) -> np.ndarray: ...

    @abc.abstractmethod
    def command(self, q: np.ndarray, gripper: float) -> None: ...

    def close(self) -> None:
        pass


class MockArm(ArmBackend):
    def __init__(self, q0):
        self.q = np.array(q0, float)
        self.gripper = 0.0

    def read_joints(self):
        return self.q.copy()

    def command(self, q, gripper):
        self.q = np.array(q, float)
        self.gripper = float(gripper)


class IsaacArm(ArmBackend):
    """Joint targets -> Isaac over UDP; state <- Isaac.
    cmd:   {"q": [6], "g": 0..1}
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
        self.connected = False

    def read_joints(self):
        while True:
            try:
                msg = json.loads(self.rx.recv(65536))
            except BlockingIOError:
                return self.q.copy()
            self.q = np.array(msg["q"], float)
            self.objects = msg.get("objects", self.objects)
            self.connected = True

    def command(self, q, gripper):
        self.tx.sendto(json.dumps({"q": list(map(float, q)), "g": float(gripper)}).encode(), self.addr)


class RealArm(ArmBackend):
    def __init__(self, **kw):
        raise NotImplementedError("Real arm control path not known yet - see docs/REAL_ARM.md")

    def read_joints(self):
        raise NotImplementedError

    def command(self, q, gripper):
        raise NotImplementedError
