"""URDF forward kinematics + damped-least-squares IK for the rover2026 arm (numpy only).

The TCP is a fixed offset from `a6_EE_holder`'s origin. The tool face points along +x in that frame
(see sim/make_ee_urdf.py), so a gripper's TCP is (tcp_offset, 0, 0) until the real claw is measured.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


def rpy_to_mat(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_angle(axis: np.ndarray, q: float) -> np.ndarray:
    x, y, z = axis
    c, s, C = np.cos(q), np.sin(q), 1 - np.cos(q)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def rot_err(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Orientation error as a rotation vector (world frame) taking R_cur to R_des."""
    R = R_des @ R_cur.T
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - angle < 1e-6:  # near 180 deg: pick the axis from the diagonal
        ax = np.sqrt(np.clip((np.diag(R) + 1) / 2, 0, None))
        return angle * ax
    return angle * v / (2 * np.sin(angle))


@dataclass
class Joint:
    name: str
    origin: np.ndarray  # 4x4 parent->joint
    axis: np.ndarray
    lower: float
    upper: float


class ArmModel:
    def __init__(self, urdf_path: str, base: str = "base_link", tip: str = "a6_EE_holder",
                 tcp_offset=(0.0, 0.0, 0.0)):
        root = ET.parse(urdf_path).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, link = [], tip
        while link != base:
            j = by_child[link]
            chain.append(j)
            link = j.find("parent").get("link")
        self.joints: list[Joint] = []
        self.fixed_prefix: list[np.ndarray] = []
        for j in reversed(chain):
            o = j.find("origin")
            T = np.eye(4)
            if o is not None:
                T[:3, :3] = rpy_to_mat(*map(float, o.get("rpy", "0 0 0").split()))
                T[:3, 3] = list(map(float, o.get("xyz", "0 0 0").split()))
            if j.get("type") == "fixed":
                raise NotImplementedError("fixed joints inside the arm chain")
            lim = j.find("limit")
            lo, hi = float(lim.get("lower", 0)), float(lim.get("upper", 0))
            if j.get("type") == "continuous" or lo == hi:  # exporter writes 0/0 for continuous a6
                lo, hi = -np.pi, np.pi
            ax = np.array(list(map(float, j.find("axis").get("xyz").split())))
            self.joints.append(Joint(j.get("name"), T, ax / np.linalg.norm(ax), lo, hi))
        self.tcp = np.eye(4)
        self.tcp[:3, 3] = tcp_offset
        self.names = [j.name for j in self.joints]
        self.lower = np.array([j.lower for j in self.joints])
        self.upper = np.array([j.upper for j in self.joints])

    @property
    def dof(self) -> int:
        return len(self.joints)

    def fk(self, q: np.ndarray, with_frames: bool = False):
        T = np.eye(4)
        origins, axes = [], []
        for j, qi in zip(self.joints, q):
            T = T @ j.origin
            origins.append(T[:3, 3].copy())
            axes.append(T[:3, :3] @ j.axis)
            R = np.eye(4)
            R[:3, :3] = axis_angle(j.axis, qi)
            T = T @ R
        T = T @ self.tcp
        return (T, origins, axes) if with_frames else T

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        T, origins, axes = self.fk(q, with_frames=True)
        p = T[:3, 3]
        J = np.zeros((6, self.dof))
        for i, (o, a) in enumerate(zip(origins, axes)):
            J[:3, i] = np.cross(a, p - o)
            J[3:, i] = a
        return J

    def ik_step(self, q: np.ndarray, T_des: np.ndarray, damping: float = 0.05,
                rot_weight: float = 0.3, max_dq: float = 0.1) -> np.ndarray:
        """One DLS step toward T_des. Returns the new q (clipped to joint limits)."""
        T = self.fk(q)
        err = np.concatenate([T_des[:3, 3] - T[:3, 3], rot_weight * rot_err(T[:3, :3], T_des[:3, :3])])
        J = self.jacobian(q)
        J[3:] *= rot_weight
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), err)
        dq = np.clip(dq, -max_dq, max_dq)
        return np.clip(q + dq, self.lower, self.upper)

    def ik(self, q0: np.ndarray, T_des: np.ndarray, iters: int = 200, tol: float = 1e-3, **kw):
        q = np.array(q0, dtype=float)
        for _ in range(iters):
            q = self.ik_step(q, T_des, **kw)
            T = self.fk(q)
            if np.linalg.norm(T[:3, 3] - T_des[:3, 3]) < tol and \
                    np.linalg.norm(rot_err(T[:3, :3], T_des[:3, :3])) < 10 * tol:
                break
        return q
