"""Deterministic skills: Intent -> list of (TCP position, gripper) waypoints.

Nemotron picks the skill and its arguments; the geometry here is plain code, so it is testable
and never hallucinates a pose. Orientation is a fixed top-down grasp for now.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import Action, Intent
from .scene import Scene

TRANSIT_Z = 0.30  # m: minimum travel height; Scene.transit_z raises it above the tallest object on the table
APPROACH = 0.10   # m above the grasp point
LIFT = 0.12
BESIDE = 0.12     # m offset for beside/left/right/front/behind

_OFFSETS = {"beside": (0, BESIDE, 0), "left_of": (0, BESIDE, 0), "right_of": (0, -BESIDE, 0),
            "in_front_of": (BESIDE, 0, 0), "behind": (-BESIDE, 0, 0)}


@dataclass
class Waypoint:
    pos: np.ndarray
    gripper: float  # 0 open .. 1 closed
    label: str
    obj: str | None = None  # object the waypoint acts on (for close/open bookkeeping)
    dwell: float = 0.0      # s to wait at the waypoint (let the fingers close/open)
    tol_xy: float | None = None  # m; tighter SIDEWAYS alignment needed before descending (None = default 1 cm, 3D)
    speed: float = 0.30          # m/s of the straight-line "carrot" the IK chases toward this waypoint

GRIP_DWELL = 0.8
# Before descending onto / placing an object the gripper must be settled tightly: the bottle is 70 mm wide and the
# fingers open to 85 mm, so there is only 7.5 mm clearance per side. With the old 10 mm tolerance the arm started
# down while still 5-8 mm off and a finger landed on the rim (2026-09-30). Sideways only: the drives sit ~3-4 mm
# low at steady state, so a 3 mm 3D tolerance never settles (gate: 0/9).
ALIGN_TOL, ALIGN_DWELL = 0.003, 0.3
# The final descent must be a STRAIGHT line: jumping the IK target 10 cm down made the joints take a curved path that
# swung the gripper 13 mm sideways into the bottle rim (traced 2026-09-30). Slow carrot for the last 10 cm.
APPROACH_SPEED = 0.05


class SkillError(ValueError):
    pass


def plan(intent: Intent, scene: Scene, held: str | None) -> list[Waypoint]:
    """`held` = name of the object in the gripper (None if empty)."""
    up = np.array([0, 0, 1.0])
    if intent.action == Action.pick:
        if held is not None:
            raise SkillError(f"already holding the {held}; say 'let go' or 'put it beside ...' first")
        o = _obj(scene, intent.object)
        p = o.pos.copy()
        tz = scene.transit_z
        return [Waypoint(_transit(p, tz), 0, "above-object"), Waypoint(p + APPROACH * up, 0, "pre-grasp", tol_xy=ALIGN_TOL, dwell=ALIGN_DWELL),
                Waypoint(p, 0, "grasp-pos", speed=APPROACH_SPEED), Waypoint(p, 1, "close", o.name, GRIP_DWELL),
                Waypoint(_transit(p, tz), 1, "lift")]
    if intent.action == Action.place:
        if held is None:
            raise SkillError("not holding anything")
        return _place(intent, scene, _obj(scene, held))
    if intent.action == Action.move:
        if held is not None:
            raise SkillError(f"already holding the {held}")
        pick = plan(Intent(action=Action.pick, object=intent.object), scene, None)
        return pick + _place(intent, scene, _obj(scene, intent.object))
    raise SkillError(f"no motion skill for {intent.action.value}")


def _place(intent: Intent, scene: Scene, obj) -> list[Waypoint]:
    """Release `obj` so it ends up in `relation` to the reference. Assumes both stand on the same surface."""
    up = np.array([0, 0, 1.0])
    ref = _obj(scene, intent.reference)
    if intent.relation in ("on", "into"):
        ref_bottom = ref.pos[2] - ref.grasp_z
        goal = np.array([ref.pos[0], ref.pos[1], ref_bottom + ref.height + obj.grasp_z + 0.01])
    else:
        goal = ref.pos + np.array(_OFFSETS.get(intent.relation or "beside", _OFFSETS["beside"]))
        goal[2] = obj.pos[2]  # same support surface: release at the object's own grasp height
    tz = scene.transit_z
    return [Waypoint(_transit(goal, tz), 1, "above-goal"), Waypoint(goal + LIFT * up, 1, "pre-place", tol_xy=ALIGN_TOL, dwell=ALIGN_DWELL),
            Waypoint(goal, 1, "place-pos", speed=APPROACH_SPEED), Waypoint(goal, 0, "open", obj.name, GRIP_DWELL),
            Waypoint(_transit(goal, tz), 0, "retreat")]


def _transit(p: np.ndarray, tz: float = None) -> np.ndarray:
    """Same xy, raised to the travel height (never lowered)."""
    return np.array([p[0], p[1], max(p[2], TRANSIT_Z if tz is None else tz)])


def _obj(scene: Scene, name: str | None):
    o = scene.get(name) if name else None
    if o is None:
        raise SkillError(f"unknown object {name!r}; known: {scene.names()}")
    return o
