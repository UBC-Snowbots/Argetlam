"""Deterministic skills: Intent -> list of (TCP position, gripper) waypoints.

Nemotron picks the skill and its arguments; the geometry here is plain code, so it is testable
and never hallucinates a pose. Orientation is a fixed top-down grasp for now.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import Action, Intent
from .scene import Scene

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


class SkillError(ValueError):
    pass


def plan(intent: Intent, scene: Scene, holding: bool) -> list[Waypoint]:
    up = np.array([0, 0, 1.0])
    if intent.action == Action.pick:
        o = _obj(scene, intent.object)
        return [Waypoint(o.pos + APPROACH * up, 0, "pre-grasp"), Waypoint(o.pos, 0, "grasp-pos"),
                Waypoint(o.pos, 1, "close"), Waypoint(o.pos + LIFT * up, 1, "lift")]
    if intent.action == Action.place:
        if not holding:
            raise SkillError("not holding anything")
        return _place(intent, scene)
    if intent.action == Action.move:
        pick = plan(Intent(action=Action.pick, object=intent.object), scene, holding)
        return pick + _place(intent, scene)
    raise SkillError(f"no motion skill for {intent.action.value}")


def _place(intent: Intent, scene: Scene) -> list[Waypoint]:
    up = np.array([0, 0, 1.0])
    ref = _obj(scene, intent.reference)
    if intent.relation in ("on", "into"):
        goal = ref.pos + (ref.height + 0.02) * up
    else:
        goal = ref.pos + np.array(_OFFSETS.get(intent.relation or "beside", _OFFSETS["beside"]))
    return [Waypoint(goal + LIFT * up, 1, "pre-place"), Waypoint(goal, 1, "place-pos"),
            Waypoint(goal, 0, "open"), Waypoint(goal + APPROACH * up, 0, "retreat")]


def _obj(scene: Scene, name: str | None):
    o = scene.get(name) if name else None
    if o is None:
        raise SkillError(f"unknown object {name!r}; known: {scene.names()}")
    return o
