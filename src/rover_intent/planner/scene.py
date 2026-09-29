"""Object registry: name -> pose in the arm base frame.

Scaffold: positions come from the config. Later: ArUco markers / a detector on the laptop camera
publish updates, and shared autonomy snaps the approach to the nearest object.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SceneObject:
    name: str
    pos: np.ndarray            # grasp point, base frame (m)
    height: float = 0.10       # for place-on offsets
    grasp_z: float = 0.0       # grasp point height above the object's bottom (from the config pos, objects on z=0)


class Scene:
    def __init__(self, objects: dict[str, dict]):
        self.objects = {n: SceneObject(n, np.array(o["pos"], float), o.get("height", 0.10), o["pos"][2])
                        for n, o in objects.items()}

    def names(self) -> list[str]:
        return list(self.objects)

    def get(self, name: str) -> SceneObject | None:
        return self.objects.get(name)

    def update(self, name: str, pos) -> None:
        if name in self.objects:
            self.objects[name].pos = np.asarray(pos, float)
        else:
            self.objects[name] = SceneObject(name, np.asarray(pos, float))

    def update_from_bottom(self, name: str, bottom) -> None:
        """Perception update (sim ground truth now, ArUco later): the object's bottom-centre position."""
        o = self.objects.get(name)
        if o is not None:
            o.pos = np.asarray(bottom, float) + np.array([0, 0, o.grasp_z])

    def nearest(self, p: np.ndarray, within: float) -> SceneObject | None:
        best = min(self.objects.values(), key=lambda o: np.linalg.norm(o.pos - p), default=None)
        return best if best is not None and np.linalg.norm(best.pos - p) <= within else None
