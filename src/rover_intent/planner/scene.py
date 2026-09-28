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


class Scene:
    def __init__(self, objects: dict[str, dict]):
        self.objects = {n: SceneObject(n, np.array(o["pos"], float), o.get("height", 0.10))
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

    def nearest(self, p: np.ndarray, within: float) -> SceneObject | None:
        best = min(self.objects.values(), key=lambda o: np.linalg.norm(o.pos - p), default=None)
        return best if best is not None and np.linalg.norm(best.pos - p) <= within else None
