"""Known-object library: the same objects as the sim scene (name, footprint, height, grasp height).

A single fixed RGB camera cannot measure depth, so perception leans on what is known a-priori:
the objects rest on a flat table at a known height and each known object has a known size.  The
detector only has to say *which* object and *where in the image*; this library supplies the rest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ObjInfo:
    name: str
    radius: float        # m, footprint half-extent (cylinder radius, or box half-width)
    height: float        # m, full height standing on the table
    grasp_z: float       # m, grasp-point height above the object's bottom
    shape: str = "cylinder"
    desc: str = ""

    def silhouette_points(self, xy, z_table: float, n: int = 24) -> np.ndarray:
        """Sample points on the object's surface standing at footprint-centre (x, y) on z_table.

        Used to compute a ground-truth image bounding box (project these, take min/max).  Good
        enough for both the cylinder and the box (the box is sampled on its circumscribing shell,
        which slightly over-covers the corners - fine for a silhouette bound).
        """
        x, y = float(xy[0]), float(xy[1])
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        if self.shape == "box":
            # the 8 corners of the axis-aligned box
            r = self.radius
            zs = [z_table, z_table + self.height]
            return np.array([[x + sx * r, y + sy * r, z] for sx in (-1, 1) for sy in (-1, 1) for z in zs])
        ring = np.stack([x + self.radius * np.cos(ang), y + self.radius * np.sin(ang)], axis=1)
        bottom = np.hstack([ring, np.full((n, 1), z_table)])
        top = np.hstack([ring, np.full((n, 1), z_table + self.height)])
        return np.vstack([bottom, top])


def library_from_scene(scene_cfg: dict) -> dict[str, ObjInfo]:
    """Build the object library from a config `scene:` block (configs/arms/<arm>.yaml)."""
    lib = {}
    for name, o in scene_cfg.items():
        pos_z = o["pos"][2]
        height = o.get("height", 0.10)
        # grasp_z is the grasp point above the object's bottom; the config pos[2] is the grasp
        # point in the base frame and objects stand on z=support (taken as 0 here, overridden below).
        lib[name] = ObjInfo(name=name, radius=o.get("radius", 0.03), height=height,
                            grasp_z=pos_z, shape=o.get("shape", "cylinder"), desc=o.get("desc", ""))
    return lib


def set_support(lib: dict[str, ObjInfo], support_z: float) -> None:
    """Fix each object's grasp_z to be relative to the given support height (table top)."""
    for o in lib.values():
        o.grasp_z = o.grasp_z - support_z
