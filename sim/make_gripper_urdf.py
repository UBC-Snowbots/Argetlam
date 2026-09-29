"""assets/rover2026/rover2026.urdf -> rover2026_gripper.urdf: the arm + a 2-finger parallel gripper.

Gripper proportions follow the Robotiq 2F-85 (docs/GRIPPER.md): 85 mm max opening, ~0.9 kg,
TCP (finger-pad centre) ~145 mm from the mounting face. Geometry is simple boxes, not the Robotiq mesh.
Fingers are two independent prismatic joints driven to the same target (no mimic joints needed in PhysX).

Also fixes a6_rotation: the exporter wrote lower=upper=0 for a continuous joint, which Isaac imports as LOCKED.

Mount: the flat r=40 mm tool face of a6_EE_holder is at x=0 facing +x (the -x side is the shaft in the wrist).
    python sim/make_gripper_urdf.py
"""
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets/rover2026/rover2026.urdf"
DST = ROOT / "assets/rover2026/rover2026_gripper.urdf"

PLATE_L, PLATE_R = 0.010, 0.040                 # mounting plate (along x)
PALM = (0.090, 0.100, 0.045)                    # palm box (x, y, z)
PALM_X0 = PLATE_L                               # palm starts after the plate
FINGER = (0.065, 0.012, 0.022)                  # finger box (x length, y thickness, z width)
FINGER_X0 = PALM_X0 + PALM[0]                   # fingers start at the palm front: x = 0.100
STROKE = 0.0425                                 # per finger; 2 x 42.5 = 85 mm opening
TCP_X = FINGER_X0 + 0.045                       # pad centre, 45 mm into the 65 mm finger -> 0.145
PALM_MASS, FINGER_MASS = 0.80, 0.05
FINGER_EFFORT, FINGER_VEL = 40.0, 0.15          # N per finger, m/s


def box_inertia(m, x, y, z):
    return m * (y * y + z * z) / 12, m * (x * x + z * z) / 12, m * (x * x + y * y) / 12


def inertial(link, mass, xyz, dims):
    i = ET.SubElement(link, "inertial")
    ET.SubElement(i, "origin", xyz=xyz, rpy="0 0 0")
    ET.SubElement(i, "mass", value=f"{mass}")
    ixx, iyy, izz = box_inertia(mass, *dims)
    ET.SubElement(i, "inertia", ixx=f"{ixx:.3e}", ixy="0", ixz="0", iyy=f"{iyy:.3e}", iyz="0", izz=f"{izz:.3e}")


def shape(link, kind, xyz, rpy, geom: dict, rgba=None):
    for tag in ("visual", "collision"):
        e = ET.SubElement(link, tag)
        ET.SubElement(e, "origin", xyz=xyz, rpy=rpy)
        g = ET.SubElement(ET.SubElement(e, "geometry"), kind, **geom)
        if tag == "visual" and rgba:
            m = ET.SubElement(e, "material", name=f"{link.get('name')}_mat")
            ET.SubElement(m, "color", rgba=rgba)
    return g


def joint(robot, name, kind, parent, child, xyz, axis=None, lo=None, hi=None, effort=None, vel=None):
    j = ET.SubElement(robot, "joint", name=name, type=kind)
    ET.SubElement(j, "origin", xyz=xyz, rpy="0 0 0")
    ET.SubElement(j, "parent", link=parent)
    ET.SubElement(j, "child", link=child)
    if axis:
        ET.SubElement(j, "axis", xyz=axis)
        ET.SubElement(j, "limit", lower=f"{lo}", upper=f"{hi}", effort=f"{effort}", velocity=f"{vel}")


def main():
    tree = ET.parse(SRC)
    robot = tree.getroot()
    for j in robot.findall("joint"):
        if j.get("name") == "a6_rotation":
            lim = j.find("limit")
            for a in ("lower", "upper"):
                lim.attrib.pop(a, None)
            lim.set("effort", "10")
            lim.set("velocity", "6")

    base = ET.SubElement(robot, "link", name="gripper_base")
    inertial(base, PALM_MASS, f"{PALM_X0 + PALM[0] / 2} 0 0", PALM)
    shape(base, "cylinder", f"{PLATE_L / 2} 0 0", "0 1.5707963 0", {"radius": f"{PLATE_R}", "length": f"{PLATE_L}"},
          "0.15 0.15 0.17 1")
    # second visual/collision pair for the palm
    shape(base, "box", f"{PALM_X0 + PALM[0] / 2} 0 0", "0 0 0", {"size": " ".join(map(str, PALM))},
          "0.15 0.15 0.17 1")
    joint(robot, "gripper_mount", "fixed", "a6_EE_holder", "gripper_base", "0 0 0")

    for side, sign in (("left", 1), ("right", -1)):
        f = ET.SubElement(robot, "link", name=f"finger_{side}")
        inertial(f, FINGER_MASS, f"{FINGER[0] / 2} 0 0", FINGER)
        shape(f, "box", f"{FINGER[0] / 2} 0 0", "0 0 0", {"size": " ".join(map(str, FINGER))}, "0.75 0.75 0.78 1")
        y_open = sign * (STROKE + FINGER[1] / 2)  # inner face at +-42.5 mm when open
        joint(robot, f"finger_{side}_joint", "prismatic", "gripper_base", f"finger_{side}",
              f"{FINGER_X0} {y_open} 0", axis=f"0 {-sign} 0", lo=0.0, hi=STROKE,
              effort=FINGER_EFFORT, vel=FINGER_VEL)

    tcp = ET.SubElement(robot, "link", name="tcp")
    inertial(tcp, 0.001, "0 0 0", (0.01, 0.01, 0.01))
    joint(robot, "tcp_joint", "fixed", "gripper_base", "tcp", f"{TCP_X} 0 0")

    ET.indent(tree)
    tree.write(DST, xml_declaration=True, encoding="utf-8")
    print(f"wrote {DST}  (TCP {TCP_X:.3f} m from the tool face, opening {2 * STROKE * 1000:.0f} mm)")


if __name__ == "__main__":
    main()
