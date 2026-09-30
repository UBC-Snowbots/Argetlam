"""assets/dev_arm/dev_arm_original.urdf -> dev_arm_gripper.urdf (see assets/dev_arm/SOURCE.md).

- mesh paths: package://dev_arm_description/meshes/X -> meshes/X
- the upstream finger links/joints are removed (origins ~45 cm from the gripper body, tilted axes, names swapped)
- placeholder parallel gripper on link_6, tool axis = link_6 +z (like upstream link_tt): a palm box and two prismatic
  fingers; the finger PAD CENTRE is the TCP at 0.2 m (= upstream link_tt), 85 mm max opening, 40 N per finger.
  Same finger/palm proportions as the rover2026 placeholder (palm front 4.5 cm above the TCP, fingertips 2 cm below).
- adds a massless-ish `tcp` link at the pad centre (the ghost line and grasp checks use it)
- joint efforts in the upstream URDF are placeholders (1.5); the Isaac actuators are set in configs/arms/dev_arm.yaml.

    python sim/make_dev_arm_urdf.py
"""
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets/dev_arm/dev_arm_original.urdf"
DST = ROOT / "assets/dev_arm/dev_arm_gripper.urdf"

TCP_Z = 0.20                       # upstream link_tt
STROKE = 0.0425                    # per finger (85 mm opening)
FINGER = (0.022, 0.012, 0.065)     # x width, y thickness, z length (along the tool axis)
PALM = (0.045, 0.100, 0.055)       # x, y, z (z along the tool axis)
FINGER_Z0 = TCP_Z - 0.045          # finger root: pad centre 45 mm into a 65 mm finger (tips 2 cm past the TCP)
PALM_Z0 = FINGER_Z0 - PALM[2]      # palm directly behind the fingers
PALM_MASS, FINGER_MASS = 0.35, 0.05


def box_inertia(m, x, y, z):
    return m * (y * y + z * z) / 12, m * (x * x + z * z) / 12, m * (x * x + y * y) / 12


def inertial(link, mass, xyz, dims):
    i = ET.SubElement(link, "inertial")
    ET.SubElement(i, "origin", xyz=xyz, rpy="0 0 0")
    ET.SubElement(i, "mass", value=f"{mass}")
    ixx, iyy, izz = box_inertia(mass, *dims)
    ET.SubElement(i, "inertia", ixx=f"{ixx:.3e}", ixy="0", ixz="0", iyy=f"{iyy:.3e}", iyz="0", izz=f"{izz:.3e}")


def box(link, xyz, dims, rgba):
    for tag in ("visual", "collision"):
        e = ET.SubElement(link, tag)
        ET.SubElement(e, "origin", xyz=xyz, rpy="0 0 0")
        ET.SubElement(ET.SubElement(e, "geometry"), "box", size=" ".join(map(str, dims)))
        if tag == "visual":
            m = ET.SubElement(e, "material", name=f"{link.get('name')}_mat")
            ET.SubElement(m, "color", rgba=rgba)


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
    for m in robot.iter("mesh"):
        m.set("filename", "meshes/" + m.get("filename").split("/")[-1])
    for j in list(robot.findall("joint")):
        if j.get("name") in ("finger_left_joint", "finger_right_joint"):
            robot.remove(j)
        elif j.get("name") == "joint_ee":
            ax = j.find("axis")
            if ax is not None:
                j.remove(ax)  # stray axis on a fixed joint
    for l in list(robot.findall("link")):
        if l.get("name") in ("finger_left", "finger_right"):
            robot.remove(l)

    palm = ET.SubElement(robot, "link", name="gripper_palm")
    inertial(palm, PALM_MASS, f"0 0 {PALM_Z0 + PALM[2] / 2}", PALM)
    box(palm, f"0 0 {PALM_Z0 + PALM[2] / 2}", PALM, "0.15 0.15 0.17 1")
    joint(robot, "gripper_mount", "fixed", "link_6", "gripper_palm", "0 0 0")
    for side, sign in (("left", 1), ("right", -1)):
        f = ET.SubElement(robot, "link", name=f"finger_{side}")
        inertial(f, FINGER_MASS, f"0 0 {FINGER[2] / 2}", FINGER)
        box(f, f"0 0 {FINGER[2] / 2}", FINGER, "0.75 0.75 0.78 1")
        y_open = sign * (STROKE + FINGER[1] / 2)
        joint(robot, f"finger_{side}_joint", "prismatic", "gripper_palm", f"finger_{side}",
              f"0 {y_open} {FINGER_Z0}", axis=f"0 {-sign} 0", lo=0.0, hi=STROKE, effort=40.0, vel=0.15)
    tcp = ET.SubElement(robot, "link", name="tcp")
    inertial(tcp, 0.001, "0 0 0", (0.01, 0.01, 0.01))
    joint(robot, "tcp_joint", "fixed", "link_6", "tcp", f"0 0 {TCP_Z}")

    ET.indent(tree)
    tree.write(DST, xml_declaration=True, encoding="utf-8")
    print(f"wrote {DST}: TCP {TCP_Z} m along link_6 z, opening {2 * STROKE * 1000:.0f} mm, "
          f"palm front {TCP_Z - FINGER_Z0:.3f} m above the TCP")


if __name__ == "__main__":
    main()
