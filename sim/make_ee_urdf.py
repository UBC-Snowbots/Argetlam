"""Make rover2026_ee.urdf: original arm + a temporary key-presser end effector.

- Fixes a6_rotation: exporter wrote lower=upper=0 (Isaac imported it LOCKED); make it a real continuous joint.
- Adds `ee_presser` (4 mm mount plate + 6 mm rod, 60 mm long + 8 mm rubber ball) on the tool face of
  a6_EE_holder, and a massless-ish frame link `ee_tip` at the very end of the ball = the contact point to target.
Tool face from the a6_EE_holder STL: the flat r=40 mm mounting face is at x=0, facing +x (link frame);
the -x side (to -0.091) is the r=21 mm shaft that sits inside the wrist.
"""
import os
import xml.etree.ElementTree as ET

HOME = os.path.expanduser("~")
SRC = f"{HOME}/projects/ubc-rover-arm/isaac/rover2026_import/rover2026.urdf"
DST = f"{HOME}/projects/ubc-rover-arm/isaac/rover2026_import/rover2026_ee.urdf"

FACE = 0.0            # holder origin -> tool face (m), along +x
PLATE_T, PLATE_R = 0.004, 0.030
ROD_L, ROD_R = 0.060, 0.003
BALL_R = 0.008
TIP = FACE + PLATE_T + ROD_L + BALL_R  # 0.072 m

tree = ET.parse(SRC)
robot = tree.getroot()

for j in robot.findall("joint"):
    if j.get("name") == "a6_rotation":
        lim = j.find("limit")
        for a in ("lower", "upper"):
            if a in lim.attrib:
                del lim.attrib[a]
        lim.set("effort", "10")
        lim.set("velocity", "6")


def inertial(parent, mass, ixx):
    i = ET.SubElement(parent, "inertial")
    ET.SubElement(i, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(i, "mass", value=str(mass))
    ET.SubElement(i, "inertia", ixx=str(ixx), ixy="0", ixz="0", iyy=str(ixx), iyz="0", izz=str(ixx))


def shape(parent, tag, xyz, rpy, geom, rgba):
    el = ET.SubElement(parent, tag)
    ET.SubElement(el, "origin", xyz=xyz, rpy=rpy)
    g = ET.SubElement(el, "geometry")
    ET.SubElement(g, geom[0], **geom[1])
    if tag == "visual":
        m = ET.SubElement(el, "material", name=f"mat_{geom[0]}_{rgba.replace(' ', '_')}")
        ET.SubElement(m, "color", rgba=rgba)


# presser link frame == holder frame; geometry placed along -x
pl = ET.SubElement(robot, "link", name="ee_presser")
inertial(pl, 0.02, 1e-6)
plate_c = f"{FACE + PLATE_T / 2:.4f} 0 0"
rod_c = f"{FACE + PLATE_T + ROD_L / 2:.4f} 0 0"
ball_c = f"{FACE + PLATE_T + ROD_L:.4f} 0 0"
for tag in ("visual", "collision"):
    shape(pl, tag, plate_c, "0 1.5708 0", ("cylinder", {"radius": str(PLATE_R), "length": str(PLATE_T)}), "0.20 0.20 0.22 1")
    shape(pl, tag, rod_c, "0 1.5708 0", ("cylinder", {"radius": str(ROD_R), "length": str(ROD_L)}), "0.85 0.85 0.88 1")
    shape(pl, tag, ball_c, "0 0 0", ("sphere", {"radius": str(BALL_R)}), "0.05 0.05 0.05 1")
jp = ET.SubElement(robot, "joint", name="ee_presser_mount", type="fixed")
ET.SubElement(jp, "origin", xyz="0 0 0", rpy="0 0 0")
ET.SubElement(jp, "parent", link="a6_EE_holder")
ET.SubElement(jp, "child", link="ee_presser")

tl = ET.SubElement(robot, "link", name="ee_tip")
inertial(tl, 0.001, 1e-8)
jt = ET.SubElement(robot, "joint", name="ee_tip_joint", type="fixed")
ET.SubElement(jt, "origin", xyz=f"{TIP:.4f} 0 0", rpy="0 0 0")
ET.SubElement(jt, "parent", link="ee_presser")
ET.SubElement(jt, "child", link="ee_tip")

ET.indent(tree, space="  ")
tree.write(DST, xml_declaration=True, encoding="utf-8")
print("wrote", DST, "tip at", TIP, "m from holder origin")
