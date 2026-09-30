# dev_arm (the 2023 / "development" arm)

Vendored from UBC-Snowbots/RoverFlake2 `src/dev_arm_description/` (see UPSTREAM_COMMIT.txt), created by the team
(Roozki, 2024-11). "our development arm (the arm used in testing from Fall 2024 - ___)". Believed to be the 2023 arm:
gear reductions and shoulder/upper-arm geometry match the 2023 firmware/URDF (inferred, not documented).

**License: RoverFlake2 has no LICENSE file. Get the team's OK before publishing this folder in a public repo.**

Changes (sim/make_dev_arm_urdf.py → dev_arm_gripper.urdf): mesh paths made relative; the upstream finger joints
(origins ~45 cm from the gripper body, tilted axes, left/right names swapped; raw CAD offsets) are replaced by a
placeholder 2-finger parallel gripper on link_6 whose pad centre is the upstream tool frame `link_tt` (0.2 m along
link_6 z). The real finger geometry/travel is UNCONFIRMED.
