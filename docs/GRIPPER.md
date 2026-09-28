# Gripper reference

The rover2026 URDF has no gripper (the keypad presser from LearnFlake is dropped). We use proven designs
as references instead of inventing one:

| Role | Reference | Why |
|---|---|---|
| **Sim** | **Robotiq 2F-85** (85 mm stroke, parallel, adaptive fingers) | Isaac Lab ships it (`UR10e_ROBOTIQ_2F_85_CFG` in `isaaclab_assets/robots/universal_robots.py`) with the mimic-joint drive already solved; the most-used research gripper, so grasp behaviour in sim is believable. |
| **Real (build)** | **SO-101 (LeRobot) gripper**: open-source, 3D-printed, one Feetech STS3215 serial-bus servo | Printable in a day, cheap, huge community, and the servo reports position/load, so "grasped vs missed" can be checked. |
| Alternative | Parallel jaw on a single hobby servo (rack-and-pinion) | If the team already has servos/printers but not STS3215. |

Integration notes
- Mount: the +x tool face of `a6_EE_holder` (r = 40 mm flat face at x = 0 in the link frame; −x is the shaft).
- TCP: set `arm.tcp_offset` in configs/default.yaml to the finger-pad centre once the real gripper is chosen
  (placeholder 0.15 m; 2F-85 is ~0.15 m face-to-pad-centre).
- Payload: the specs sheet gives 34 N·m worst-case shoulder torque with 0.5 kg at the EE; gripper + cup must fit
  under the real motors' budget (datasheets still missing).
