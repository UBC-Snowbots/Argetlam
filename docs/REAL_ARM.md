# Real arm revival checklist (the team's old arm)

Status 2026-09-28: control path unknown ("needs revival"). Kinematics assumed similar to rover2026, to confirm.
`RealArm` in src/rover_intent/robot/backends.py raises until this is filled in.

1. **Inventory**: motors (model, gearbox ratio), drivers/ESCs, MCU + firmware repo, encoders (absolute?), power supply, gripper.
2. **Kinematics**: measure link lengths and joint axes and compare to rover2026.urdf (doc: rover2026_arm_specs.md).
   If they differ, export a URDF for this arm and point `arm.urdf` at it.
3. **Control path**: how a joint target gets to a motor today (ROS 2 topic / serial protocol / CAN frames).
   Write the smallest possible driver: `read_joints()` + `command(q, gripper)`.
4. **Limits**: joint ranges, max safe speed/accel per joint → `arm.max_vel/max_acc` (replace the PLACEHOLDERS).
5. **Safety hardware**: physical e-stop that cuts motor power, independent of software. Deadman on the laptop.
6. **Bring-up order**: one joint at a time at 10 % speed → all joints in joint space → IK in free space → mirroring.
