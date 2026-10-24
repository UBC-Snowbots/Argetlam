# Body teleop modes

Body teleop drives the arm from a webcam pose (laptop/body_client.py -> MediaPipe -> UDP -> app). There are exactly
two modes, set by `body.mode` in the config:

- **plane** (default everywhere, `body.mode: plane`) - image-plane TCP teleop. Your hands move PARALLEL to the camera
  image; MediaPipe depth (the noisy axis) is never read. Right hand drives the gripper left/right + up/down, left-hand
  height drives reach as a joystick rate. Runs through the IK path. Full detail in [PLANE_TELEOP.md](PLANE_TELEOP.md).
- **arm_mirror** (fallback, `body.mode: arm_mirror`) - full-arm joint mirroring. Your whole arm's pose (azimuth,
  elevation, elbow, wrist) maps straight to the robot joints, no IK. Full detail in [ARM_MIRROR.md](ARM_MIRROR.md).

Both modes are *relative / clutched*: on engage the app anchors at the current pose so there is no jump, then follows
your motion. Engage with **F**, an open palm held up ~1 s, or a raised left hand (plane uses the left hand for reach,
so there it is F / palm only). Both use a plain open/close gripper driven by your hand (`grip_from_hand`), with no
AUTO grab-assist.

## Removed (2026-10-24)

The legacy body modes were removed, leaving only plane + arm_mirror:

- **position** - wrist-position-relative-to-shoulder mapped into a small TCP box (`body/retarget.py:Retargeter`).
  Webcam depth jitter pushed the target around and pinned it on the box walls; this is what plane/arm_mirror replaced.
- **gesture** - swing/raise/reach -> yaw/height/reach cylindrical map (`body/mirror.py:GestureMirror`).
- **calibration** - the guided 4-target calibration session + `CalibratedMapper` (`body/calibration.py`,
  `body/calib_session.py`) that produced a saved TCP profile. The C / X keys and the calibration overlays are gone.

Note: the **clutch** still has a "calibrating" state - that is just the hold-still-to-re-centre step before following,
unrelated to the removed calibration session. The body-teleop grab-assist (fist -> AUTO pick) went away with position
mode; plane and arm_mirror never had it. The arbiter's grab-assist code is left in place (voice skills share parts of
it), but the hand-triggered AUTO assist path is no longer reachable from body teleop.
