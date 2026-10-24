# Full-arm body teleop ("arm mirror")

Drive the dev_arm by mirroring your WHOLE arm as joint angles, with no IK. This replaces the old body-teleop path
for the real arm (`dev_arm_real`), which mapped the wrist POSITION into a small workspace box and ran frozen-wrist
IK. On the first real run that felt bad: webcam depth jitter (28 mm even while holding still, 9x worse laterally)
pushed the robot forward, the target was pinned on a box wall 46 % of the time and below the floor 39 %, it ran
uncalibrated, and the clutch thrashed. Mirroring joints instead of a boxed TCP sidesteps all of that.

## How it works (5 lines)
1. From the MediaPipe pose, build a TORSO frame (both shoulders + hip midpoint) so the mapping does not depend on
   where the camera sits.
2. Read the operator's arm as angles in that frame: J1 = azimuth of the upper arm, J2 = its elevation, J3 = the
   elbow bend (upper arm vs forearm), J5 = the wrist bend (forearm vs hand). J4 (forearm roll) and J6 (tool roll)
   stay at the base pose.
3. Map each angle to a robot joint absolutely: `q = clip(home + sign*gain*(raw - neutral), joint_limits)`.
4. Smooth per joint: One-Euro filter on the joint angles, a small deadband, a per-joint rate limit (= the profile
   `max_vel`), and on engage a 2 s ramp from the current pose to the mirrored pose so the arm never jumps.
5. The joint targets go STRAIGHT into the SafetyLayer (the same no-IK path the button presets and the real-arm
   mirror use), never through IK.

Module: `src/rover_intent/body/arm_mirror.py` (`arm_angles()` pure function + `ArmMirror` mapper). Enabled by
`body.mode: arm_mirror`; the only other body teleop mode is `plane` (the default, see docs/PLANE_TELEOP.md). The
legacy position / calibrated / gesture modes were removed (see docs/BODY_TELEOP.md).

## Kinematics (verified against assets/dev_arm/dev_arm_gripper.urdf)
| joint | motion | driven by |
|---|---|---|
| J1 | base yaw (axis y of link_1) | azimuth of the upper arm about the torso vertical |
| J2 | shoulder pitch (axis x) | elevation of the upper arm (asin of its vertical component) |
| J3 | elbow pitch (axis z), upper arm 0.36 m | included angle between upper arm and forearm |
| J4 | forearm roll (axis y), forearm 0.466 m | held at the base pose |
| J5 | wrist pitch (axis z) | forearm vs hand (index/pinky midpoint); held if no hand landmark |
| J6 | tool roll (axis z) | held at the base pose |

There is no humeral twist in the arm, so the upper-arm direction fixes J1+J2 and the elbow fixes J3; this decoupling
is why we read the upper arm (not the wrist) for J1/J2.

## Enable / disable and the drop-to-base behaviour
- ONE deliberate trigger engages/disengages, reusing the clutch: press **F** in the body client, or raise your
  **LEFT** hand above your left shoulder (hold-to-follow). Engaging first calibrates (hold still ~2 s), then follows.
- On engage, the arm ramps from where it is to the mirrored pose over `ramp_s` (2 s). No jump.
- On DISABLE (F off / left hand lowered) or TRACKING LOST, the arm returns to the **base** preset ("when I drop my
  arm, return it to base"). This reuses `arm.on_drop: base` and an extended drop-reason set in arm_mirror mode.
- Voice **"stop"** and the STOP button still just STOP (freeze in place), and a lost link (nobody watching) freezes.
- J1 is HELD whenever the upper arm is within `j1_hold_vert_deg` (20 deg) of vertical, because azimuth is undefined
  when the arm hangs. A non-visible pose is skipped (confidence gate), so the arm holds instead of chasing noise.

## Camera placement (recommended)
Put the webcam **beside** the operator (looking across the body), not in front. Then the arm's up/down and
forward/back motion is in the image plane, so J2 (elevation), J3 (elbow) and J5 (wrist) are read from clean image
axes; only J1 (azimuth = how far you swing across your body) depends on depth, and it is filtered hard and gated near
vertical. The torso-frame construction needs both shoulders and both hips in view, so stand far enough back that your
torso is in frame; a slight turn so the camera sees the front of your torso helps the shoulder/hip depth.
With the camera in front (as in the 2026-10-23 recording) everything still works, but azimuth is the noisiest angle.

## Tuning (configs/arms/dev_arm_real.yaml -> `arm_mirror`)
`q = clip(home + sign*gain*(raw - neutral), limits)` per joint. Defaults come from Pranav's 2026-10-23 session
medians (camera in front).
- `neutral: {az, el, elbow, wrist}` (radians): the raw arm angle that should map to the base pose. To re-centre:
  stand in your comfortable neutral, read the logged raw angles (or print `arm_angles(pose)`), and set these.
- `gain: {j1, j2, j3, j5}` (rad robot per rad human): 1.0 is 1:1 mirroring. Lower a gain if that joint clamps too
  often; raise it if the arm under-reaches. `j3`/`j5` default to 0.8 so a full elbow/wrist swing fits the travel.
- `sign: {...}`: flip if a joint moves the wrong way (depends on the camera and which arm). Check J1 first: swing
  your arm left and confirm the base yaws the same way you'd expect; flip `sign.j1` if not. Left arm usually needs a
  J1 sign flip vs right.
- `one_euro: {min_cutoff, beta}` on the JOINT angles: lower `min_cutoff` = steadier but laggier; `beta` adds speed-
  dependent cut. Start 1.0 / 0.05.
- `deadband_deg` (1.5): hold a joint until its target drifts this far, killing micro-jitter when still.
- `j1_hold_vert_deg` (20), `ramp_s` (2.0), `conf_gate` (true) as above.
- `side: right|left`: which arm the operator uses (the body client also needs `--side`).

## Try it live
```
# VM: bring the sim stack up on the real profile (NEVER drives the real arm; mock/isaac backend only)
rover-stack on --arm dev_arm_real

# laptop: the body client, right arm (add --mirror if your webcam mirrors; M toggles)
python body_client.py --side right            # or --tunnel host:port if UDP can't reach the VM
```
Press **F** (or raise your left hand) to engage: hold still ~2 s to calibrate, then move your arm and the robot
mirrors it. Press **F** again (or lower your left hand) to return to base. Say "stop" to freeze. The green ghost in
Isaac shows where the gripper is heading (amber = a joint is clamped at its travel limit).

## Not verified until a live run
Signs/neutral are tuned from a camera-in-front recording; with the camera beside you, J1 sign/neutral will need a
one-time retune. Joint-mirror feel, the exact gains, and the wrist-bend (J5) mapping have only been checked in sim
and on replayed packets, not yet with a person driving the real arm.
