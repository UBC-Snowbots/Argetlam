# Plane teleop (body.mode = plane)

Drive the dev_arm with your hands moving PARALLEL to the camera image plane, with the camera in FRONT of you.
MediaPipe depth (the z axis, motion toward/away from the camera) is never read, because on the first real run depth
was the noisy axis: ~28 mm std while holding still, versus ~3 mm in the image plane. So every control is read from
image-plane landmarks (x, y) only, normalized by your shoulder width so it does not matter how far you stand from the
camera. This is the dev_arm_real default; `arm_mirror`, `position` and `gesture` modes are still selectable.

## Control mapping (5 lines)
- robot **y (lateral)** <- your RIGHT hand horizontal. Move your right hand to YOUR right -> the gripper goes to your
  right (robot -y). (Both you and the arm face the same way "over the shoulder", so your right is the arm's right.)
- robot **z (vertical)** <- your RIGHT hand height. Raise it -> the gripper goes up.
- robot **x (reach)** <- your LEFT hand height. Raise it -> the gripper reaches forward / away from the base; lower it
  -> it comes back. (Depth would be the obvious control here, but depth is exactly the noisy axis, so reach is driven
  by a clean in-plane signal instead.)
- Relative / clutched: on engage the app captures your hand positions and the current TCP, then
  `target = tcp0 + ramp * gain * (hand_now - hand0)`, One-Euro filtered, deadbanded, clamped to the workspace box.
- gains (metres of TCP per shoulder-width of hand motion): lateral 0.20, vertical 0.15, reach 0.15. A comfortable
  range covers the box: lateral +/-1.0 sw -> +/-0.20 m; vertical +/-0.9 sw -> +/-0.135 m; reach +0.6/-0.5 sw ->
  +0.09/-0.075 m.

## Reachable box + IK
Box (dev_arm_real, carved from the real reachable shell): `[0.47, -0.22, 0.38] -> [0.64, 0.18, 0.60]` m, home TCP
`(0.545, -0.027, 0.520)`. IK holds the base-pose tool orientation and solves to joint goals with `teleop_orient:
frozen` (joints 4-6 held at the base pose, position-only IK on joints 1-3). On a 7^3 grid of the box, frozen reaches
95.6 % of points (axis 67.6 %, full 62.4 %), worst miss 72 mm at a corner. Targets outside the box are clamped and
the ghost turns amber.

## Engage / release
- The left-hand-raise clutch trigger is DISABLED in plane mode (your left hand drives reach now).
- Engage: press **F**, hold an open RIGHT palm up for ~1 s, or say "follow me". Engaging first calibrates (hold still
  ~2 s), then follows; the gain eases in over a 2 s ramp so the arm never jumps.
- Release: press **F** again, say "stop" (freezes in place), or DROP your control (right) arm - hold the right wrist
  below your hip line for >= 0.7 s and the arm stows to **base** (`on_drop`). Losing tracking for >= 1 s also stows to
  base; a lost link (nobody watching) freezes in place.

## How to stand / camera placement
Camera in FRONT of you, about chest height, far enough back that your whole upper body is in frame - both shoulders,
both hips, and both hands. Face the camera. Keep both hands up and visible while following (the right hand steers
lateral + vertical, the left hand sets reach; if the left hand drops out of view the reach simply holds). If the arm
moves left/right the wrong way, your webcam is mirrored: press **M** in the body client (or flip `plane.sign.lateral`).

## Tuning (configs/arms/dev_arm_real.yaml -> `plane`)
- `gain: {lateral, vertical, reach}` metres per shoulder-width: raise for more range per hand motion, lower if a
  control clamps on the box too often.
- `sign: {lateral, vertical, reach}`: flip if a control drives the wrong way (camera / handedness).
- `one_euro: {min_cutoff, beta}` on the normalized hand coords: lower min_cutoff = steadier but laggier.
- `deadband` (shoulder-widths), `ramp_s`, `drop_below_hip_s`, `conf_gate` as above. `side` picks the control hand.

## Try it live
```
# VM: bring the sim stack up on the real profile (mock/isaac backend only; never drives the real arm)
rover-stack on --arm dev_arm_real
# laptop: the body client (add --mirror if your webcam mirrors; M toggles)
python body_client.py --side right        # or --tunnel host:port if UDP can't reach the VM
```
Press **F** (hold still ~2 s to calibrate), then move your hands. Drop your right arm (wrist below your hips) or say
"stop" to stop. The green ghost in Isaac shows where the gripper is heading (amber = clamped at the box edge).

## Not verified until a live run
The 2026-10-23 replay used to pick the gains/signs was a camera-in-front arm_mirror recording that carried NO hand
landmark and had the left hand mostly out of view, so the reach (left-hand) control is only unit-tested, not
exercised on real data, and the recorded motion was large arm swings (not gentle plane motion), so the replay clamp %
is an offline upper bound, not a plane-teleop figure. Signs, gains and the feel need a person driving. `scripts/gate.sh`
was not run (it needs the live stack on 47100 = master).
