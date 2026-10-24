# Plane teleop (body.mode = plane)

Drive the dev_arm with your hands moving PARALLEL to the camera image plane, with the camera in FRONT of you.
MediaPipe depth (the z axis, motion toward/away from the camera) is never read, because on the first real run depth
was the noisy axis: ~28 mm std while holding still, versus ~3 mm in the image plane. So every control is read from
image-plane landmarks (x, y) only, normalized by your shoulder width so it does not matter how far you stand from the
camera. This is the default body teleop mode everywhere; the only other mode is `arm_mirror` (full-arm joint
mirroring, see docs/ARM_MIRROR.md and docs/BODY_TELEOP.md).

## Control mapping (5 lines)
- robot **y (lateral)** <- your RIGHT hand horizontal. Move your right hand to YOUR right -> the gripper goes to your
  right (robot -y). (Both you and the arm face the same way "over the shoulder", so your right is the arm's right.)
- robot **z (vertical)** <- your RIGHT hand height. Raise it -> the gripper goes up.
- robot **x (reach)** <- your LEFT hand height, as a JOYSTICK (a RATE, not a position - new 2026-10-24, see "Reach
  is a rate" below). Hold the hand near **shoulder height** and the gripper creeps **forward**; near your **lower
  chest / mid torso** and it creeps **back**; **in between it holds**. The further past the threshold, the faster,
  up to a cap. (Depth would be the obvious reach control, but depth is exactly the noisy axis, so reach rides a clean
  in-plane height signal instead.)
- Relative / clutched (lateral + vertical only): on engage the app captures your right-hand position and the current
  TCP, then `y,z = tcp0 + ramp * gain * (hand_now - hand0)`, One-Euro filtered, deadbanded, clamped to the box.
- gains (metres of TCP per shoulder-width of hand motion): lateral 0.20, vertical 0.15. A comfortable range covers
  the box: lateral +/-1.0 sw -> +/-0.20 m; vertical +/-0.9 sw -> +/-0.135 m. (The reach gain 0.15 only applies in
  the old `reach_mode: position`.)

## Reach is a rate (joystick), not a position (2026-10-24)
The first live drive (2026-10-23, Pranav on the real arm) found reach "sucked": lateral + vertical from the right
hand were "pretty good", but reach from left-hand height as a POSITION map gave almost no forward room (the box only
allowed x 0.47..0.64, about +9.5 cm forward of the base pose) and, being relative to the hand height AT ENGAGE, left
no room to go back when the hand hung low at engage - plus height->forward as a position felt unintuitive. So reach
is now a **joystick rate** (`reach_mode: rate`, the dev_arm_real default):

- The left-wrist height is read against **your own body** (image plane only, shoulder-width normalized, **absolute,
  not relative to engage**). The shoulder line and your hip midpoint set your torso; two thresholds sit a fraction of
  that torso span below the shoulders: hand **above the forward threshold (~shoulder height)** -> reach moves
  **forward**; **below the back threshold (~lower chest / mid torso)** -> **back**; **between** -> **dead zone, reach
  holds**. Defaults: forward_frac 0.10, back_frac 0.45 of the shoulder->hip span.
- Speed scales with how far past the threshold your hand is, up to **max_speed 0.04 m/s (4 cm/s)** reached 0.5
  shoulder-widths past the threshold - well inside the real arm's limits (REAL_DATA.md 4a: the arm runs at half the
  commanded speed). A little **hysteresis (0.08 sw)** at each threshold stops chatter.
- The rate is integrated onto the TCP x at engage, clamped to the box. **At a box wall the rate just stops** (ghost
  turns amber) and **reversing moves away immediately** (no wind-up). If the **left hand is not tracked, reach holds**
  (rate 0). The engage ramp eases the rate in too, so engaging never jerks.
- Status: the app reports `reach_rate` (signed m/s) and `reach_state` (`>>` forward / `<<` back / `hold`) so a HUD /
  body client can show "REACH >>". Keep the old position map with `plane.reach_mode: position`.

## Reachable box + IK
Box (dev_arm_real, carved from the real reachable shell): `[0.47, -0.22, 0.38] -> [0.70, 0.18, 0.60]` m, home TCP
`(0.545, -0.027, 0.520)`. The x max was widened **0.64 -> 0.70 on 2026-10-24** for more forward reach under the rate
control: a frozen-wrist IK sweep (2026-10-24) reaches x to ~0.75 across the whole y,z face, so 0.70 keeps ~5 cm
margin; forward room from the base pose grows from 9.5 cm to 15.5 cm (back room unchanged at 7.5 cm; x min kept at
0.47, already near the reachable back edge). IK holds the base-pose tool orientation and solves to joint goals with
`teleop_orient: frozen` (joints 4-6 held at the base pose, position-only IK on joints 1-3). On a 5^3 / 7^3 grid of
the widened box, frozen reaches 95.2 % / 96.5 % of points, worst miss 72 mm at a y/z corner (unchanged by the x
widening). Targets outside the box are clamped and the ghost turns amber.

## Engage / release
- The left-hand-raise clutch trigger is DISABLED in plane mode (your left hand drives reach now).
- Engage: press **F**, hold an open RIGHT palm up for ~1 s, or say "follow me". Engaging first calibrates (hold still
  ~2 s), then follows; the gain eases in over a 2 s ramp so the arm never jumps.
- Release: press **F** again, say "stop" (freezes in place), or DROP your control (right) arm - hold the right wrist
  below your hip line for >= 0.7 s and the arm stows to **base** (`on_drop`). Losing tracking for >= 1 s also stows to
  base; a lost link (nobody watching) freezes in place.

## How to stand / camera placement
Camera in FRONT of you, about chest height, far enough back that your whole upper body is in frame - both shoulders,
both hips, and both hands. Face the camera. The RIGHT hand steers lateral + vertical. The LEFT hand is the reach
joystick: **rest it around mid-chest to hold reach**, raise it toward your shoulder to creep forward, drop it toward
your waist to creep back. If the left hand drops out of view the reach simply holds (rate 0). If the arm moves
left/right the wrong way, your webcam is mirrored: press **M** in the body client (or flip `plane.sign.lateral`).

## Tuning (configs/arms/dev_arm_real.yaml -> `plane`)
- `reach_mode: rate | position` - `rate` (default) is the joystick reach above; `position` restores the old
  height->x position map (uses `gain.reach`).
- `reach_rate:` (rate mode only)
  - `forward_frac` / `back_frac` - thresholds as a fraction of the shoulder->hip torso span below the shoulders
    (0.10 ~ shoulder height, 0.45 ~ lower chest / mid torso). Lower back_frac for a smaller dead zone.
  - `max_speed` (m/s) - the reach-speed cap (default 0.04). `full_scale` (shoulder-widths) - how far past a
    threshold you reach max_speed (default 0.5). `hysteresis` (shoulder-widths) - anti-chatter margin (default 0.08).
- `gain: {lateral, vertical, reach}` metres per shoulder-width: raise for more range per hand motion, lower if a
  control clamps on the box too often (reach gain is used only in `reach_mode: position`).
- `sign: {lateral, vertical, reach}`: flip if a control drives the wrong way (camera / handedness); `sign.reach`
  flips the reach direction in both modes.
- `one_euro: {min_cutoff, beta}` on the normalized hand coords: lower min_cutoff = steadier but laggier.
- `deadband` (shoulder-widths), `ramp_s`, `drop_below_hip_s`, `conf_gate` as above. `side` picks the control hand.

## Live tuning (new 2026-10-24) - signs/gains/thresholds were set from recordings, so tune them live
The plane knobs (reach speed, lateral/vertical sensitivity, reach thresholds, dead zone, sign flips) were all set
from recordings. The joystick-rate reach has NEVER been driven live, so tomorrow's first session is a TUNING
session: change a knob, feel it, change it again. Every change is bounded + clamped, speaks a confirmation, shows in
the HUD, and can be saved. Nothing here can exceed the profile speed/box limits.

Body-client keys (plane mode):
- `+` / `-`  reach faster / slower (max_speed x/÷1.25, clamped 0.01..0.08 m/s)
- `]` / `[`  more / less sensitive (lateral gain x/÷1.2 in 0.05..0.35; vertical in 0.05..0.30)
- `.` / `,`  reach goes forward / back sooner (threshold fracs ±0.05 in 0.03..0.60, dead zone kept non-zero)
- `g`  guided tune check    `o`  save tuning    `0`  reset tuning to the profile defaults

Voice (instant, local; never reaches the cloud, never steals a jog like "move left 5 cm"):
- "faster reach" / "slower reach", "more sensitive" / "less sensitive",
  "reach forward sooner" / "reach back sooner", "bigger dead zone" / "smaller dead zone",
  "flip left right" / "flip up down" / "flip reach", "save tuning", "reset tuning", "tune check".

A sign flip re-anchors at the current target, so flipping mid-follow never jumps the arm.

Compact HUD / client status line: `REACH >> | spd4.0 lat.20 vrt.15 db.015` (reach zone, reach speed cm/s, lateral
and vertical gains, position dead band).

Guided tune check (`g` / "tune check", ~20 s): prompts you to (1) raise your LEFT hand to shoulder height, (2) lower
it to your waist, (3) reach your RIGHT hand out to the side, samples each held pose, and sets the forward/back reach
thresholds from YOUR measured heights (as fractions of your shoulder-to-hip span). It reports the measured ranges;
the fit applies live and is kept when you "save tuning".

Save / reset / persistence: `o` / "save tuning" writes `data/profiles/plane_tune_<arm>.json` (gitignored); the app
loads it at startup if present, so your tuning survives a stack restart. `0` / "reset tuning" restores the profile
baseline (does not delete the file; save again to persist the reset).

## Tuning session: 5-minute procedure for tomorrow
1. Bring up the sim stack on the real profile (mock/isaac only; never drives the real arm):
   `rover-stack on --arm dev_arm_real`, start the body client on the laptop (`python body_client.py --side right`,
   add `--mirror` if the webcam mirrors). Face the camera, whole upper body in frame.
2. Engage: press **F** (hold still ~2 s to calibrate), the arm starts following.
3. Fix a wrong direction FIRST. Move your right hand right - gripper should go right; raise it - gripper up. If
   either is reversed, say "flip left right" or "flip up down" (or press **M** if the whole webcam is mirrored).
   Raise your LEFT hand - reach should go FORWARD; if it goes back, say "flip reach".
4. Set the reach feel with the guided check: press **g** (or say "tune check") and follow the three prompts
   (left hand to shoulder, to waist, right hand out). The dead zone and thresholds are now sized to YOUR body.
5. Dial it in by feel: reach too slow/fast -> `+` / `-` ("faster/slower reach"); hand motion too touchy or too
   stiff -> `[` / `]` ("less/more sensitive"); reach triggers too late/early -> `,` / `.` ("reach back/forward
   sooner"); drifting when you hold still -> make the dead zone bigger. Watch the `REACH >>` line and the ghost.
6. When it feels right, press **o** (or say "save tuning"). It is written to the gitignored profile and reloads on
   the next stack start. If you overshoot, press **0** ("reset tuning") to go back to the profile defaults and redo.

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
- The rate-reach thresholds, speed and dead-zone feel are **unit-tested and checked on a spare sim app only**; they
  need a person driving the real arm camera-in-front. Start conservative (max_speed 0.04 m/s).
- Replay of the 2026-10-23 live session (3042 visible frames, left hand tracked 100%) through the new rate mapper:
  reach swept the full box 0.470..0.700 m; by zone 70% back / 20% forward / 11% hold (89% commanding a rate). But in
  that session Pranav was driving the OLD position map and kept his left hand hanging low (median reach_up -1.95 sw,
  near his hips), so under the rate thresholds it reads as "back" almost the whole time. The replay is therefore an
  **offline mechanics check, not a feel prediction**: with rate control you hold the hand at mid-chest to hold.
- Lateral + vertical were reported "pretty good" live; unchanged here.
- `scripts/gate.sh` was not run (it needs the live stack on 47100 = master).
