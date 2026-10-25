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
- robot **x (reach)** <- your LEFT hand **THUMB**, as a JOYSTICK (a RATE, not a position - thumb reach new 2026-10-25,
  see "Reach is the left thumb" below). **Thumb up** -> the gripper creeps **forward** at the reach speed; **thumb
  down** -> **back**; **anything else** (open hand, fist, sideways thumb, hand out of view) -> **hold**. (Depth would
  be the obvious reach control, but depth is exactly the noisy axis, so reach rides a clean thumb gesture instead.)
- Relative / clutched (lateral + vertical only): on engage the app captures your right-hand position and the current
  TCP, then `y,z = tcp0 + ramp * gain * (hand_now - hand0)`, One-Euro filtered, deadbanded, clamped to the box.
- gains (metres of TCP per shoulder-width of hand motion): lateral 0.20, vertical 0.15. A comfortable range covers
  the box: lateral +/-1.0 sw -> +/-0.20 m; vertical +/-0.9 sw -> +/-0.135 m. (The reach gain 0.15 only applies in
  the old `reach_mode: position`.)

## Reach is the left thumb (joystick rate), not the hand height (2026-10-25)
Reach used to be a rate from the left-hand HEIGHT (hand above shoulder = forward, below chest = back, a dead zone in
between). That is replaced by a simple thumb gesture on the LEFT (reach) hand:

- **Thumb UP = reach forward, thumb DOWN = reach back, anything else = hold.** The reach runs at a fixed
  **max_speed 0.04 m/s (4 cm/s)** while the thumb is up or down - bang-bang, not scaled by how far anything is - well
  inside the real arm's limits (REAL_DATA.md 4a: the arm runs at half the commanded speed).
- The thumb is read from the reach hand's 21 MediaPipe Hands landmarks (`BodyPose.reach_hand2d`, sent by the body
  client). It reads **up/down only when the thumb is extended, the other four fingers are curled, and the thumb
  points clearly along the image y axis** - so an open hand, a wave, a fist, or a sideways thumb all read as "hold".
  The classifier is a pure function in `src/rover_intent/body/thumb.py` (unit-tested, no camera needed).
- A short **dwell (~0.2 s)** means a direction has to be held briefly before it starts, so a flicker never jogs the
  arm; losing the gesture (or the hand) drops to **hold immediately**. A reversal (up -> down) passes through hold.
- The signed rate is integrated onto the TCP x at engage, clamped to the box. **At a box wall the rate just stops**
  (ghost turns amber) and **reversing moves away immediately** (no wind-up). If the **reach hand is not tracked, reach
  holds** (rate 0). The engage ramp eases the rate in too, so engaging never jerks.
- Status: the app reports `reach_rate` (signed m/s), `reach_state` (`>>` forward / `<<` back / `hold`) and `thumb`
  (`up` / `down` / `none`) so a HUD / body client can show "REACH >> thumb:UP". Keep the old height->x position map
  with `plane.reach_mode: position` (that legacy mode is not thumb-driven).

Why compute the thumb on the VM, not the laptop: the body client sends the reach hand's landmarks and the VM runs the
one, unit-tested classifier + dwell, instead of duplicating the logic inside the standalone client. Old body clients
that do not send `reach_hand2d` simply leave the reach holding (safe, backwards compatible).

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
- The left-hand-raise clutch trigger is DISABLED in plane mode (your left hand/thumb drives reach).
- Engage: press **F**, hold an open RIGHT palm up for ~1 s, or say "follow me". Engaging first calibrates (hold still
  ~2 s), then follows; the gain eases in over a 2 s ramp so the arm never jumps.
- Release: press **F** again, say "stop" (freezes in place), or DROP your control (right) arm - hold the right wrist
  below your hip line for >= 0.7 s and the arm stows to **base** (`on_drop`). Losing tracking for >= 1 s also stows to
  base; a lost link (nobody watching) freezes in place.

## How to stand / camera placement
Camera in FRONT of you, about chest height, far enough back that your whole upper body is in frame - both shoulders,
both hips, and both hands. Face the camera. The RIGHT hand steers lateral + vertical. The LEFT hand drives reach with
the **thumb**: **thumb up to creep forward, thumb down to pull back, and any neutral hand (relaxed, open, or fist)
holds**. If the left hand drops out of view the reach simply holds. If the arm moves left/right the wrong way, your
webcam is mirrored: press **M** in the body client (or flip `plane.sign.lateral`). The thumb up/down reading is not
affected by the mirror.

## Tuning (configs/arms/dev_arm_real.yaml -> `plane`)
- `reach_mode: rate | position` - `rate` (default) is the thumb reach above; `position` restores the old height->x
  position map (uses `gain.reach`; not thumb-driven).
- `reach_rate:` (rate mode only)
  - `max_speed` (m/s) - the reach speed while the thumb is up/down (default 0.04).
  - `dwell_s` (s) - how long the thumb must hold before a direction starts (default 0.2, anti-flicker).
  - `forward_frac` / `back_frac` / `full_scale` / `hysteresis` - **LEGACY** (the old height thresholds). Kept so old
    saved tuning files still load and the snapshot/HUD stay shaped, but the thumb reach does not read them.
- `gain: {lateral, vertical, reach}` metres per shoulder-width: raise for more range per hand motion, lower if a
  control clamps on the box too often (reach gain is used only in `reach_mode: position`).
- `sign: {lateral, vertical, reach}`: flip if a control drives the wrong way (camera / handedness); `sign.reach`
  flips the reach direction (thumb up then pulls back) in both modes.
- `one_euro: {min_cutoff, beta}` on the normalized hand coords: lower min_cutoff = steadier but laggier.
- `deadband` (shoulder-widths), `ramp_s`, `drop_below_hip_s`, `conf_gate` as above. `side` picks the control hand.

## Live tuning - speed, sensitivity and sign flips
The plane knobs were set from recordings, so tune them live: change a knob, feel it, change it again. Every change is
bounded + clamped, speaks a confirmation, shows in the HUD, and can be saved. Nothing here can exceed the profile
speed/box limits.

Body-client keys (plane mode):
- `+` / `-`  reach faster / slower (max_speed x/÷1.25, clamped 0.01..0.08 m/s)
- `]` / `[`  more / less sensitive (lateral gain x/÷1.2 in 0.05..0.35; vertical in 0.05..0.30)
- `o`  save tuning    `0`  reset tuning to the profile defaults

Voice (instant, local; never reaches the cloud, never steals a jog like "move left 5 cm"):
- "faster reach" / "slower reach", "more sensitive" / "less sensitive",
  "flip left right" / "flip up down" / "flip reach", "save tuning", "reset tuning".

A sign flip re-anchors at the current target, so flipping mid-follow never jumps the arm.

The old height-reach knobs are now **no-ops** (reach is the thumb): "reach forward/back sooner", "bigger/smaller dead
zone" and "tune check" still parse (so an old body client's `,` `.` `g` keys and these phrases do not error) but they
just reply a short note that reach is the left thumb now. They are no longer advertised in the client help line.

Compact HUD / client status line: `REACH >> thumb:UP | spd4.0 lat.20 vrt.15` (reach arrow, detected thumb, reach
speed cm/s, lateral and vertical gains).

Save / reset / persistence: `o` / "save tuning" writes `data/profiles/plane_tune_<arm>.json` (gitignored); the app
loads it at startup if present, so your tuning survives a stack restart (old files with the legacy threshold/dead-zone
fields still load; those fields are ignored by the thumb reach). `0` / "reset tuning" restores the profile baseline.

## Tuning session: procedure
1. Bring up the sim stack on the real profile (mock/isaac only; never drives the real arm):
   `rover-stack on --arm dev_arm_real`, start the body client on the laptop (`python body_client.py --side right`,
   add `--mirror` if the webcam mirrors). Face the camera, whole upper body in frame.
2. Engage: press **F** (hold still ~2 s to calibrate), the arm starts following.
3. Fix a wrong direction FIRST. Move your right hand right - gripper should go right; raise it - gripper up. If
   either is reversed, say "flip left right" or "flip up down" (or press **M** if the whole webcam is mirrored).
   Thumb up with your LEFT hand - reach should go FORWARD; if it goes back, say "flip reach".
4. Dial it in by feel: reach too slow/fast -> `+` / `-` ("faster/slower reach"); right-hand motion too touchy or too
   stiff -> `[` / `]` ("less/more sensitive"). Watch the `REACH >> thumb:UP` line and the ghost.
5. When it feels right, press **o** (or say "save tuning"). It is written to the gitignored profile and reloads on
   the next stack start. If you overshoot, press **0** ("reset tuning") to go back to the profile defaults and redo.

## Try it live
```
# VM: bring the sim stack up on the real profile (mock/isaac backend only; never drives the real arm)
rover-stack on --arm dev_arm_real
# laptop: the body client (add --mirror if your webcam mirrors; M toggles)
python body_client.py --side right        # or --tunnel host:port if UDP can't reach the VM
```
Press **F** (hold still ~2 s to calibrate), then move your hands. Thumb up/down on your left hand to reach
forward/back. Drop your right arm (wrist below your hips) or say "stop" to stop. The green ghost in Isaac shows where
the gripper is heading (amber = clamped at the box edge).

## Not verified until a live run
- The thumb reach is **unit-tested only** (classifier on synthetic landmark sets, mapper reach on synthetic poses).
  It needs a person in front of a camera to confirm MediaPipe reports the reach-hand thumb reliably (lighting, hand
  size, the hand picked near the non-control wrist) and that the ~0.2 s dwell + fixed 4 cm/s feel right. Start
  conservative (max_speed 0.04 m/s).
- The body client change (sending `reach_hand2d`) is **required** for the feature: the laptop must pull this branch.
  An un-updated client still works but the reach just holds.
- Lateral + vertical were reported "pretty good" live; unchanged here.
- `scripts/gate.sh` was not run (it needs the live stack on 47100 = master).
