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
so there it is F / palm only). Both use a plain open/close gripper (no AUTO grab-assist) driven by two sources at
once - see the gripper section below.

## Gripper: two sources at once (hand + Muse)

The gripper receiver takes BOTH a hand source and a Muse source and merges them into one state, **last event wins**:

- **Hand** (`gripper.sources` includes `hand`; `body.grip_from_hand` is the backward-compatible alias): the hand
  feeds **edge** events. A debounced **open -> fist** transition is a **CLOSE**; **fist -> open** is an **OPEN**. The
  openness is EMA-smoothed, sits in a 0.30 / 0.65 dead band, and the new posture must be held `HAND_HOLD_S` (0.6 s)
  before the edge fires (so flicker never spams). Because it is an edge and not a level, **a held fist no longer
  re-closes the gripper after the Muse (or a voice command) has opened it** - it stays open until you actually open
  your hand and make a fist again.
- **Muse** (`gripper.sources` includes `eeg`): a double jaw clench sends a one-shot `GripToggle` that **flips** the
  current state. (The EEG motor-imagery `p_grasp` decoder is the same source.)

Restrict with `gripper.sources: [hand]` or `[eeg]`. The app records which source made the last change and shows it
everywhere: the HUD reads "gripper CLOSED (muse)" / "(hand)", and the Query state / status carry `grip_source`.

**Clash policy** (`gripper.coalesce_s`, default 0.6 s) stops the two sources fighting when they land together: a Muse
toggle within the window **after** a hand edge is read relative to the state **before** that edge, so a fist plus a
near-simultaneous clench is one CLOSE, never a CLOSE then a flip back to open; a hand edge after a Muse toggle that
already reached the same state is a no-op; two **opposite** intents inside the window are a logged **clash** (both
sources named) where the later event wins; outside the window everything is normal. The clash count is in the status,
the Query state, and the logs.

Assisted grab/release (fist near an object -> AUTO pick) still applies only in the non-plain modes where it is
reachable; it is unchanged by the dual-source merge.

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
