# rover-intent

**Drive a 6-axis rover arm with your body, your voice and (experimentally) your brain.**
A webcam tracks your arm and drives the gripper in the plane you see on screen, NVIDIA Nemotron on Nebius Token
Factory (with an instant local fast path) turns speech into structured robot commands, a Muse 2 jaw-clench toggles
the grip, and an EEG decoder estimates grasp intent. Everything is developed against a physics simulation of the
team's real arm in NVIDIA Isaac Sim / Isaac Lab, and the same joint stream, through the same safety layer, now drives
the physical arm.

Built by UBC Rover for the **Nebius × NVIDIA Global AI Hackathon** (Physical AI track).

> Status (2026-10): the simulation stack, teleop, calibration, voice and the EEG research are working and tested in
> sim, and the **physical dev arm is now driven**: the sim streams joint targets to a follower
> (`scripts/sim_to_real.py`) that feeds Rowan's ROS 2 Jazzy `stepper_arm_driver`, which drives the real arm through
> the same safety layer under the `dev_arm_real` profile. Voice presets and jogs have been run on the real arm; it
> tracks the sim to within about 0.5 deg after settling. Numbers below are from simulation or offline datasets unless
> they say "real arm".

---

## Contents
- [What it does](#what-it-does)
- [System architecture](#system-architecture)
- [How the body tracking works](#how-the-body-tracking-works)
- [Calibration: syncing you with the robot](#calibration-syncing-you-with-the-robot)
- [Clutch and safety](#clutch-and-safety)
- [Voice: Nemotron on Nebius Token Factory](#voice-nemotron-on-nebius-token-factory)
- [Autonomous skills](#autonomous-skills)
- [The arm and the simulation](#the-arm-and-the-simulation)
- [EEG grasp-intent research](#eeg-grasp-intent-research)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Running it](#running-it)
- [Testing and the regression gate](#testing-and-the-regression-gate)
- [Configuration](#configuration)
- [Network ports](#network-ports)
- [Results so far](#results-so-far)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [License](#license)

---

## What it does

| Input | What you do | What the arm does |
|---|---|---|
| **Body** (webcam) | move your right hand (default **plane** mode: image-plane only) | the gripper follows left/right and up/down in the frame you see; the left hand drives a joystick-rate reach |
| **Hand** | make a fist / open your hand | the gripper closes / opens |
| **Voice** | "move left 10 centimetres", "go to ready", "stow then home" | the instant local fast path runs it in the same tick, no cloud round trip |
| **Voice** | "put the red cup next to the tall bottle" | Nemotron parses it; the arm plans and executes the pick-and-place |
| **Voice** | "stop", "hold on", "reset" | handled locally and instantly (never waits on the cloud) |
| **Muse 2** (EMG) | double jaw-clench | toggles the gripper open/closed (`eeg/clench.py`) |
| **EEG** (research) | imagine / prepare a grasp | a decoder estimates grasp intent (offline and pseudo-online results below) |

Visual feedback is built for one screen (streamed to the operator over Moonlight): the Isaac view shows a **ghost
target** with a line from the gripper, the **reachable box**, a **neutral cross** and **calibration targets**; a HUD
overlay shows a mirror-view skeleton, clutch state, prompts and link latency.

---

## System architecture

```
 OPERATOR LAPTOP                                        SIMULATION / CONTROL HOST (GPU)
 ───────────────                                        ───────────────────────────────
 ┌──────────── laptop/body_client.py ──────────────┐
 │ webcam → MediaPipe Pose (33 pts, 3-D, metres)   │
 │        → MediaPipe Hands (fist / open)          │
 │  right-arm lock · glitch gate · stand-here guide│
 │  keys: C calibrate · X redo axis · F clutch     │
 └──────────────┬──────────────────────────────────┘
                │ 30 Hz JSON: shoulders, elbow, wrists, hips, hand state, seq
 ┌──────────────┴───────────┐
 │ laptop/voice_client.py   │ push-to-talk → local Whisper → transcript
 └──────────────┬───────────┘
                │ UDP (tailnet) ─── or ─── SSH tunnel → scripts/tcp_relay.py
                ▼
 ┌──────────────────────── rover_intent.app (50 Hz control loop) ─────────────────────────┐
 │  pose ─► Clutch ─► CalibratedMapper (profile) ─► gripper target ─┐                     │
 │  speech ─► local stop/reset ─┐                                   │                     │
 │         └► Nemotron (Nebius Token Factory, JSON-schema output) ─► Intent ─► Skills     │
 │                                                                  ▼                     │
 │                        Arbiter (follow / auto / hold, shared autonomy, grip)           │
 │                                     │ straight-line "carrot" target                    │
 │                        DLS inverse kinematics (URDF)                                   │
 │                                     │ joint targets                                    │
 │                        SafetyLayer: joint limits · vel/acc caps · reach box ·          │
 │                                     watchdog · e-stop · anti-windup                    │
 └─────────────────────────────────────┼──────────────────────────────────────────────────┘
                                       │ UDP: joints, gripper, ghost, targets
                                       ▼            ▲ joints, object poses, camera pose
 ┌──────────────── sim/isaac_bridge.py (Isaac Lab, 120 Hz physics, CPU PhysX) ────────────┐
 │  arm + gripper + cup / block / bottle · ghost + line · reach wireframe · neutral cross │
 └────────────────────────────────────────────────────────────────────────────────────────┘
   scripts/sim_to_real.py follower ─► Rowan's ROS 2 Jazzy stepper_arm_driver ─► Teensy 4.1 stepper firmware ─► real dev_arm
```

The real arm is driven by mirroring the simulated joint targets: `scripts/sim_to_real.py` reads the app's joint
stream and republishes it (at a conservative speed fraction) to Rowan's `stepper_arm_driver` over ROS 2 Jazzy, so
the sim and the hardware run the same trajectory through the same safety layer. `scripts/real_to_sim.py` mirrors the
other way (the real arm's measured joints drive the sim view). See [The arm and the simulation](#the-arm-and-the-simulation).

Design rules that shaped the code:
- **Only decisions go to the cloud.** Nemotron decides *what* to do; motion, IK, safety and the control loop stay
  local and deterministic.
- **Everything the arm executes goes through one safety layer**, whatever the source (body, voice, EEG).
- **Same code path in sim and on hardware**: the app talks to an `ArmBackend` (mock / Isaac / real).
- **Measured, not assumed**: workspace, home pose, grasp heights and camera views were chosen from IK reach maps
  and live checks; every claim in this README has a test or a logged run behind it.

---

## How the body tracking works

```
 webcam frame
   │
   ▼
 MediaPipe Pose ──► world landmarks: 3-D points in metres, centred on the hips
   │                (x image-right, y down, z depth; depth is the noisiest axis)
   ├─ arm:   right shoulder (12) · elbow (14) · wrist (16)
   ├─ hand:  MediaPipe Hands finds up to 2 hands; keep the one whose wrist sits on the
   │         tracked arm (handedness labels are not trusted: they flip on mirrored webcams)
   ├─ grip:  fingertip spread / palm size → 0 (fist) … 1 (open); "not found" never means "open"
   ├─ gate:  wrist jumps faster than 6 m/s are tracking glitches → skipped (≤ 6 frames)
   └─ guide: shoulder width in pixels (distance) + torso centre (left/right, height)
             → the stand-here outline turns green → in_position
```

The client sends 30 packets/s with a sequence number; the app echoes the latest one so the client can show the real
round-trip time (excluding the app's own hold time) and packet loss.

### Teleop modes

- **`plane`** (default on the real arm, [`docs/PLANE_TELEOP.md`](docs/PLANE_TELEOP.md)): the camera sits in front of
  you and only the image plane is used. The right hand moves the gripper left/right and up/down; the left hand is a
  joystick-rate reach (hold it high to creep forward, low to pull back, a dead zone in between). Depth, the noisiest
  MediaPipe axis, is ignored, so there is no calibration capture. This is the mode behind the real-arm sessions.
- **`arm_mirror`** ([`docs/ARM_MIRROR.md`](docs/ARM_MIRROR.md)): full-arm joint mirroring, no IK. Your shoulder,
  elbow and wrist angles map straight to the arm's joints. Also has no calibration capture.

The calibrated-mapper path below (a measured body-to-robot frame fit) is used by the older mapping mode and is kept
for reference; `plane` and `arm_mirror` do not load a calibration profile.

---

## Calibration: syncing you with the robot

Instead of assuming axes, the operator's frame of reference is measured and fitted to the camera they are looking at.

```
 1. BODY FRAME (independent of where the webcam is)
        up (z) = hips → shoulders
          ▲    forward (y) = z × x  (toward the webcam)
          │   ╱
   L ─────●────► right (x) = left shoulder → right shoulder      hand = wrist − right shoulder
          R shoulder (origin)

 2. GUIDED CAPTURE (~15 s; prompts in the HUD, webcam window and a speech bubble)
    NEUTRAL 2 s → RIGHT → UP → "REACH TOWARD THE CAMERA" → LEFT → DOWN   (1.5 s each)
    a hold is rejected if it jitters > 2 cm or moved < 8 cm from neutral;
    capture only runs while the stand-here guide is green

 3. FIT
    rotation R (body → command frame) by Kabsch on the measured unit directions
    command frame = the CURRENT camera's heading, gravity kept vertical:
        your right → screen-right (level) · your up → up · your forward → into the scene (level)
    per-direction scale: your comfortable reach → the wall of the reachable box (ray from the centre)
    neutral hand → the box centre (white cross)
    camera changes later? the command frame is recomputed; no re-capture

 4. VERIFY: touch 4 magenta targets with the gripper (within 3 cm, held 1 s)
    accepted → profile saved (data/profiles/<arm>_<name>.json) and reused next session
    not accepted → the worst axis is reported; X re-captures only that axis
```

Every clutch-in afterwards is a **quick re-centre** (your pose at that moment becomes neutral) followed by a slow
**glide** (12 cm/s) to the mapped point, so the arm never jumps.

Implementation: [`body/calibration.py`](src/rover_intent/body/calibration.py) (maths),
[`body/calib_session.py`](src/rover_intent/body/calib_session.py) (session + mapper).

---

## Clutch and safety

```
              F  /  open palm held UP 1 s  /  LEFT hand raised (0.3 s, debounced)
 ┌──────────┐ ───────────────────────────────────────────────► ┌──────────────┐
 │  FROZEN  │                                                  │  RE-CENTRE   │
 │ arm holds│ ◄─── moved > 3 cm: countdown restarts            │  hold still  │
 └──────────┘                                                  └──────┬───────┘
     ▲   ▲   F again / left hand down 0.5 s                           │ 2 s
     │   └──────────────────────────────────────┐                     ▼
     │   no packets 0.5 s  ("link lost")        └────────────── ┌──────────────┐
     └── arm not visible 1 s ("tracking lost") ──────────────── │  FOLLOWING   │
                                                                └──────────────┘
 During calibration capture the gesture triggers are ignored (the "UP" pose must not move the arm).
```

The `SafetyLayer` sits between every command source and the arm:
- joint position limits from the URDF; per-joint velocity and acceleration caps (discrete braking curve,
  stress-tested over 20 000 random steps: never exceeded);
- a Cartesian reach box measured from IK (targets outside are clamped, and the ghost turns amber);
- an input watchdog that brakes to a stop when a source goes stale; a latching e-stop;
- anti-windup: the commanded trajectory never runs more than 0.25 rad ahead of the measured joints.

---

## Voice: Nemotron on Nebius Token Factory

```
 "could you set the little yellow one down next to the red cup"
     │ laptop: Whisper (local)                       stop / hold on / reset → handled locally, instantly
     ▼
 app → Nebius Token Factory (OpenAI-compatible) → nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B
       system prompt: the scene's objects with descriptions + one line per action + "clarify"
       response_format: JSON schema of Intent
     ▼
 {"action": "move", "object": "block", "relation": "beside", "reference": "cup", "say": "..."}
```

- **Instant local fast path** (`src/rover_intent/planner/fast_path.py`): a strict local parser runs in the control
  loop *before* the cloud call and handles the short commands with no round trip (< 50 ms): named poses
  (stow / ready / go back / home ...), Cartesian jogs ("move left 10 cm", "forward a bit"), joint jogs ("turn the
  base 20 degrees"), speed ("slower" / "normal speed"), short sequences ("stow then home"), and context ("again",
  "the other way"). Anything it does not fully understand returns `None` and falls through to Nemotron, so it can
  never misfire on a sentence like "put the cup left of the bottle".
- The cloud call runs **off the control loop** (worker thread); a later "stop" cancels pending cloud results.
- No hand-written language rules or object whitelists: the model resolves colours, synonyms and ambiguity itself
  and may answer `clarify` ("which one?"); the skill layer still refuses objects it cannot see.
- On 7 test phrases Nano parsed 7/7 and Super 5/7 (Super heard "whoa whoa hold on" as *grasp*), so Nano is the
  default. Typical latency 1.5-4 s.
- **On the real-arm profile** (`dev_arm_real`) named poses and the no-camera jogs/sequences/speed move at once, with
  no spoken "yes" (the two-step confirm was too slow on the real mic); the pick/place **skills stay confirm-gated**.
- Without `NEBIUS_API_KEY` an offline keyword parser keeps the loop usable. Full latency/accuracy table in
  [`docs/VOICE.md`](docs/VOICE.md).

---

## Autonomous skills

`pick`, `place`, `move` (pick + place), `grasp`, `release`, `home`, `stop`, `follow`, `reset`.

```
 pick X:   above X at travel height → pre-grasp (settle sideways ≤ 3 mm, 0.3 s) → straight down at 5 cm/s
           → close (0.8 s dwell) → up to travel height
 place:    above goal → pre-place (settle) → straight down → open → up
 travel height = tallest object top + 8 cm;  tall objects are grasped near the top (palm clearance rule)
```

The arbiter feeds the IK a **straight-line "carrot"** that slides toward each waypoint (0.3 m/s travel, 5 cm/s final
approach). This fixed a real bug: jumping the IK target 10 cm down produced a curved joint path that swung the gripper
13 mm sideways into the rim of a 70 mm bottle (7.5 mm clearance per side).

---

## The arm and the simulation

Three arm profiles are supported (`configs/arms/*.yaml`): **dev_arm** (sim), **dev_arm_real** (the same arm tuned for
the physical hardware), and **rover2026**.

| | **dev_arm** (the team's 2023 / development arm) | rover2026 |
|---|---|---|
| source | team description `dev_arm_description` | 2025-26 SolidWorks export |
| joints | 6 revolute, base yaw ±86° | 6 revolute, base yaw ~327° |
| tool | +z of `link_6`, TCP 0.20 m | +x of `a6_EE_holder`, TCP 0.145 m |
| actuation (real) | steppers + Teensy 4.1, USB serial | Maxon motors, moteus over CAN |
| reach box (top-down, from home) | x 0.40-0.62, y ±0.40, z 0.03-0.32 m | x 0.35-0.80, y ±0.40, z 0.15-0.45 m |
| objects | on the arm's own surface | on a 12 cm table |

### Driving the real arm (`dev_arm_real`)

The `dev_arm_real` profile is the sim dev arm retuned from bench measurements (2026-10-23) so the hardware can follow
the sim safely:
- home is Rowan's return-to-base rest pose, and the workspace box is moved up and in to contain it and sit inside the
  real reachable shell;
- the firmware runs a velocity command at **about half** the commanded speed (bench sweep), so the per-joint velocity
  caps are dropped to a conservative fraction of the firmware max and the follower defaults to the matching speed, so
  the sim never outruns the arm;
- there is a measured **0.17-0.31 s command lag** before the near-instant ramp, so a gentler sim ramp is used.

`scripts/record_real.sh` records a raw session, `scripts/real_bag_to_csv.py` converts it to CSV, and the
`scripts/bench/` sweeps (axis, drift, end-stop) produced the numbers above. See
[`docs/REAL_DATA.md`](docs/REAL_DATA.md) and [`docs/BENCH_SESSION.md`](docs/BENCH_SESSION.md).

Simulation details:
- Isaac Sim 6.0.1 + Isaac Lab 3.0, standalone script, CPU PhysX for a single arm (≈5× faster per step than the GPU
  pipeline at this scale), rendering gated to every 4th step, global-clock pacing → **1.00× real time** under load.
- Grasps are physical (friction, finger force limits); no attach/detach shortcuts.
- Arm links ignore gravity (perfect gravity compensation); stiff PD position drives stand in for the steppers.
- The bridge reports object poses (sim ground truth stands in for perception) and its camera pose (used by the
  calibration).
- The upstream gripper geometry of the dev arm was unusable in sim (raw CAD offsets); a placeholder parallel gripper
  (85 mm opening) sits at the upstream tool frame until the real end effector is measured.

---

## EEG grasp-intent research

Offline and pseudo-online studies on public datasets, using 8 motor-strip channels (what an 8-channel OpenBCI board
can cover). Chance is 50 %.

**PhysioNet EEG Motor Movement/Imagery** (109 subjects, imagined fist clench vs rest):

| model | calibrated (within-subject CV) | new person (leave-subjects-out) |
|---|---|---|
| CSP + LDA | 68.5 % | 58.5 % |
| Riemannian tangent space + LR | **69.5 %** | **60.2 %** |

**BNCI Horizon 001-2020** (15 subjects, real self-paced reach-and-grasp, causal filters):

| contrast | features | calibrated | new person |
|---|---|---|---|
| 1 s before the reach vs in-run idle | mu/beta | 74.4 % | 66.1 % |
| 1 s before the grasp vs idle | mu/beta | **86.2 %** | **75.3 %** |
| hand closing vs reach start (arm moving in both) | MRCP | **82.1 %** | 72.8 % |

Checks: eye-movement channels alone are weak for intent (60 %); regressing out EOG barely changes the results;
rest blocks recorded at the end of a session inflate accuracy (96 %), so only in-run comparisons are quoted.

**Hybrid decoder** (mu/beta "grasp coming" × MRCP "close now", streaming causal filters, pseudo-online through the
real arbiter): at ≤ 1 false close per idle minute it catches only ~19 % of grasps (calibrated) and ~5 % (new person).
Window accuracy does not survive continuous use yet; context gating (only listen near an object) is the next step.
EEG therefore *assists* grasping in this project; it is not the primary trigger.

### Muse 2: double jaw clench toggles the gripper

We have a Muse 2 headband. Its 4 dry sensors (TP9, AF7, AF8, TP10) sit behind the ears and on the forehead, with
nothing over the motor strip, so the imagined-grasp decoders above can't run on it. What it picks up very well is
**jaw-muscle (EMG) activity**, so the Muse drives a hands-free **grip toggle**: clench twice quickly to close the
gripper, twice again to open it. It is a muscle signal from the headband, not decoded brain activity, and we label it
that way.

How it works (`src/rover_intent/eeg/clench.py`): each channel is notch-filtered (60 Hz), band-passed 30-100 Hz and
turned into a loudness envelope. The score is the median over channels of loudness / that channel's relaxed level, so
one loose electrode can't trigger it. A toggle needs **exactly two** short clenches with the second starting within
0.8 s of the first, then 0.5 s of quiet. Three or more in a row count as chewing and are ignored; a clench held over
1.5 s is ignored. The detector sends a one-shot `GripToggle` message (not a level), so it never fights voice
"release", resets or grab assists. In FOLLOW mode a toggle behaves like a held fist / open hand: near an object it
starts the grab assist, while holding it starts the release.

```bash
# laptop with the Muse (Bluetooth on); clench.py also works as a single copied file
pip install numpy scipy brainflow
python -m rover_intent.eeg.clench calibrate --out data/profiles/clench_pranav.json   # ~1.5 min, guided:
#   relax 20 s -> 8 x "CLENCH" -> talk + chew 20 s; prints GOOD / MARGINAL / POOR
python -m rover_intent.eeg.clench run --profile data/profiles/clench_pranav.json --host <vm>   # --dry to just print
python -m rover_intent.eeg.clench replay --profile ... --file data/eeg_recordings/clench_*.npz  # offline re-check
```
Set `body.grip_from_hand: false` while using it (otherwise an open hand re-opens the gripper). `--board muse2-bled
--serial COM3` uses a BLED112 dongle instead of native Bluetooth.

### Muse 2: genuine-EEG controls (research, offline)

Jaw clench is EMG, not brain activity. For a real brain-signal control we characterised two options offline on public
datasets, restricted to the electrode sites nearest the Muse's, before the headband arrives. Full tables, caveats and
a day-one checklist are in [`docs/muse_readiness.md`](docs/muse_readiness.md).

- **Alpha / eyes-closed** ("close your eyes to confirm") works on the Muse's own electrodes: on EEGMMIDB (109
  subjects, chance 50 %) the Muse-proxy sites reach ~0.84-0.86 balanced accuracy calibrated and ~0.73-0.76 cold at a
  2-3 s window. This is the recommended day-one control.
- **SSVEP** ("look at a flickering box") is near chance on the Muse's temporal electrodes (Wang2016, 9 subjects:
  2-class <= 0.68), but a single occipital electrode takes it to ~0.94 (2-class) / ~0.90 (4-class). It is the stretch
  goal, only if the Muse AUX port can drive an occipital electrode (unverified).

Tools (standalone on the laptop, `pip install -e '.[muse]'`): `eeg/muse_session.py` records a guided ~10 min session
(alpha blocks + frame-locked SSVEP flicker, `--check-display` to test the screen), `eeg/muse_report.py` turns it into
a readiness report, and `eeg/muse_detect.py` runs the live SSVEP/alpha detector and emits a confirm/grab over UDP.

Results so far are on **synthetic** Muse-like signals only (assumed levels: a clench is tens to hundreds of uV of
EMG vs a few uV of background above 30 Hz): 60/60 random double clenches caught (relax gaps down to 0.15 s), 0 false
toggles in 10 min of talking + blinking + chewing, toggle fires ~1.6 s after the first clench starts. Not yet run on
the real headband.

**Day one, one command** (`eeg/sunday.py`, see [`docs/SUNDAY.md`](docs/SUNDAY.md)): `python -m rover_intent.eeg.sunday
--board muse2 --name pranav` records a short labelled session (signal-quality gate, rest, single/double clenches,
negatives, eyes-open/closed), fits YOUR clench + eyes-closed-alpha calibration on a held-out split, writes the
profiles where the tools load them (old ones backed up), and runs a live double-clench -> gripper toggle into the SIM
(spare port, never the real arm). It is resumable and re-runs a single block with `--only <stage>`. Saturday dry run
with no headband: `python -m rover_intent.eeg.sunday --preflight` (synthetic board, end to end, < 3 min). Your EEG
stays on local disk and is never committed or uploaded.

---

## Repository layout

```
rover-intent/
├── configs/
│   ├── default.yaml            shared settings (loop, network, body mapping, Nemotron, EEG, clutch)
│   └── arms/{dev_arm,dev_arm_real,rover2026}.yaml   arm-specific: URDF, joints, home, limits, reach box, scene, modes
├── src/rover_intent/
│   ├── app.py                  the 50 Hz control loop wiring everything together
│   ├── config.py               profile-aware config loader (ROVER_ARM / --arm)
│   ├── types.py                message types (also the laptop <-> host JSON)
│   ├── body/                   plane (image-plane teleop), arm_mirror (full-arm joint mirror), clutch,
│   │                           calibration (frame, capture, Kabsch fit, profile), calib_session (session + mapper)
│   ├── control/                kinematics (URDF FK, Jacobian, DLS IK), safety, arbiter, stepper_map
│   ├── planner/                fast_path (instant local parser), nemotron (intent parser), scene, skills
│   ├── robot/backends.py       MockArm · IsaacArm (UDP) · RealArm (unused; real arm driven via scripts/sim_to_real.py)
│   ├── transport/udp.py        JSON-over-UDP link
│   └── eeg/                    datasets, motor-imagery training, reach-and-grasp studies, hybrid decoder, clench, live
├── sim/
│   ├── isaac_bridge.py         Isaac Lab scene + UDP bridge, markers, camera views
│   ├── make_dev_arm_urdf.py    dev arm + placeholder gripper -> URDF
│   ├── make_gripper_urdf.py    rover2026 + gripper -> URDF
│   └── import_gripper.py       URDF -> USD (Isaac Sim importer)
├── ros/servo/                  MoveIt Servo on the simulated arm (experimental; sim / mock only)
├── laptop/                     body_client.py (webcam), voice_client.py (Whisper) - standalone single-file scripts
├── scripts/                    sim_to_real.py / real_to_sim.py (real-arm follower + mirror), record_real.sh +
│                               real_bag_to_csv.py (recorder + CSV), bench/ (characterization), tcp_relay.py, hud.py,
│                               check_grasps.py, gate.sh, replay_session.py, fake_laptop.py, publish.sh
├── assets/                     dev_arm (team model, see SOURCE.md), rover2026
├── tests/                      unit + integration tests (control loop, calibration flow, decoder streaming)
└── docs/                       PLAN.md, VOICE.md, PLANE_TELEOP.md, ARM_MIRROR.md, REAL_ARM.md, REAL_DATA.md,
                                BENCH_SESSION.md, VIDEO_HUD.md, EEG.md, GRIPPER.md
```

---

## Setup

### Simulation / control host (Linux, NVIDIA GPU)
```bash
# Python environment for the app (CPU only)
uv venv .venv && uv pip install -p .venv/bin/python -e ".[dev]"
uv pip install -p .venv/bin/python mne pyriemann scikit-learn joblib scipy   # EEG studies (optional)

# Isaac Sim 6.0.1 + Isaac Lab 3.0 in a separate venv (path used below: ../isaac/isaacenv6)

# build the arm USD once
.venv/bin/python sim/make_dev_arm_urdf.py
OMNI_KIT_ACCEPT_EULA=YES ../isaac/isaacenv6/bin/python sim/import_gripper.py \
    --urdf assets/dev_arm/dev_arm_gripper.urdf --out assets/dev_arm/usd

# Nemotron (optional; offline parser without it)
echo "NEBIUS_API_KEY=..." > .env && chmod 600 .env
```

### Operator laptop (Windows / macOS / Linux)
The laptop scripts are standalone (copy the single file):
```bash
python -m venv rover-body && rover-body/Scripts/activate      # or: source rover-body/bin/activate
pip install mediapipe opencv-python numpy                      # body_client.py (MediaPipe Tasks API)
pip install faster-whisper sounddevice numpy                   # voice_client.py
pip install numpy scipy brainflow                              # eeg/clench.py (Muse 2 jaw-clench toggle)
```

---

## Running it

```bash
# host: Isaac bridge (GPU) + app + relay + HUD
OMNI_KIT_ACCEPT_EULA=YES ../isaac/isaacenv6/bin/python sim/isaac_bridge.py --viz kit   # GUI on :0
.venv/bin/python -m rover_intent.app --backend isaac          # control loop (ROVER_ARM=rover2026 to switch arms)
.venv/bin/python scripts/tcp_relay.py                         # only needed for the SSH-tunnel path
DISPLAY=:0 python3 scripts/hud.py                             # HUD overlay

# laptop (direct UDP)
python body_client.py --host <host-tailnet-ip>
python voice_client.py --host <host-tailnet-ip>

# laptop (when UDP can't reach the host)
ssh -N -L 47120:127.0.0.1:47120 <host>
python body_client.py --tunnel 127.0.0.1:47120

# real arm: run the app on the dev_arm_real profile, then the follower (needs Rowan's stepper_arm_driver up)
.venv/bin/python -m rover_intent.app --backend isaac --arm dev_arm_real
source /opt/ros/jazzy/setup.bash && python3 scripts/sim_to_real.py --speed 0.2   # --dry to print only
```

First session: stand so the guide turns green → **C** (calibrate, ~35 s) → **F** or raise your left hand to follow →
fist to grab, or just say what you want done.

Camera views can be switched live: send `{"view": "behind"|"operator"|"high"|"top"|"side"}` to the bridge's command port.

The body client also streams its annotated webcam to a panel **under** the HUD, so the operator can glance at one VM
screen instead of the laptop window. It is **on by default**; `python body_client.py --no-video` turns it off. The HUD
needs Pillow for the decode (`pip install pillow`); without it the panel shows a hint and the rest of the HUD still
works. See `docs/VIDEO_HUD.md` for the flags, sizing and the UDP-vs-tunnel size note.

---

## Testing and the regression gate

```bash
.venv/bin/python -m pytest -q        # 314 tests: kinematics, safety, parser, fast path, skills, arbiter, clutch,
                                     # calibration, plane / arm_mirror teleop, the full control loop and calibration
                                     # flow over UDP (mock arm), streaming EEG decoder == batch path, sim_to_real
                                     # follower, per-arm IK reach of every skill waypoint
scripts/gate.sh                      # tests + live grasp check in Isaac (every pick and every "move X beside Y")
```

The gate must pass after any change to the scene, home pose, gripper, skills, IK or config. It exists because a
home-pose change once broke the bottle grasp and only a re-run caught it.

---

## Configuration

`configs/default.yaml` (shared) is deep-merged with `configs/arms/<profile>.yaml`; select the profile with
`arm_profile`, the `ROVER_ARM` environment variable or `--arm`. Values marked `PLACEHOLDER` are not real hardware
data yet (accelerations, the real gripper, rover2026 home).

Key settings: `body.mode` (`plane` | `arm_mirror`; `plane` is the real-arm default), `body.clutch`,
`body.grip_from_hand` (set `false` when the Muse clench toggle drives the gripper), `safety.workspace_min/max`,
`arm.max_vel/max_acc`, `nemotron.model`, `confirm.presets/motion` (off on `dev_arm_real` so voice poses and jogs
move at once).

---

## Network ports

| port | protocol | who | what |
|---|---|---|---|
| 47100 | UDP | app | body poses, transcripts, clutch keys, queries in; replies + 10 Hz status out |
| 47110 | UDP | bridge | joint / gripper / marker commands, camera view changes |
| 47111 | UDP | app | joint states, object poses, camera pose from the bridge |
| 47120 | TCP | tcp_relay | newline-JSON ↔ UDP 47100 (for SSH tunnels) |
| 47130 | UDP | HUD | status for the on-screen overlay |
| 47131 | UDP | HUD video | live webcam JPEG stream, forwarded by the app to the video panel under the HUD |
| 47101 | UDP | speech bubble | prompts / replies shown on screen |

---

## Results so far

| what | result |
|---|---|
| grasp gate in Isaac (dev arm) | 9/9: picks lift 23 / 11 / 26 cm, all six "move X beside Y" land within 1.2 cm |
| sim speed | 1.00× real time under load (was 0.82×) |
| **real arm** (dev_arm_real, voice presets + jogs) | tracks the sim to within about 0.5 deg after settling; firmware runs ~half the commanded speed, 0.17-0.31 s command lag (bench sweep) |
| calibration (scripted operator standing 15° off-axis) | fit residual 0.1°, targets 2.0 / 2.1 / 2.2 / 1.8 cm, ~36 s |
| voice parsing (Nemotron Nano) | 7/7 test phrases, 1.5-4 s |
| safety layer | 20 000-step stress test: velocity/acceleration limits never exceeded |
| EEG | see [EEG grasp-intent research](#eeg-grasp-intent-research) |

The real arm has been driven by voice presets and jogs under the `dev_arm_real` profile. The laptop camera HUD panel
and the `plane` left-hand joystick reach are implemented but not yet verified in a live session; everything else above
is simulation, scripted input or offline data.

---

## Known limitations

- **Real arm driven, not yet fully commissioned**: the sim drives the hardware through the `sim_to_real.py` follower
  and Rowan's `stepper_arm_driver` (the `RealArm` serial backend is unused), and voice presets and jogs have run on
  it; broader live operator sessions, per-axis sign/offset refinement and the live camera HUD are still to come.
- **Gripper**: the real end effector's geometry is unknown; a placeholder parallel gripper is used in sim.
- **Monocular depth**: forward/back is the noisiest axis; the calibration's per-axis scales and filtering compensate
  but don't remove it.
- **Mirrored webcams**: the fit absorbs the rotation, but the tracked hand would be the left one; the client's M key
  flips it.
- **Stand-here guide thresholds** are tuned for a laptop webcam at about 1.5 m.
- **EEG** is not reliable enough to trigger grasps alone (see above).
- **Sim shortcuts**: perfect gravity compensation, no stepper missed-step model, object poses from ground truth.

---

## Roadmap

1. Finish commissioning the real arm: per-axis sign/offset refinement, wider workspace, and verifying the camera HUD
   and the `plane` joystick reach in a live session.
2. Live operator sessions with the body teleop; tune the stand-here guide.
3. Context-gated EEG (listen only near an object) and per-user calibration.
4. Perception for the real scene (ArUco / detection) to replace sim ground truth.
5. Demo video and submission.

---

## License

Apache-2.0 (see `LICENSE`). `assets/dev_arm/` is the team's development-arm description from the UBC-Snowbots
RoverFlake2 repository (see `assets/dev_arm/SOURCE.md`).
