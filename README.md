# rover-intent

Control a 6-axis rover arm with your **body** (webcam mirroring), your **brain** (EEG motor imagery for
grasp/release) and your **voice** (Whisper → NVIDIA Nemotron on Nebius Token Factory). Shared autonomy
finishes the approach when your hand gets close to an object.

Built for the **Nebius x NVIDIA Global AI Hackathon** (Physical AI track) by UBC Rover (Ptolemy).
Status: scaffold. See [docs/PLAN.md](docs/PLAN.md) for milestones.

## How it fits together
```
laptop  webcam → MediaPipe ─┐   mic → Whisper ─┐   EEG → BrainFlow → MI decoder ─┐
                            └──── UDP (tailnet) ┴────────────────────────────────┘
VM      Nemotron (Token Factory): text → Intent JSON  →  Arbiter (follow / auto / hold, shared autonomy)
        → DLS IK (URDF) → SafetyLayer (joint limits, vel/acc limits, watchdog, e-stop) → arm backend
        backends: MockArm (tests) · IsaacArm (Isaac Sim 6) · RealArm (team arm)
```
Only the language decision goes to the cloud; the control loop is local.

| Where | What | Code |
|---|---|---|
| NVIDIA Nemotron via Nebius Token Factory | spoken command → structured `Intent` (JSON schema output) | `src/rover_intent/planner/nemotron.py` |
| NVIDIA Isaac Sim 6 / Isaac Lab 3 | simulated rover2026 arm + gripper | `sim/` |
| local | skills (pick/place/move), IK, safety, arbitration | `src/rover_intent/{planner,control}` |
| laptop | body tracking, speech-to-text | `laptop/` |
| local | EEG motor-imagery decoder | `src/rover_intent/eeg/` |

## Quick start (VM / any Linux box)
```bash
uv venv .venv && uv pip install -p .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q                       # kinematics, safety, parser, skills, arbiter
.venv/bin/python -m rover_intent.app --seconds 20 &  # control loop, mock arm
.venv/bin/python scripts/fake_laptop.py --say "follow me" --circle 6 --eeg-grasp-at 3
.venv/bin/python scripts/fake_laptop.py --say "move the cup beside the bottle"
```
Set `NEBIUS_API_KEY` to use Nemotron; without it an offline keyword parser is used.

## Laptop
```bash
pip install -e ".[laptop]"
python laptop/stream_body.py  --host <vm-tailnet-ip> --show
python laptop/stream_voice.py --host <vm-tailnet-ip>
```

## EEG
```bash
uv pip install -p .venv/bin/python -e ".[eeg]"
.venv/bin/python -m rover_intent.eeg.train_mi --subjects 1-20 --out models/mi_riemann.joblib
.venv/bin/python -m rover_intent.eeg.live --model models/mi_riemann.joblib --board synthetic
```

## Prior work
The rover2026 arm URDF/meshes and the Isaac Sim setup existed before the hackathon (UBC Rover, keypad-typing RL
experiments). Everything in `src/`, `laptop/`, `sim/isaac_bridge.py` and the gripper work is new in the submission period.

## License
Apache-2.0
