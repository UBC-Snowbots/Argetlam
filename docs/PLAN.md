# Plan: rover-intent (Nebius x NVIDIA Global AI Hackathon, Physical AI track)

Deadline **2026-10-30 13:00 EDT**. Submission: public repo + OSS license, ≤3 min YouTube video with
**≥1 min of the real arm operating**, description, Nebius/NVIDIA feedback, and a note on what's new vs prior work.

## The idea
One person controls a 6-axis rover arm three ways at once:
**body** (webcam arm mirroring) for where, **brain** (EEG motor imagery) for grasp/release,
**voice** (Whisper → Nemotron on Token Factory) for what. Near an object, shared autonomy
finishes the approach. "Put it beside the bottle" hands over to autonomous execution.

```
 laptop:  webcam → MediaPipe ──┐          EEG → BrainFlow → MI decoder ──┐
          mic → Whisper ───────┤ UDP over tailnet                        │
                               ▼                                          ▼
 VM:      Nemotron (Token Factory) → Intent ─→ Arbiter (follow/auto/hold + shared autonomy)
                                                   ↓ TCP target
                                              DLS IK → SafetyLayer (limits, watchdog, e-stop)
                                                   ↓ joint targets
                                         MockArm | IsaacArm (sim) | RealArm (old team arm)
```
Only the language decision goes to the cloud; the control loop is local.

## Milestones (build in this order; each one is a demo on its own)
| # | Milestone | Done when |
|---|---|---|
| 0 | Scaffold | ✅ tests pass; mock loop runs follow → EEG grasp → voice move (2026-09-28) |
| 1 | **Sim arm + gripper** | ✅ 2026-09-28: isaac_bridge.py + 2F-85-sized gripper; voice 'move X beside Y' does physical grasps (placed within 1–8 mm) |
| 2 | **Body mirroring in sim** | 🟡 sim side verified with a fake wrist; laptop webcam run pending |
| 3 | **Voice** | ✅ 2026-09-28: live mic → laptop Whisper → Nemotron → sim; reset/stop local |
| 4 | **EEG offline** | EEGMMIDB clench-vs-rest decoder: within-subject and leave-subjects-out numbers reported |
| 5 | **Real arm revived** | control path known, RealArm backend, joint limits/speeds measured, e-stop hardware |
| 6 | **Real arm mirroring + voice** | ≥1 min of the real arm following + voice command on camera |
| 7 | EEG live | lab headset (OpenBCI/research) → live grasp on the real arm, or honest offline replay if not |
| 8 | Submission | video, README, feedback, prior-work note |

Fallbacks: if the real arm isn't ready by ~Oct 20, the video shows the real arm doing whatever it can
(even joint-space teleop) plus the sim for the rest; the rule needs ≥1 min of real hardware.

## Open questions (owner: Pranav / team)
- Real arm: control electronics + firmware, which joints work, gripper or not. → docs/REAL_ARM.md
- Real gripper choice → docs/GRIPPER.md
- EEG lab contact for after the offline decoder works. → docs/EEG.md
- Token Factory: API key + exact Nemotron model id(s) from the catalog.
