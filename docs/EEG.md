# EEG: grasp intent from motor imagery

Plan: prove it offline on public data first, then ask a lab for a headset.

- **Task**: binary *imagined hand clench* vs *rest* → gripper close/open (with hysteresis + 0.5 s dwell in the Arbiter).
- **Data**: PhysioNet EEG Motor Movement/Imagery (EEGMMIDB): 109 subjects, 64 ch, 160 Hz; imagery runs 4, 8, 12.
  Later candidates: BCI Competition IV 2a (22 ch, 9 subjects), Cho2017 (52 subjects), via MOABB.
- **Channels**: 8 motor-strip channels (FC3 FC4 C3 Cz C4 CP3 CP4 Pz) = what an 8-ch OpenBCI Cyton can cover.
- **Models**: CSP+LDA baseline, Riemannian tangent space + LR. Report within-subject CV and leave-subjects-out separately.
  Expect roughly 60–75 % for a brand-new user, higher with calibration. Report the real numbers either way.
- **Live**: `python -m rover_intent.eeg.live` (BrainFlow; synthetic board now, Cyton later).
- **Honesty rule for the video**: if the live headset isn't available, show a replay of held-out real EEG
  driving the gripper and say so on screen.

Run training through tsp (downloads ~2 GB for 109 subjects; start with 1-20):
```
tsp bash -c 'cd ~/projects/ubc-rover-arm/rover-intent && .venv/bin/python -m rover_intent.eeg.train_mi --subjects 1-20 > logs/mi_1-20.log 2>&1'
```
