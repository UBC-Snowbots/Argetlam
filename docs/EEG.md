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

## Muse 2 MI (imagined fist) - `eeg/muse_mi.py`

The Muse 2 has only TP9, AF7, AF8, TP10 (behind the ears + forehead). There is NO motor-strip
electrode, so the motor-8 models above (`train_mi.py`, `mi_riemann.joblib`) cannot run on it at all.
`muse_mi.py` retrains the SAME task (imagined fist T1/T2 vs rest T0, EEGMMIDB runs 4/8/12) on the
nearest EEGMMIDB electrodes to the Muse placement, so a new user's expected accuracy is measured
before any headset is worn, and provides a live cue protocol + personal-model training on the real board.

- **Muse-proxy channels** (Muse order): T9 (for TP9), AF7, AF8, T10 (for TP10). EEGMMIDB's montage has
  no TP9/TP10; after `eegbci.standardize` it has T9/T10 (used) and TP7/TP8 (fallback). `--channels` makes
  this configurable so a 5th AUX electrode (e.g. C3 over the motor strip) can be added later; the board
  read in `clench.open_board` still exposes 4 channels, so adding AUX also needs that slice widened.
- **Honesty / artifact controls**: the Muse sits over muscle (jaw at TP9/TP10) and eyes (AF7/AF8), both of
  which a decoder can latch onto. We report (1) an EMG guard that rejects trials whose 30-100 Hz power is
  >3 SD above rest; (2) EMG-only (>30 Hz) and EOG-only (<4 Hz frontal) control decoders on the same trials,
  with a printed verdict if a control matches the main model (then it is decoding artifact, not imagery);
  (3) a binomial p<0.05 chance threshold and a label-permutation p-value, not just "above 50 %".
- **Models**: Riemann TS + LR and CSP + LDA (reused) plus a mu/beta log-band-power + shrinkage-LDA baseline.
  Personal training does per-session Euclidean alignment (re-centering) after band-pass, leave-one-run-out CV
  (windows of one trial never split across folds), and cross-session (leave-one-session-out) when >=2 sessions.

Base model (new-user, EEGMMIDB Muse-proxy vs motor-8). Run via tsp (threads <= 4; cooler is faulty):
```
tsp bash -c 'cd <repo> && PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python -m rover_intent.eeg.muse_mi train-base --subjects 1-20 \
  --out models/muse_proxy_mi.joblib > logs/base_1-20.log 2>&1'
```

Live on the real board (Pranav runs these on his machine with the Muse; use a window, or `--headless`):
```
# measure the base model live (randomized REST/IMAGINE cues, EMG guard, CI + permutation):
PYTHONPATH=src python -m rover_intent.eeg.muse_mi live-eval --model models/muse_proxy_mi.joblib \
    --board muse2 --trials 20 --cue-s 4 --rest-s 3
# collect labelled data for a personal model (optionally 4 runs x 20, with real-clench trials):
PYTHONPATH=src python -m rover_intent.eeg.muse_mi collect --board muse2 --trials 80 --runs 4 --executed 2
# train a Muse-native personal model on one or more collected sessions:
PYTHONPATH=src python -m rover_intent.eeg.muse_mi train-personal \
    --sessions data/eeg_live/<ts>_collect [data/eeg_live/<ts2>_collect ...] \
    --out models/muse_personal_mi.joblib
```
`--board synthetic` works everywhere for a dry run; `--headless` prints cues instead of opening a window
(the only mode safe to run on the VM, which drives Pranav's Moonlight screen on DISPLAY :0). Sessions and the
personal model are personal biometric data: they are saved under `data/` (gitignored) and never committed.

### Base-model results (EEGMMIDB subjects 1-20, 2026-10-25)

Imagined fist (T1/T2) vs rest (T0), 8-30 Hz, 0.5-2.5 s epochs, 160 Hz. Chance = 0.50; the one-sided binomial
p<0.05 floor is ~0.59 at the ~90 balanced epochs/subject used here. LOSO = brand-new user, zero calibration;
within = per-subject 5-fold CV. Values are mean +- SD across 20 subjects (none skipped).

| channel set | pipeline | within-subject CV | leave-subjects-out | within >= .60 |
|---|---|---|---|---|
| Muse-proxy (T9 AF7 AF8 T10) | Riemann TS + LR | 0.669 +- 0.112 | **0.594 +- 0.086** | 14/20 |
| Muse-proxy | CSP + LDA | 0.651 +- 0.119 | 0.569 +- 0.100 | 13/20 |
| Muse-proxy | band-power + LDA | 0.592 +- 0.087 | 0.569 +- 0.060 | 8/20 |
| motor-8 (FC3..Pz) | Riemann TS + LR | 0.691 +- 0.105 | 0.567 +- 0.086 | 15/20 |
| motor-8 | CSP + LDA | 0.680 +- 0.108 | 0.552 +- 0.078 | 15/20 |
| motor-8 | band-power + LDA | 0.607 +- 0.074 | 0.581 +- 0.051 | 11/20 |
| EOG control (AF7 AF8, <4 Hz) | band-power + LDA | 0.659 +- 0.110 | **0.638 +- 0.109** | 14/20 |

Read this honestly:
- A brand-new user on the Muse placement gets ~0.59 (Riemann, LOSO) on imagined fist vs rest - barely above the
  ~0.59 significance floor, with a wide spread. Within-subject (with calibration) is ~0.67. The Muse placement is
  NOT much worse than the 8 motor-strip channels here, but that is because EEGMMIDB LOSO is weak for everyone
  (motor-8 Riemann LOSO is only 0.567); it is a hard cross-subject dataset, not evidence the Muse sees motor imagery.
- Artifact verdict (automatic): the EOG-only control (low-frequency frontal, i.e. eye/EOG, 0.638 LOSO) BEATS the
  motor-imagery model (0.594 LOSO). So most of the "imagery vs rest" separability on these channels is explained
  by low-frequency frontal activity (eye movement / task engagement differences between cue and rest), NOT
  motor-cortex mu/beta. Treat the base Muse model as a weak, artifact-prone prior, not a real imagined-movement
  decoder. The live `live-eval` EMG guard + per-user `train-personal` controls are there to keep us honest on the
  real headband.

Model saved to `models/muse_proxy_mi.joblib` (Riemann TS + LR fit on all 20 subjects; report JSON alongside).
