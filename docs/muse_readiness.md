# Muse 2 EEG readiness: which genuine-EEG pipeline to use on day one

Goal: before the Muse 2 headband arrives, decide from offline evidence which real-EEG control to run in the first
guided session. Jaw clench (EMG, `eeg/clench.py`) is kept as the reliable hands-free input, but Pranav asked for a
genuine brain-signal control. The two realistic candidates on a 4-channel dry headband are:

- **SSVEP** ("look at a flickering box to confirm"): read which on-screen flicker frequency the user is looking at.
- **Alpha / eyes-closed** ("close your eyes to confirm"): 8-12 Hz power rises when the eyes shut (Berger effect).

Both were characterised offline on public gel-electrode datasets, restricted to the electrode sites nearest the
Muse's, and compared against a proper posterior/occipital montage as a ceiling. All numbers below come from the two
study scripts (`eeg/study_alpha.py`, `eeg/study_ssvep.py`) and the artifacts they wrote
(`data/eeg_studies/alpha_results.json`, `data/eeg_studies/ssvep_results.json`). Nothing here is hand-entered.

> The Muse 2 (BrainFlow `MUSE_2_BOARD`=38 native BLE, `MUSE_2_BLED_BOARD`=22) streams 256 Hz from four dry
> electrodes: TP9, AF7, AF8, TP10 (reference Fpz). There is no occipital or motor-strip electrode. The board
> description also lists one `other_channels` entry (possibly the micro-USB AUX input); this is UNVERIFIED and may
> need a BrainFlow preset, so the AUX-electrode results below are an "if we can add one occipital electrode"
> scenario, not a promise the Muse delivers it out of the box.

---

## The honest caveat up front

These are **proxies**. The study electrodes are wet gel electrodes at sites *near* the Muse's, on different people,
in different sessions. In particular:

- EEGMMIDB (alpha study) has **no TP9/TP10**; the Muse-proxy set uses AF7, AF8 and the nearest temporal sites
  (TP7, TP8, T9, T10). The Muse's real electrodes are dry, which is noisier and drifts more than gel.
- Wang2016 (SSVEP study) has **no TP9/TP10 and no AF7/AF8** (it has AF3/AF4); the Muse-proxy set uses TP7, TP8, T7,
  T8. Its "AUX" scenario is a single clean occipital Oz electrode.
- The alpha study uses two separate baseline runs (eyes open = R01, eyes closed = R02), so it measures **steady-state
  separability, not the latency** of a live open to closed switch. Budget extra time for that live.

So treat the proxy numbers as an optimistic-ish upper bound for the gel sites and a *rough* guide for the dry Muse.
The day-one session recorder exists precisely to replace these proxies with real Muse numbers in ~10 minutes.

---

## Task B result: SSVEP (Tsinghua benchmark, Wang2016)

9 subjects, 40-target benchmark restricted to our class sets. Training-free CCA and FBCCA, and calibrated ensemble
TRCA (leave-one-block-out). Window = decision time; ITR is Wolpaw bits/min at that window (a live system adds a
gaze-shift gap on top, so real ITR is lower). Chance: 2-class 50%, 4-class 25%. Frequencies chosen so a 60 Hz screen
can render them (12 Hz = 5 frames, 15 Hz = 4 frames) and to avoid the ~10 Hz alpha band.

### 2class (freqs [12.0, 15.0])

**muse_proxy** (acc / ITR bits/min)
| window s | CCA | FBCCA | TRCA |
|---|---|---|---|
| 0.5 | 0.53 / 0 | 0.48 / 0 | 0.51 / 0 |
| 1.0 | 0.60 / 2 | 0.56 / 1 | 0.53 / 0 |
| 2.0 | 0.56 / 0 | 0.59 / 1 | 0.62 / 1 |
| 3.0 | 0.63 / 1 | 0.64 / 1 | 0.65 / 1 |
| 4.0 | 0.67 / 1 | 0.68 / 1 | 0.65 / 1 |

**aux_oz** (acc / ITR bits/min)
| window s | CCA | FBCCA | TRCA |
|---|---|---|---|
| 0.5 | 0.65 / 8 | 0.71 / 16 | 0.64 / 7 |
| 1.0 | 0.75 / 11 | 0.83 / 21 | 0.74 / 10 |
| 2.0 | 0.85 / 12 | 0.89 / 15 | 0.83 / 10 |
| 3.0 | 0.89 / 10 | 0.91 / 11 | 0.90 / 10 |
| 4.0 | 0.93 / 9 | 0.94 / 10 | 0.92 / 9 |

**occipital9** (acc / ITR bits/min)
| window s | CCA | FBCCA | TRCA |
|---|---|---|---|
| 0.5 | 0.81 / 35 | 0.89 / 60 | 0.94 / 78 |
| 1.0 | 0.86 / 25 | 0.94 / 41 | 0.97 / 49 |
| 2.0 | 0.93 / 19 | 0.97 / 25 | 0.99 / 28 |
| 3.0 | 0.96 / 15 | 0.98 / 17 | 0.98 / 17 |
| 4.0 | 0.97 / 12 | 0.99 / 14 | 0.99 / 14 |

### 4class (freqs [12.0, 13.0, 14.0, 15.0])

**muse_proxy** (acc / ITR bits/min)
| window s | CCA | FBCCA | TRCA |
|---|---|---|---|
| 0.5 | 0.24 / 0 | 0.23 / 0 | 0.31 / 2 |
| 1.0 | 0.31 / 1 | 0.26 / 0 | 0.35 / 2 |
| 2.0 | 0.29 / 0 | 0.37 / 2 | 0.43 / 3 |
| 3.0 | 0.31 / 0 | 0.47 / 3 | 0.35 / 1 |
| 4.0 | 0.38 / 1 | 0.47 / 2 | 0.38 / 1 |

**aux_oz** (acc / ITR bits/min)
| window s | CCA | FBCCA | TRCA |
|---|---|---|---|
| 0.5 | 0.41 / 10 | 0.52 / 29 | 0.63 / 55 |
| 1.0 | 0.66 / 32 | 0.67 / 34 | 0.70 / 38 |
| 2.0 | 0.79 / 28 | 0.85 / 35 | 0.78 / 27 |
| 3.0 | 0.89 / 27 | 0.90 / 27 | 0.85 / 23 |
| 4.0 | 0.92 / 22 | 0.91 / 21 | 0.85 / 17 |

**occipital9** (acc / ITR bits/min)
| window s | CCA | FBCCA | TRCA |
|---|---|---|---|
| 0.5 | 0.58 / 43 | 0.70 / 77 | 0.91 / 169 |
| 1.0 | 0.81 / 59 | 0.92 / 89 | 0.96 / 103 |
| 2.0 | 0.89 / 40 | 0.97 / 53 | 0.98 / 55 |
| 3.0 | 0.94 / 32 | 0.99 / 37 | 0.98 / 37 |
| 4.0 | 0.97 / 27 | 0.99 / 29 | 0.99 / 28 |

**Reading of Task B.** On the Muse's own electrode sites (TP/T, `muse_proxy`) SSVEP is barely above chance: 2-class
tops out around 0.68 at a 4 s window, 4-class around 0.47. That is not a usable control. A **single clean occipital
electrode** (`aux_oz`) changes the picture completely: 2-class reaches ~0.83 at 1 s and ~0.94 at 4 s, 4-class ~0.90 at
3 s. The full 9-channel occipital montage is near-perfect even at 0.5 s (the benchmark's design point). FBCCA beats
plain CCA at short windows and needs no calibration; TRCA only wins with the full montage and per-user training.

**Implication:** SSVEP is strong *only if an occipital electrode is in play*. On the bare Muse it is not a day-one
control. If the AUX port turns out to drive an occipital electrode, SSVEP becomes the best option.

---

## Task A result: alpha / eyes-closed (EEGMMIDB R01 vs R02)

### within_lda
| window s | muse_proxy | posterior |
|---|---|---|
| 0.5 | 0.74±0.12 | 0.74±0.13 |
| 1.0 | 0.79±0.12 | 0.78±0.14 |
| 1.5 | 0.83±0.12 | 0.81±0.14 |
| 2.0 | 0.84±0.12 | 0.82±0.14 |
| 2.5 | 0.86±0.12 | 0.83±0.15 |
| 3.0 | 0.86±0.12 | 0.83±0.14 |

### within_threshold
| window s | muse_proxy | posterior |
|---|---|---|
| 0.5 | 0.67±0.10 | 0.69±0.12 |
| 1.0 | 0.71±0.11 | 0.72±0.13 |
| 1.5 | 0.76±0.13 | 0.75±0.15 |
| 2.0 | 0.78±0.13 | 0.77±0.15 |
| 2.5 | 0.81±0.13 | 0.78±0.16 |
| 3.0 | 0.82±0.13 | 0.80±0.16 |

### newperson_lda
| window s | muse_proxy | posterior |
|---|---|---|
| 0.5 | 0.64±0.10 | 0.68±0.11 |
| 1.0 | 0.68±0.11 | 0.71±0.12 |
| 1.5 | 0.71±0.13 | 0.73±0.14 |
| 2.0 | 0.73±0.13 | 0.75±0.15 |
| 2.5 | 0.75±0.14 | 0.75±0.15 |
| 3.0 | 0.76±0.14 | 0.76±0.16 |

**Reading of Task A.** Eyes-closed alpha is a usable control on the Muse's own electrode sites. With a quick per-user calibration (`within_lda`), the Muse-proxy temporal/forehead sites reach 0.84 balanced accuracy at a 2 s window and 0.86 at 2.5-3 s - essentially tied with the full posterior montage (0.82-0.83). Even the simple relative-alpha threshold (`within_threshold`, what `AlphaDetector` uses) gets 0.78-0.82 at 2-3 s. A brand-new user with no calibration (`newperson_lda`) still gets 0.73-0.76 at 2-3 s. Unlike SSVEP, alpha does not need an occipital electrode: it spreads broadly enough that TP9/TP10/AF7/AF8 capture it almost as well as O1/Oz/O2, which is why the two montages are so close here. Caveat (restated): R01 and R02 are separate runs, so this is steady-state separability; the live onset latency of actually closing your eyes is not measured and will add ~1-2 s on top of the window.

---

## Recommendation for day one

**Use alpha (eyes-closed) as the day-one genuine-EEG control, with SSVEP held in reserve.**

- **Alpha is the go.** It works on the bare Muse (no extra electrode), needs only a light per-user threshold, and reaches ~0.84-0.86 calibrated / ~0.73-0.76 cold at a 2-3 s window (chance 0.50). Map it to "close your eyes ~2 s to confirm the grab", gated by a grab candidate being in reach so a blink or a rest cannot fire it. It is slow (a 2 s window plus onset) but dependable, and it degrades gracefully: the report's `alpha.overall_ratio` and accuracy tell you on the day whether this user clears the bar.
- **SSVEP is the stretch goal, gated on an occipital electrode.** On the Muse's TP/T sites it is near chance (2-class <= 0.68, 4-class <= 0.47), so it is NOT a day-one control as the headband ships. But a single clean occipital electrode takes 2-class to ~0.94 and 4-class to ~0.90, with much higher ITR than alpha. If the Muse AUX port can be made to drive an occipital electrode (UNVERIFIED), switch the primary control to SSVEP "look at the confirm box" and keep alpha as the fallback. FBCCA is the right training-free method; add per-user TRCA from the session only with a good occipital montage.
- **Jaw clench stays the hard fallback** (`eeg/clench.py`): EMG, placement-independent, already merged and tested.

Net: day one, run `muse_detect run --detector alpha`. Record the guided session first; if its report shows SSVEP working (occipital AUX in play), switch to `--detector ssvep` with alpha as fallback (`--detector both`).

---

## How a new EEG event would wire into the arm (no behaviour changed yet)

The detectors emit a one-shot **event**, never a continuous level, so they cannot fight voice / reset / assist (same
reasoning as the clench toggle). Two message types exist:

- **`GripToggle`** (`types.py`, already consumed by `Arbiter.on_grip_toggle`): the day-one path. `eeg/muse_detect.py`
  sends it with `source="ssvep"` or `source="alpha"`. A confirm closes the gripper if open, opens it if closed, and
  near a grab candidate it starts the same grab assist the fist gesture uses. **No app.py / arbiter.py change needed.**
- **`EEGSelect`** (new, added to `types.py` and `transport/udp.py` with tests): a richer event carrying *which* SSVEP
  target was chosen (`label`, `freq`, `confidence`), for a multi-target menu (confirm vs cancel, or a 4-way choice).
  The arbiter does **not** consume it yet. To wire it in later, add an `Arbiter.on_eeg_select(msg)` that maps
  `label` to an action (e.g. `"grab"` -> `_commit_grip` close, `"cancel"` -> release) and dispatch it from `app.py`
  alongside the existing `GripToggle` handling. Until then it is logged and available on the wire.

Recommended control logic (in `eeg/muse_detect.py`, tunable by flags): SSVEP "look at the confirm box to grab", GATED
by a grab candidate being present (`--confirm-freq` picks the grab target; a second frequency can mean cancel), with
alpha "close your eyes to confirm" as the fallback when SSVEP is too weak on the dry electrodes.

---

## Day-one checklist for Pranav

Everything runs on the laptop with the headband; the VM only receives the confirm event over UDP. Install once:

```
uv pip install -e '.[muse]'        # brainflow + scipy + pygame  (eeg/clench.py needs only numpy+scipy+brainflow)
```

1. **Pair the headband.** Turn the Muse on, enable Bluetooth on the laptop. Native BLE is the default board
   (`--board muse2`); if pairing is flaky use a BLED112 dongle (`--board muse2-bled --serial <port>`). You can target a
   specific headband with `--name Muse-XXXX`.
2. **Check the display** (no headband needed), because the whole SSVEP idea depends on frame-locked flicker:
   ```
   python -m rover_intent.eeg.muse_session --check-display --freqs 12,15
   ```
   Go: `measured_refresh_hz` is stable near 60 (or your real rate), `realized_freqs` are close to 12 and 15, and
   `frac_dropped` < 0.02. No-go: many dropped frames or a refresh that will not render 12/15 Hz cleanly (then pick
   frequencies that divide your refresh, e.g. 60/5 and 60/4).
3. **Record the guided session** (~10 min: eyes open/closed blocks, then SSVEP flicker trials):
   ```
   python -m rover_intent.eeg.muse_session record --board muse2 --out data/eeg_recordings/session_pranav.npz --freqs 12,15
   ```
   Follow the on-screen cues. It saves EEG, EEG-clock markers and a frame-timing log into one npz.
4. **Read the readiness report**:
   ```
   python -m rover_intent.eeg.muse_report --file data/eeg_recordings/session_pranav.npz
   ```
   What the numbers mean (go / no-go):
   - **`verdict.display_ok` must be true** (frame drops under 2%). If not, fix the display before trusting SSVEP.
   - **Alpha**: `alpha.overall_ratio` (eyes-closed / eyes-open relative alpha) should be >= ~1.5 and
     `alpha.accuracy` balanced accuracy >= ~0.8 at a 1-2 s window. That makes "close your eyes to confirm" a go.
   - **SSVEP**: `ssvep_accuracy.fbcca` at 2-4 s should clear ~0.85 (2-class) to be a go. On the bare Muse this will
     likely be weak (near chance), matching Task B; that is the expected no-go for SSVEP-without-occiput.
   - **`ssvep_snr_db`** per channel per frequency: positive dB at the flicker frequency on at least one channel means
     some SSVEP is present; all-negative means none is reaching the dry electrodes.
5. **Run the chosen detector live** (emits a confirm/grab to the VM):
   ```
   # alpha fallback (most likely the day-one go):
   python -m rover_intent.eeg.muse_detect run --detector alpha --board muse2 --host <vm-ip> --alpha-thr <from report>
   # SSVEP, only if the report says it works (e.g. with an AUX occipital electrode):
   python -m rover_intent.eeg.muse_detect run --detector ssvep --freqs 12,15 --confirm-freq 15 --board muse2 --host <vm-ip>
   ```
   Add `--dry` first to watch detections without sending. `replay` runs the same detector on the saved npz offline.

If none of the EEG controls clears its bar on day one, fall back to the jaw-clench toggle (`eeg/clench.py`), which is
EMG and does not depend on electrode placement.
