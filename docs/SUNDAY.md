# SUNDAY runbook: Muse 2 day-one EEG session

One command records a short labelled session from your own head, fits YOUR clench + eyes-closed-alpha calibration,
writes it where the tools already load it (old one backed up), and runs a live test into the SIM. Everything is on
the laptop; the VM only receives the grip-toggle event. Your EEG is personal biometric data: it stays on local disk
(`data/eeg_sessions/`, gitignored), never uploaded, never committed.

Tool: `src/rover_intent/eeg/sunday.py`. Run it as `python -m rover_intent.eeg.sunday ...` from the repo root.

## What actually gets fit (be honest)
- **Clench detector** (`eeg/clench.py`) - EMG, your jaw muscle, the chosen control. We refit your relaxed level +
  the on/off threshold on your real rest / clench / negative recordings. This is the primary target and the reliable one.
- **Eyes-closed alpha detector** (`eeg/alpha.py`) - genuine EEG. We fit your relative-alpha threshold on eyes-open
  vs eyes-closed blocks. Dependable but slow (~2 s to fire).
- **Not fit here:** SSVEP (near chance on the bare Muse, needs an occipital AUX electrode) and the motor-imagery
  decoder (needs motor-strip electrodes the Muse does not have). `--with-ssvep` only *measures* SSVEP for a report.

---

## SATURDAY (10 min, NO Muse) - preflight

On the Windows laptop, in the repo, with the venv active:

```
pip install -e ".[muse]"              # brainflow + scipy + pygame (one time)
python -m rover_intent.eeg.sunday --preflight
```

Finishes in under 3 minutes and runs the WHOLE session on BrainFlow's synthetic board. You want to see
`PREFLIGHT PASS`. Lines explained:
- `import numpy / scipy / brainflow` must be **OK** (these are fatal). `pygame` only matters for `--with-ssvep`.
- `bluetooth adapter` is a **WARN** on a machine with no detectable adapter; on Windows it should say *detected*.
  If it cannot see your adapter, turn Bluetooth on in Windows settings before Sunday.
- `full dry run (synthetic board, end to end)` must be **OK** - that proves recording -> fit -> profile write ->
  live replay all work without the headband.

If any fatal line says MISSING, run the `pip install` it prints and re-run the preflight.

---

## SUNDAY (~15-25 min total) - the real session

0. **Bring up the SIM** on a SPARE port (never 47100, never the real arm). In one terminal on the VM:
   ```
   cp configs/default.yaml /tmp/muse_sim.yaml
   sed -i 's/listen_port: 47100/listen_port: 47195/' /tmp/muse_sim.yaml
   python -m rover_intent.app --config /tmp/muse_sim.yaml --backend mock        # or --backend isaac for the sim arm
   ```
   (For the Isaac sim, use `~/bin/rover-stack` as usual but on the spare-port config. Mock is fine for the toggle demo.)

1. **Fit the headband and turn it on.** Bluetooth on. Then run the one command on the laptop:
   ```
   python -m rover_intent.eeg.sunday --board muse2 --name pranav --host <vm-ip> --port 47195
   ```
   (`--board muse2-bled --serial COM3` if you use the BLED112 dongle; `--line-hz 50` outside North America.)

2. Follow the on-screen prompts. The steps, in order, with a time budget:

   | step | what you do | ~time |
   |---|---|---|
   | signal check | sit still ~6 s; it prints per-electrode status | ~1 min |
   | rest baseline | relax, jaw loose, blink normally | 30 s |
   | single clenches | one firm half-second bite each time the counter shows (x8) | ~1 min |
   | **double clenches** | two quick bites in a row, then rest (x10) - the real control gesture | ~1.5 min |
   | negatives: talk/chew | read aloud, chew gum - do NOT double-clench | 30 s |
   | negatives: blinks | blink and move your eyes | 20 s |
   | negatives: head turns | turn/tilt your head around | 20 s |
   | eyes open | look softly at a fixed point | 1 min |
   | eyes closed | eyes shut, relax | 1 min |
   | fit + metrics | automatic | ~30 s |
   | live test | say "follow me" at the sim, then double-clench to toggle the gripper | ~1 min |

   Recording is ~9-10 min of headband time; the rest is setup and the metrics. Well within 25-30 min.

3. **What good signal looks like** (the signal-check step): each of TP9, AF7, AF8, TP10 should read `good`, with
   rms roughly 5-60 uV and the 60 Hz ratio under ~0.6. If a channel is bad it refuses to record and tells you which:
   - `flat` -> the sensor is not on skin. Push the band down; part the hair under TP9/TP10 (behind the ears).
   - `mains` -> too much 60 Hz hum. Wet the sensors (a dab of water), press them onto the skin, move off a charger.
   - `drift` -> big slow movement. Sit still ~10 s and it usually settles.

4. **Read the metrics** it prints (plain English). Good targets:
   - Double-clench detection (held-out) **>= 80 %**.
   - False triggers while talking/chewing/blinking/head-turning **<= ~2 per minute** (lower is better).
   - Eyes-closed vs eyes-open accuracy **>= ~75 %** (chance 50 %); alpha ratio **>= ~1.5x**.
   It also prints how the synthetic default would have done on the same data, so you can see the calibration helped.

5. The new calibration is written automatically:
   - `data/profiles/clench_pranav.json` (loaded by `eeg/clench.py run` and `eeg/muse_detect.py`)
   - `data/profiles/muse_pranav.json` (the alpha threshold)
   The previous files, if any, are renamed to `*.bak.<timestamp>` - nothing is overwritten.

6. After the session, drive the arm for real from the headband (SIM only, spare port):
   ```
   python -m rover_intent.eeg.clench run --profile data/profiles/clench_pranav.json --board muse2 --host <vm-ip> --port 47195
   ```

## If something fails
- **Muse will not connect:** check Bluetooth is on; try `--muse-name Muse-XXXX` to pick the headband; if native BLE
  is flaky, use the BLED112 dongle (`--board muse2-bled --serial COM3`). Native BLE on Windows is the one thing we
  could not test without the hardware - have the dongle ready as a fallback.
- **A block went wrong** (coughed, moved): re-run just that block, the earlier ones are kept:
  ```
  python -m rover_intent.eeg.sunday --board muse2 --name pranav --only clench_double --host <vm-ip> --port 47195
  ```
  Stage names: rest, clench_single, clench_double, neg_chew, neg_blink, neg_head, eyes_open, eyes_closed, train, live.
- **It crashed partway:** just re-run the same command. It resumes - blocks already saved are skipped. Use
  `--no-resume` to force re-recording.
- **Refit without re-recording:** `--only train`. **Re-run just the live test:** `--only live --host <vm-ip> --port 47195`.
- **Clench accuracy is poor:** redo the clench + negative blocks after wetting the sensors; the threshold is only as
  good as the separation between your clench and your rest.
- **Nothing toggles in the sim during the live test:** the arm only responds when it is *following*. Say "follow me"
  to the sim (the live test sends this for you) or stand in the body-teleop frame, then double-clench.

## Hard rules (unchanged)
- Live tests go to the SIM on a spare port (47195), never 47100, never the real arm. `sunday.py` refuses `--port 47100`.
- Your EEG never leaves the laptop and is never committed (the `data/` folder is gitignored).
- `--with-ssvep` is optional and only useful if an occipital electrode is wired to the Muse AUX port (unverified).
