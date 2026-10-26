# Muse: put on, calibrate, go

One command, every time the headband goes on, on the Windows laptop. Anyone can slip on the Muse 2, calibrate, confirm
it works, and go live in about 2-3 minutes. The console is what the wearer reads; it walks them through four steps and
never needs the VM.

```
python -m rover_intent.eeg.muse_ready --name <person>
```

The Muse is opened once and stays open through all four steps. We do NOT release and re-open the board between
calibration and streaming, because BLE re-acquire on Windows is slow and flaky.

## The four steps

1. **FIT CHECK** - per-sensor contact quality (TP9, AF7, AF8, TP10), refreshed about once a second. It names the bad
   sensor and where it sits, and loops until all four read `good` for a few seconds in a row. Press Enter to go ahead
   anyway (you get a warning that contact was not confirmed). The quality heuristic and its thresholds are the same
   ones `eeg.sunday` uses for its signal check (flat / drift / mains), not a second copy.

   | sensor | where it sits |
   |--------|---------------|
   | TP9  | behind the LEFT ear |
   | AF7  | LEFT forehead (above the left eyebrow) |
   | AF8  | RIGHT forehead (above the right eyebrow) |
   | TP10 | behind the RIGHT ear |

   Fixes it suggests: press the sensor to the skin / wet it / move hair out from under it (flat), sit still and reseat
   the band (drift), push the band down onto skin and move away from chargers and screens (mains 50/60 Hz).

2. **QUICK CALIBRATION** (~60-90 s) - relax ~20 s, five cued double clenches, then a short negatives block (chew/talk,
   blink). It fits the clench detector on your own signal (`clench.calibrate`) and writes
   `data/profiles/clench_<name>.json`, backing up any existing one to `*.bak.<timestamp>`. It prints the fit quality
   honestly (`on_ratio`, clench peak, margin, false toggles). On a **poor** fit it says so and offers a redo
   (Enter = redo, `s` = continue anyway).

3. **READY CHECK** - a live clench score bar with the new profile. Try a double clench: you should see
   `TOGGLE (test, not sent)`. Nothing is sent to the app in this step. Press Enter when you are ready to go live.

4. **GO** - the same open board is handed to `stream_tx.stream_session`: raw EEG streams to the hub, and a double
   clench sends the `GripToggle` UDP packet to the app. It prints `READY: streaming EEG to the hub and clench ->
   gripper is LIVE`. Ctrl+C stops cleanly (zeros and closes the board the way `stream_tx` does).

## Options

```
--name <person>              wearer's name; profile -> data/profiles/clench_<name>.json  (required)
--board muse2                native BLE (default)
--board muse2-bled --serial COM3   BLED112 dongle
--board synthetic            no-Muse smoke test
--line-hz 50                 50 Hz mains (default 60)
--hub-host / --hub-port      hub (default rover-vm : 47150)
--app-host / --app-port      robot app for GripToggle (default rover-vm : 47100)
--skip-calib --profile <path>   fit check + Enter + go with an existing profile (no new calibration)
--no-stream                  stop after the Enter (fit + calibrate only; nothing goes live)
--profiles-dir <dir>         where to read/write clench_<name>.json (default data/profiles)
```

## The laptop file: `muse-go.cmd`

Windows Application Control on the laptop blocks the venv's `python.exe` launcher, so run the module with the uv base
Python 3.12 and point `PYTHONPATH` at both the repo `src` and the venv's installed packages (same workaround as
`muse-stream.cmd`). Find the base Python once with `uv python find 3.12`, paste it into `PY`, save this as
`muse-go.cmd` in the repo root, and run it as `muse-go alice` (or just double-click for the default name).

```bat
@echo off
REM --- edit these two lines for this laptop ---
set REPO=C:\Users\pranav\rover-intent
set PY=%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.12.7-windows-x86_64-none\python.exe
REM -------------------------------------------
set NAME=%1
if "%NAME%"=="" set NAME=guest
set PYTHONPATH=%REPO%\src;%REPO%\.venv\Lib\site-packages
cd /d %REPO%
"%PY%" -m rover_intent.eeg.muse_ready --name %NAME% --board muse2 ^
       --hub-host rover-vm --hub-port 47150 --app-host rover-vm --app-port 47100
pause
```

(`uv python find 3.12` prints the exact base-Python path; paste it into `PY`. The `.venv\Lib\site-packages` on the
PYTHONPATH is how brainflow/numpy get imported without the blocked venv launcher. SciPy may be blocked on the laptop;
`clench.py` falls back to the numpy-only filters in `eeg/np_signal.py`, so base Python + numpy + brainflow is enough.)

## Reading the live display (can you see it detecting your clench?)

STEP 3 (READY CHECK) and STEP 4 (GO) show one status line, refreshed ~10 times a second:

```
JAW [########|#......]   5.9x (need 4.7x) base  7.6x  CLENCH 1 of 2         gripper: OPEN
```

- The bar fills with the current ratio; the `|` is the threshold, so a clench visibly pushes the fill past the marker.
- `5.9x (need 4.7x)` is the live ratio vs the on threshold. In moving mode `base 7.6x` is the tracked floor (your
  current resting level while you move); in fixed mode there is no `base`.
- The state word walks `RELAXED -> CLENCH 1 of 2 -> GOT 1 of 2 - clench again -> CLENCH 2 of 2 -> GOT 2 - confirming`.
  If a double is rejected you see WHY: `single clench (need a quick second)`, `looked like chewing (3+), ignored`,
  `held too long, ignored`, `second clench too late`.
- Every toggle prints its own big line (not overwritten), tracking the expected gripper state (the laptop never hears
  the robot's state back, so it is the EXPECTED state, alternating from OPEN):

```
>>> GRIPPER CLOSED sent  (toggle #1 -> CLOSED expected, 17:42:10)
```

In STEP 4 the stream/network health (`hub=yes drop=0 seq=... rec=...`) rides as a compact tail on the same line so it
does not drown the clench feedback.

### Baseline mode (sitting still vs body teleop)

A jaw clench is detected as a short SPIKE. The default `--baseline moving` tracks your CURRENT muscle level, so it
still catches a clench while you are moving your body (arms raised, head turning, talking) - where the old fixed mode
sat permanently "high" and never completed a double. For a sit-still demo you can force `--baseline fixed`. ANSI colour
degrades gracefully; add `--no-color` for a plain cmd.exe. `--score-log run.csv` writes a ~5 Hz CSV
(`t, raw_score, baseline, rel_score, state, n_toggles`) so a teleop run can be compared against rest afterwards.

## Privacy

The clench profile and any raw EEG stay under the gitignored `data/` tree (`data/profiles/`, `data/eeg_sessions/`,
`data/eeg_recordings/`); nothing is committed or uploaded. Raw EEG only ever goes from the laptop to the VM over
Tailscale.
