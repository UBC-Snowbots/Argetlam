# Live raw-EEG streaming: laptop -> VM

> For the everyday "put the headband on and go" routine (fit check -> calibrate -> confirm -> stream, one command),
> see **docs/MUSE_READY.md** (`python -m rover_intent.eeg.muse_ready --name <person>`). It wraps the streamer below so
> a new wearer never has to run the steps by hand.

Streams the Muse 2's raw EEG from the Windows laptop (where the headband is paired over Bluetooth) to the VM over
Tailscale, so VM-side tools can run models on live Muse data. Only ONE process can hold the Muse at a time, so the
laptop streamer (`stream_tx`) also does the live double-clench -> GripToggle job that `clench run` used to do. It
replaces `clench run`; do not run both.

```
  laptop                                  VM (rover-vm)
  Muse 2 --BLE--> stream_tx --TCP 47150--> stream_hub --TCP 127.0.0.1:47152--> any VM tool (--board hub)
                     |                                                            e.g. clench, muse_session, muse_detect
                     +--UDP 47100--> robot app (GripToggle, same as clench run)
```

## Wire format (newline-delimited JSON over TCP)

First a header, then one frame per chunk (~every 50 ms):

```
{"kind":"eeg_header","fs":256,"channels":["TP9","AF7","AF8","TP10"],"board":"muse2","t0":<sender time.time()>}
{"kind":"eeg","seq":0,"t":<sender time.time() of the LAST sample>,"x":[[TP9...],[AF7...],[AF8...],[TP10...]]}
{"kind":"eeg","seq":1,"t":...,"x":[[...],[...],[...],[...]]}
```

`x` is the 4 Muse EEG channels only, floats, microvolts as BrainFlow gives them, in the channel order above. `seq`
increases by 1 per frame (the hub counts gaps as dropped frames). The header is re-sent whenever the sender
reconnects, so a hub that restarts always gets the header before any frame.

## VM: start the hub

```
python -m rover_intent.eeg.stream_hub
# listens on 0.0.0.0:47150 for the laptop, fans out on 127.0.0.1:47152 to local tools.
# optional recording (stays under the gitignored data/ tree):
python -m rover_intent.eeg.stream_hub --record data/eeg_sessions
```

It prints a 1 Hz line: sender connected yes/no, fs, frames/s, samples/s, seq gaps, consumer count, and an estimate of
sender-to-hub latency (frame timestamp vs arrival time; only meaningful if the two machines' clocks are in sync).

A new sender on 47150 replaces the old one. Any number of local tools may connect to the fan-out; late joiners get the
header first.

## Laptop: start the streamer (replaces `clench run`)

Plain Python with numpy + brainflow. No scipy is required (clench.py has a numpy-only filter fallback), no pylsl.

Stream only:
```
python -m rover_intent.eeg.stream_tx --board muse2 --hub-host rover-vm --hub-port 47150
```

Stream AND keep the clench -> gripper job (recommended; this is what replaces `clench run`):
```
python -m rover_intent.eeg.stream_tx --board muse2 --hub-host rover-vm ^
       --clench-profile data/profiles/clench_pranav.json --app-host rover-vm --app-port 47100
```

On a double clench it prints `TOGGLE #n (sent)` and sends the same GripToggle UDP packet `clench run` did. A 1 Hz
console line shows board fs, frames sent, hub connected yes/no, dropped frames, and (when clench is on) the clench
score bar.

If native BLE is flaky use the dongle: `--board muse2-bled --serial COM3`. `--board synthetic` is a no-Muse smoke test.

### Windows base-Python workaround (`muse-stream.cmd`)

Windows Application Control on the laptop blocks the venv's `python.exe` launcher, so run the module with the uv base
Python 3.12 and point `PYTHONPATH` at both the repo `src` and the venv's installed packages. Find the base Python once
with `uv python find 3.12` and the repo path, then save this as `muse-stream.cmd` in the repo root and double-click it:

```bat
@echo off
REM --- edit these two lines for this laptop ---
set REPO=C:\Users\pranav\rover-intent
set PY=%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.12.7-windows-x86_64-none\python.exe
REM -------------------------------------------
set PYTHONPATH=%REPO%\src;%REPO%\.venv\Lib\site-packages
cd /d %REPO%
"%PY%" -m rover_intent.eeg.stream_tx --board muse2 --hub-host rover-vm --hub-port 47150 ^
       --clench-profile data\profiles\clench_pranav.json --app-host rover-vm --app-port 47100
pause
```

(`uv python find 3.12` prints the exact base-Python path; paste it into `PY`. The `.venv\Lib\site-packages` on the
PYTHONPATH is how brainflow/numpy get imported without the blocked venv launcher.)

## VM tools: read the live stream with `--board hub`

Any tool that opens a board through `rover_intent.eeg.clench.open_board` understands `--board hub`. It connects to the
hub fan-out at `127.0.0.1:47152` (override with `HUB_FANOUT_HOST` / `HUB_FANOUT_PORT`), waits for the header, and then
behaves like a BrainFlow board (`get_board_data()` returns everything since the last call as a `(4, n)` float array).

```
python -m rover_intent.eeg.clench run   --board hub --profile data/profiles/clench_pranav.json --dry
python -m rover_intent.eeg.muse_session record --board hub --out data/eeg_sessions/live.npz
python -m rover_intent.eeg.muse_detect  run --board hub
```

`hub` is now a choice in `clench`, `sunday`, `muse_detect`, and `muse_session`.

## Clench feedback + moving baseline (2026-10-25)

`stream_tx` (with `--clench-profile`) now shows the same glanceable clench status line as `muse_ready` STEP 4: a JAW
bar with a `|` threshold marker, the live ratio, the moving baseline, a plain-English state (RELAXED -> CLENCH 1 of 2
-> ... -> TOGGLE SENT, with the rejection reason when a double is dropped), and a big `>>> GRIPPER CLOSED/OPEN sent`
line per toggle. Stream health (`hub=yes drop=0 seq=... rec=...`) rides as a compact tail so it does not drown the
clench feedback. See `docs/MUSE_READY.md` for a worked example of the line.

Flags on `stream_tx` (and `muse_ready`): `--baseline fixed|moving` (default moving - tracks the current muscle level
so a clench is caught during body teleop, not just sitting still), `--score-log <path>` (a ~5 Hz CSV of
`t, raw_score, baseline, rel_score, state, n_toggles` to compare teleop vs rest later), and `--no-color` (plain
cmd.exe). The detector exposes a read-only `ClenchDetector.status()` -> `ClenchStatus` that drives the display.

## Privacy

Raw EEG is personal biometric data. It only ever goes from Pranav's laptop to Pranav's own VM over Tailscale (no third
party, no cloud). Recordings written by `--record` stay under `data/`, which is gitignored (as are
`data/eeg_sessions/`, `data/eeg_recordings/`, `data/profiles/`); nothing is committed or uploaded.
