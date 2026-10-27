# Table camera: object finding with one plain RGB webcam (no depth)

"Option 2": get object positions on the table from a single fixed RGB camera, so the pick / place /
move skills work without a depth sensor and without sim ground truth.

## How it works (the table-plane trick)

The objects rest on a flat table at a known height `z_table` in the arm base frame, and every known
object has a known footprint radius and height (the same object library as the sim: `scene:` in the
arm config). A single camera cannot measure depth, but with those knowns a detector box is enough:

1. A calibrated pinhole camera model maps pixels <-> rays in the arm base frame
   (`perception/camera.py`). Intrinsics from a ChArUco board; extrinsics (camera pose in the base
   frame) from one board view at a measured table pose.
2. An open-vocabulary, text-promptable detector ("short red cup", "tall blue bottle") finds each
   object and returns a pixel box (`perception/detector.py`).
3. The box is turned into a table position (`perception/projection.py`):
   back-project to the table plane `z = z_table`, then recover the footprint centre `(x, y)`. z is
   pinned to the known table height; only `(x, y)` is estimated. Objects outside the workspace box
   or below the confidence threshold are rejected.
4. The Scene's object positions are filled from the camera path before planning; the planner's
   object list is the detected names. The confirm-before-move flow is unchanged.

Two estimators (`perception.method`): **refine** (default) fits `(x, y)` so the object's predicted
silhouette box matches the observed box, using the full known geometry (Gauss-Newton, pure numpy);
it is accurate for any mount. **pushback** is a one-shot fallback (bottom-centre back-projection +
footprint-radius push-back toward the scene) that is good for an oblique mount but degrades near
top-down; keep **refine**.

## What camera, mounted where

- Any plain USB / RGB webcam (720p/1080p is plenty; we model and test at 640x480). No depth camera.
- **Fixed** rig looking at the whole workspace. The camera must not move after calibration; if it
  is bumped, re-run the extrinsics step (one photo, ~1 minute).
- Mount **high and fairly steep** (closer to overhead) when you can: the sim validation shows a
  high overhead mount and a moderate oblique mount both pass the accuracy gate, and a steeper view
  reduces the projection's sensitivity to box error and keeps objects from occluding each other.
  Keep the whole reach box in frame.
- Put it where the arm itself does not hide the objects (the sim "behind" view, behind-left and
  above the base, is a good reference).

## Calibration (no arm motion needed)

Print a ChArUco board (`BoardSpec` default: 5x7 squares, 35 mm square, 26 mm marker, DICT_4X4_50;
use whatever you print and pass the real sizes). OpenCV with aruco is required: the `vision` extra
(`opencv-contrib-python-headless`).

**1. Intrinsics** - ~15 photos of the board held at many angles and distances, filling the frame:
```
python scripts/perception_calibrate.py intrinsics --images 'calib_shots/*.jpg' \
    --squares-x 5 --squares-y 7 --square-len 0.035 --marker-len 0.026 --out-intr intr.yaml
```

**2. Extrinsics** - lay the board flat on the table against marked spots whose origin `(x, y)` and
yaw you measured from the arm base (the board's (0,0) corner is the origin; +X along squares-x, +Y
along squares-y). Take ONE photo and solve, reusing the intrinsics:
```
python scripts/perception_calibrate.py extrinsics --image table.jpg --intr intr.yaml \
    --origin 0.40 -0.12 --yaw-deg 0 --z-table 0.0 --out configs/calib/table_cam.yaml
```
Or both in one call with `... full --images 'calib_shots/*.jpg' --table-image table.jpg --origin ...`.

This needs no arm movement: you only measure where the board sits relative to the arm base (a ruler
from the base, plus a marked square for yaw). The saved `configs/calib/table_cam.yaml` holds K,
distortion, the base-frame extrinsics and `z_table`; the control host loads it with pure numpy.

**Touch-off variant (optional, later, more accurate):** instead of measuring the board by ruler,
jog the arm TCP to 3-4 known board corners and record the base-frame TCP positions; solvePnP those
against the same image corners. More accurate because it ties the camera to the arm's own frame
directly. Not implemented yet (it needs the real arm); the ruler method above is enough to start.

A **synthetic sim calibration** is shipped so the path runs without a physical camera:
`python scripts/perception_make_sim_calib.py --arm dev_arm` writes `configs/calib/table_cam.yaml`
from the Isaac "behind" view geometry. Replace it with the real calibration above.

## Turning it on

In the config (a temp copy if you do not want to touch the default), set:
```
perception:
  source: camera           # default is sim (unchanged behaviour)
  calib: configs/calib/table_cam.yaml
  detector: {backend: owlv2}   # owlv2 (default) or gdino
```
With `source: sim` nothing changes. With `source: camera` the app fills object positions from the
webcam. The detector runs on the VM GPU and needs the `vision` extra / a torch python; if it cannot
load, the app logs a warning and keeps running (objects just are not updated).

## Running it

VM (control host), with a spare-port temp config so it does not touch the live stack on 47100:
```
# intrinsics/extrinsics already in configs/calib/table_cam.yaml, perception.source: camera
PYTHONPATH=src .venv/bin/python -m rover_intent.app --config /tmp/rover_cam.yaml --backend isaac
# the detector needs torch+transformers: either install the vision extra into .venv, or run the
# app with a torch python that also has this repo's deps.
```
Laptop ("bench-laptop" at the bench), plain webcam, 2 Hz:
```
python laptop/table_cam.py --host rover-vm --hz 2            # direct UDP over the tailnet
# or, if UDP can't reach the VM, through the relay (TCP, no datagram size limit):
ssh -N -L 47120:127.0.0.1:47120 rover-vm                        # leave running
python laptop/table_cam.py --tunnel 127.0.0.1:47120 --hz 2
```
The laptop only captures and sends JPEG frames; all detection is on the VM. A 640x480 JPEG at
quality 70 is ~15-30 KB; prefer `--tunnel` for reliability (a large UDP datagram can IP-fragment
and drop).

## Expected accuracy (sim validation)

`scripts/perception_sim_eval.py` (synthetic pinhole at the sim camera pose, exact silhouette boxes
plus 2 px detector-localisation noise, 40 random layouts of all 3 objects; `logs/perception/
sim_eval.json`). Gate: median < 2 cm, 90th pct < 3 cm.

| mount | median | 90th pct | max | result |
|---|---|---|---|---|
| sim "behind" (oblique) | 0.56 cm | 1.09 cm | 1.77 cm | PASS |
| high overhead ~1.2 m | 0.33 cm | 0.57 cm | 0.88 cm | PASS |
| oblique front 45 deg | 0.29 cm | 0.52 cm | 0.89 cm | PASS |

Detector-in-the-loop on rendered scenes (`scripts/perception_detector_eval.py`, 30 layouts,
`logs/perception/detector_eval.json`), end-to-end box -> table position:

| detector | VRAM | latency (med) | detection rate | false pos | end-to-end median / p90 |
|---|---|---|---|---|---|
| OWLv2 base (owlv2) | 1.95 GB | 181 ms | 0.79 | 0 | 0.16 / 0.35 cm |
| Grounding DINO tiny (gdino) | 2.47 GB | 141 ms | 0.99 | 82 (over-detects) | 0.14 / 0.25 cm |

Both pass the accuracy gate. **OWLv2 is the default**: lighter on VRAM and zero false positives (a
robot must not act on phantom objects). Grounding DINO has higher recall but over-detects on the
crude flat-shaded test scenes (needs a confidence/IoU filter). Both fit comfortably in the ~8 GB of
spare VRAM alongside Thoth GPU jobs.

These end-to-end numbers are on **clean rendered objects** (perfect cylinders/box on a plain
background), so they are optimistic: the geometry adds under ~1 cm, but a real webcam's lighting,
textures and looser detector boxes are the real-world risk and are **not yet measured on a real
camera** (see failure modes).

## Failure modes

- **Camera moved after calibration** -> every position is biased. Re-run the extrinsics step.
- **Wrong table height / object on a riser** -> z is pinned to `z_table`; an object not on the
  calibrated surface projects to the wrong `(x, y)`. Keep objects on the table.
- **Oblique mount + large object** -> the bottom-of-box bias grows; `refine` handles it with the
  known size, but a very low grazing view is still worse. Mount steeper.
- **Detector miss / false positive** -> missing objects are simply not updated (the skill still
  refuses objects it cannot see); false positives are rejected if outside the workspace or below
  `min_conf`, but a confident false box inside the workspace would place a phantom object. OWLv2
  (0 FP in sim) is the safer default; tune `perception.min_conf`.
- **Occlusion / objects touching** -> boxes merge; separate objects or use a steeper view.
- **Large frames over UDP** -> drops; use `--tunnel`.
- **Lens distortion** -> handled if the intrinsics captured it (`dist`); a very wide/fisheye lens
  needs enough varied calibration views.

## Files

- `src/rover_intent/perception/` - camera model, object library, projection, calibration, detector,
  source, synthetic renderers.
- `configs/calib/table_cam.yaml` - saved calibration (sim default; replace with the real one).
- `scripts/perception_make_sim_calib.py`, `scripts/perception_calibrate.py`,
  `scripts/perception_sim_eval.py`, `scripts/perception_detector_eval.py`.
- `laptop/table_cam.py` - the laptop capture-and-send client.
- `tests/test_perception.py` - projection maths, calibration round-trip, config default, transport.
