# Perception: object positions from a rover camera (sim + real)

Goal: let the voice skills ("pick up the cup", "put the cup next to the bottle") run from CAMERA-derived
object positions instead of sim ground truth, and score that automatically against ground truth in sim
before it ever drives the real arm. The sim camera is a 1:1 DIGITAL TWIN of the real rover camera: same
pose, same intrinsics, same downstream path, so switching sim <-> real is a config flip.

This is the ROVER-camera path. The fixed single-table-camera variant and the ChArUco calibration
how-to live in docs/TABLE_CAM.md; both share `src/rover_intent/perception/`.

## Architecture

```
  frame source                         CameraFrame (one JSON message, base64 JPEG + metadata)
  ------------                         ------------------------------------------------------
  sim: Isaac bridge rover camera  ─┐
       (sim/rover_camera.py,       │
        +ground-truth objects)     ├─► app.py ─► perception detector ─► table-plane / direct 3D
  real: device sender              │      │          (perception/detectors.py)      │
       (perception/camera_tx.py)  ─┘      │                                          ▼
  real: VM ingest of an RTSP/MJPEG URL    │                                   PerceptionTracker
       (camera_tx.py --source <url>)      │                         (confidence gate, staleness, EMA)
                                          │                                          │
                                          ├─► HUD video panel (scripts/hud.py)       ▼
                                          └─► viewer (scripts/sim_cam_viewer.py)  Scene positions
                                                 overlay boxes + detections           │
                                                                                      ▼
                                                                    skills (refuse an object not seen)
```

* `perception/camera.py` - pinhole camera model (OpenCV optical frame) with extrinsics in the arm base
  frame; pure numpy for the no-distortion path. `camera_from_config()` builds it from the arm profile
  `camera:` block (mount + model/calib).
* `perception/objectlib.py` - known objects (name, footprint radius, height, grasp height) = the sim
  `scene:`; what a monocular camera leans on since it cannot measure depth.
* `perception/locator.py` - `TablePlaneLocator`: pixel -> table plane (ray-plane), or box + known object
  -> footprint (silhouette fit). The one place monocular detectors turn pixels into 3D.
* `perception/detectors.py` - the `Detector` protocol (`detect(image, gt_positions=None) -> [Detection]`)
  and four implementations:
  * `GroundTruthDetector` - sim oracle: projects the known objects through the camera and reports their
    TRUE positions (+ optional noise / FOV-dropout / occlusion). Needs no model and no Isaac; the scorer
    baseline, and what makes the whole camera path runnable headless today.
  * `ArucoDetector` - OpenCV ArUco markers; marker id -> object name; the tag centre is lifted to the
    table plane. Reliable, no GPU, no training - the baseline for the real arm.
  * `OpenVocabDetector` - OWLv2 / Grounding DINO text-promptable boxes + the known-size box lift. Needs
    the `vision` extra / a torch python.
  * `NullDetector` - the plugin slot (no-op) for a future open-vocab model (NanoOWL etc.).
* `perception/source.py` - `CameraPerception` (detect + accept/best-per-object) and `PerceptionTracker`
  (confidence gate, staleness timeout, EMA smoothing, provenance). The app drives these.
* `perception/camera_tx.py` - the REAL camera sender that runs on the camera's device.
* `sim/rover_camera.py` - the Isaac-side sim camera (render + stream). Marked UNTESTED in Isaac.
* `scripts/perception_eval.py` - the automatic scorer. `scripts/sim_cam_viewer.py` - the live viewer.

## Config (`camera:` and `perception:`)

Shared defaults, the intrinsics `models:` table and the real-device `real:` block are in
`configs/default.yaml`; the per-arm mounts are in `configs/arms/<arm>.yaml` (they depend on the arm's
workspace geometry). The whole thing is only used when `perception.source: camera`.

```yaml
perception:
  source: sim            # sim (alias ground_truth) = today's behaviour (DEFAULT) | camera
  min_conf: 0.25         # confidence gate
  stale_s: 1.0           # object "seen" only while its last detection is newer than this
  smoothing: 0.5         # EMA weight of the newest frame (1.0 = no smoothing)
  detector:
    backend: aruco       # ground_truth | aruco | owlv2 | gdino | null
    aruco: {dictionary: DICT_4X4_50, marker_length: 0.03, markers: {0: cup, 1: bottle, 2: block}}
    ground_truth: {pos_noise_cm: 0.0, drop_out_of_frame: true, occlusion: false}

camera:
  enabled: true
  mount: rover_front     # a named mount in the arm profile's camera.mounts (switchable at runtime)
  model: vga_default     # intrinsics model in camera.models; or set calib / fx,fy / hfov_deg
  calib: null            # a ChArUco calibration YAML (real K + distortion); OVERRIDES model when set
  width: null            # render/ingest resolution; null -> the model's
  height: null
  rate_hz: 10
  depth: true
  segmentation: true
  models: {vga_default: {width: 640, height: 480, hfov_deg: 60},
           usb_1080p:  {width: 1920, height: 1080, hfov_deg: 70}}
  real: {source: 0, reconnect_s: 2.0, watchdog_s: 5.0}

# per arm profile (configs/arms/<arm>.yaml): WHERE the camera can sit
camera:
  z_table: 0.0
  mounts:
    rover_front: {eye: [-0.35, -0.35, 0.80], look_at: [0.50, 0.00, 0.00]}
    rover_mast:  {eye: [-0.20,  0.00, 1.10], look_at: [0.52, 0.00, 0.00]}
    overhead:    {eye: [ 0.50,  0.00, 1.35], look_at: [0.50, 0.00, 0.00]}
    side:        {eye: [ 0.50, -0.85, 0.55], look_at: [0.50, 0.00, 0.05]}
```

`sim` keeps today's behaviour exactly (objects from Isaac / config). With `camera`, the skills read the
camera's tracked positions; an object not currently seen is refused with a spoken reason ("I don't see
the cup right now"), and the status/HUD reports where each object came from (sim / ground_truth / aruco /
owlv2) with its age.

## Placing the camera (digital twin)

The point of the mounts is to try a camera position in sim and SEE what it covers before building the
real mount. For each candidate mount:

1. Set `camera.mount` (or stream and switch at runtime - see below) and run the sim camera + viewer.
2. In the viewer, check: is the whole reach box in frame? Does the ARM occlude the objects at any pose
   mid-grasp? Are the objects well separated (not merging boxes)?
3. Score it: `scripts/perception_eval.py --backend ground_truth` gives the pure projection error for that
   mount (a steeper / higher mount usually has lower error and less self-occlusion). The three stock
   mounts (rover_front oblique, rover_mast steeper, overhead) all pass the gate in sim; `side` is a
   low grazing view for stress-testing.
4. When you pick one: measure the REAL camera's pose on the rover, put it in that mount's `eye/look_at`
   (or `pos/quat`), and set `camera.calib` / `camera.model` to the real camera's intrinsics. Done - the
   sim and the real camera now match.

Runtime mount switch (no Isaac restart): send the bridge a UDP command on its cmd port,
`{"camera_mount": "overhead"}` (the bridge moves the camera prim). If the live API cannot move the prim,
the bridge says so; restart it with `camera.mount: overhead`.

## Seeing what it sees (the viewer)

The sim camera's frames reach the existing VM HUD automatically (app forwards every CameraFrame to
`net.hud_video_port`); the panel title shows "SIM CAMERA (rover_front)". For the richer overlay use the
standalone viewer, which draws the ground-truth boxes (green) and the detected positions (magenta, with
label + estimated 3D + source), so the error is visible:

```
PYTHONPATH=src python scripts/sim_cam_viewer.py --arm dev_arm --video-port <forward-port> --app 127.0.0.1:<app-port>
```

The SAME viewer serves the real camera (banner "REAL CAMERA"); sim <-> real is a config flip.

## The real camera is on a SEPARATE device

The real rover camera is on a Pi / Jetson / the rover's onboard computer, not the laptop. Run the
sender ON that device (pure Python + OpenCV + stdlib, no repo needed):

```
scp src/rover_intent/perception/camera_tx.py pi@rover:~/
python3 camera_tx.py --host <VM-tailscale-ip> --port 47100 --source 0 --width 640 --height 480 --fps 5
```

`--source` is a camera index (`0`), a device path (`/dev/video0`), or an mjpeg/rtsp URL the device
exposes. If the device only serves a stream, run the same script ON THE VM to ingest it:
`python camera_tx.py --host 127.0.0.1 --source rtsp://rover:8554/cam ...`. It reconnects on failure and
reopens the source if no frame arrives within the watchdog window. The device sends the identical
CameraFrame to the VM app, so the real frames go through the same detector -> projection -> scene ->
viewer path as the sim. Large frames over UDP fragment - keep 640x480, or lower `--quality`.

Calibration for a remote device: capture the ChArUco board frames ON the device (any stills), copy them
to the VM, calibrate there with `scripts/perception_calibrate.py` (docs/TABLE_CAM.md), and point
`camera.calib` at the result. The mount pose you measure on the rover goes into the chosen mount.

## External perception (the device sends results, not frames)

For a depth camera like the RealSense D415 on the laptop, streaming frames to the VM loses the depth (it does
not survive video compression) and the SDK is local anyway. So the device does it all and sends only results:

    laptop: RealSense colour+depth -> detect -> 3D bottom-centre per object -> camera->base calibration
            -> object_poses (UDP, 2-5 Hz, a few hundred bytes) -> VM app :47100
    VM:     perception.source: external -> tracker (min_conf, stale_s, smoothing) -> scene -> voice skills/Nemotron
            -> (mirror_to_sim) Isaac objects moved to the seen positions, so the sim reconstructs the table

Message (types.ObjectPoses): `{"kind": "object_poses", "t", "seq", "source": "realsense", "objects": {name:
{"pos": [x, y, z_bottom], "conf", "err_cm", "desc"}}}`. pos is the object's BOTTOM centre in the ARM BASE frame,
metres. Names must match the scene (cup, bottle, block); unknown names are logged once and ignored. `desc` updates
the description Nemotron sees. Send only what is visible now: an object missing for stale_s counts as unseen and a
skill refuses it. The held object is never moved in the sim (the sim grasp owns it); mirroring is capped at
perception.mirror_hz (2). Reference sender + link test: `laptop/send_object_poses.py`.

To run it: set `perception.source: external` (a temp config, like the camera mode) and restart the app.

## Scoring in sim (the gate)

`scripts/perception_eval.py` compares camera-derived positions to the sim ground truth and gates on
median < 2 cm, p90 < 3 cm, plus detection rate and false positives. It runs HEADLESS (no Isaac) on
synthetic layouts, or on recorded frames (a `.jsonl` of CameraFrame records; the Isaac bridge writes
exactly these, with the ground-truth objects in the same packet):

```
PYTHONPATH=src OMP_NUM_THREADS=4 .venv/bin/python scripts/perception_eval.py --arm dev_arm --backend ground_truth --synth 40
PYTHONPATH=src OMP_NUM_THREADS=4 .venv/bin/python scripts/perception_eval.py --arm dev_arm --backend aruco        --synth 40
# recorded sim frames:
PYTHONPATH=src .venv/bin/python scripts/perception_eval.py --arm dev_arm --backend aruco --frames logs/perception/frames.jsonl
```

Measured in sim (40 synthetic layouts, dev_arm, rover_front mount): ground_truth median 0.00 cm, det
rate 1.00, 0 FP (the oracle); aruco median 0.05 cm, p90 0.10 cm, 0 FP, det rate ~0.71 (markers out of the
oblique view at some layouts). Both PASS. The geometry gate across mounts is in docs/TABLE_CAM.md.

## UNTESTED in Isaac + how to try it when the GPU is free

Thoth holds the shared RTX 5080, so there was NO Isaac launch. The Isaac-side rover camera
(`sim/rover_camera.build_camera` / `RoverCameraStreamer` / `label_object_semantics`, and the bridge
hook) is written against the Isaac Lab Camera API (isaacsim 6.0.1-rc.7; the old `isaacsim.sensors.camera`
is deprecated) but has NOT run in Isaac. The PURE helpers (camera pose/intrinsics from config, GT boxes
from geometry, JPEG/depth encode, CameraFrame build, the offset quaternion) ARE unit-tested with no
Isaac (tests/test_perception_rover.py).

To try it when the GPU is free (spare ports, never the live 47100/47110/47111/etc.; do not touch the
live stack):

```
# 1) Isaac bridge WITH the camera (sets --enable_cameras automatically). GUI or headless:
OMNI_KIT_ACCEPT_EULA=YES ~/projects/ubc-rover-arm/isaac/isaacenv6/bin/python sim/isaac_bridge.py \
    --arm dev_arm --sim-cam --headless --config /tmp/rover_cam.yaml
# /tmp/rover_cam.yaml = a copy of the config with perception.source: camera and SPARE net ports.

# 2) the app on the same spare ports, camera perception on:
PYTHONPATH=src .venv/bin/python -m rover_intent.app --arm dev_arm --backend isaac --config /tmp/rover_cam.yaml

# 3) watch it:
PYTHONPATH=src python scripts/sim_cam_viewer.py --arm dev_arm --video-port <spare> --app 127.0.0.1:<spare>

# 4) score a recorded run:
PYTHONPATH=src .venv/bin/python scripts/perception_eval.py --arm dev_arm --backend aruco --frames <recorded.jsonl>
```

Verify first: the OffsetCfg convention (we use `opencv`, which matches PinholeCamera; the research note
suggested `ros` - if you switch, re-derive the offset quaternion and check the viewer against the GT
boxes), that the instance segmentation ids map to object names (semantics labelling), and that a
runtime `{"camera_mount": ...}` actually moves the prim (else restart with the new mount).
```
