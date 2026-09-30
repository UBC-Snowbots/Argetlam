"""LAPTOP body client: webcam -> MediaPipe Pose + Hand landmarkers -> wrist/shoulder/hand-open -> robot (UDP, tailnet).

Standalone: needs only this file. Uses the MediaPipe *Tasks* API (mediapipe >= 0.10.x; 1.0 removed the old
`mp.solutions`). The two model files (~17 MB) download next to this script on first run.

    python -m venv rover-body
    rover-body\\Scripts\\activate           (Windows)   |   source rover-body/bin/activate   (mac/Linux)
    pip install mediapipe opencv-python numpy
    python body_client.py [--camera 0] [--side right]

In the preview window:  F = "follow me" (the arm stays put; your next pose is neutral)
                        S = stop   R = reset   M = flip left/right (if the robot goes the wrong way)   Q/Esc = quit
Make a fist to close the gripper, open your hand to release (stand-in until EEG is live).
Stand ~1.5-2 m from the camera with your shoulder, elbow and wrist in view. Watch Isaac from BEHIND the arm:
hand to your right -> gripper right, toward the screen -> forward, up -> up. The mini-map (top right) shows the
gripper (white), where you are steering it (green, red = outside reach) and the reachable box.
Only the chosen arm's hand is used (--side, default right); the other hand is ignored.
"""
import argparse
import json
import socket
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

DEFAULT_HOST = "rover-vm"  # rover-vm on the tailnet
PORT = 47100
SIDES = {"right": (12, 16), "left": (11, 15)}  # (shoulder, wrist) pose landmark ids
MODELS = {
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                                 "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                            "hand_landmarker/float16/latest/hand_landmarker.task",
}


def model_path(name: str) -> str:
    p = Path(__file__).resolve().parent / name
    if not p.exists():
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(MODELS[name], p)
    return str(p)


def hand_openness(lm) -> float:
    """~0 = fist, ~1 = open hand: fingertip spread relative to palm size (normalized image landmarks)."""
    w = lm[0]
    tips = [lm[i] for i in (8, 12, 16, 20)]
    spread = sum(((t.x - w.x) ** 2 + (t.y - w.y) ** 2) ** 0.5 for t in tips) / 4
    palm = ((lm[9].x - w.x) ** 2 + (lm[9].y - w.y) ** 2) ** 0.5 + 1e-6
    return max(0.0, min(1.0, (spread / palm - 1.0) / 0.9))


class Tracker:
    def __init__(self, video: bool):
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        self.video = video
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path("pose_landmarker_full.task")), running_mode=mode))
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path("hand_landmarker.task")), running_mode=mode,
            num_hands=2))  # both, then keep only the one attached to the tracked arm
        self._t0 = time.time()

    def __call__(self, bgr):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if self.video:
            ts = int((time.time() - self._t0) * 1000)
            return self.pose.detect_for_video(img, ts), self.hands.detect_for_video(img, ts)
        return self.pose.detect(img), self.hands.detect(img)


def draw(frame, landmarks, connections, color):
    h, w = frame.shape[:2]
    pts = [(int(l.x * w), int(l.y * h)) for l in landmarks]
    for c in connections:
        cv2.line(frame, pts[c.start], pts[c.end], color, 2)
    for p in pts:
        cv2.circle(frame, p, 3, color, -1)


def pick_hand(pose_res, hand_res, wr_i, max_dist=0.12):
    """The detected hand whose wrist is closest (image coords) to the tracked arm's pose wrist. Handedness labels are
    not trusted: they flip with mirrored webcams. Returns landmarks or None."""
    if not hand_res.hand_landmarks or not pose_res.pose_landmarks:
        return None
    pw = pose_res.pose_landmarks[0][wr_i]
    d = [((h[0].x - pw.x) ** 2 + (h[0].y - pw.y) ** 2) ** 0.5 for h in hand_res.hand_landmarks]
    i = int(min(range(len(d)), key=d.__getitem__))
    return hand_res.hand_landmarks[i] if d[i] <= max_dist else None


def extract(pose_res, hand_res, side, mirror=False):
    """-> (message fields or None, openness or None, chosen hand landmarks or None)"""
    sh_i, wr_i = SIDES[side]
    hand = pick_hand(pose_res, hand_res, wr_i)
    openness = hand_openness(hand) if hand is not None else None  # None = not found (NOT "open")
    if not pose_res.pose_world_landmarks:
        return None, openness, hand
    w = pose_res.pose_world_landmarks[0]
    vis = min(w[sh_i].visibility or 0.0, w[wr_i].visibility or 0.0) > 0.5
    sx = -1.0 if mirror else 1.0
    return {"wrist": [sx * w[wr_i].x, w[wr_i].y, w[wr_i].z], "shoulder": [sx * w[sh_i].x, w[sh_i].y, w[sh_i].z],
            "hand_open": openness, "visible": vis}, openness, hand


class JumpGate:
    """Drop physically implausible wrist jumps (tracking glitches on fast moves), for at most `max_skip` frames."""

    def __init__(self, max_speed=6.0, max_skip=6):
        self.max_speed, self.max_skip = max_speed, max_skip
        self.last, self.t, self.skipped = None, None, 0

    def ok(self, rel, t):
        if self.last is not None:
            v = sum((a - b) ** 2 for a, b in zip(rel, self.last)) ** 0.5 / max(t - self.t, 1e-3)
            if v > self.max_speed and self.skipped < self.max_skip:
                self.skipped += 1
                return False
        self.last, self.t, self.skipped = rel, t, 0
        return True


def minimap(frame, st):
    """Top view (forward/left) and side view (forward/up) of the reachable box, gripper and steering target."""
    if not st or "ws_min" not in st:
        return
    lo, hi = st["ws_min"], st["ws_max"]
    H, W = frame.shape[:2]
    size, pad = 150, 10
    for k, (ax_h, ax_v, title) in enumerate([(1, 0, "top (you: right/fwd)"), (2, 0, "side (up/fwd)")]):
        x0, y0 = W - size - pad, pad + k * (size + 25)
        cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (60, 60, 60), -1)
        cv2.putText(frame, title, (x0, y0 + size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

        def px(p):  # horizontal: -y (screen right) for top view, z (up) drawn vertically for side view
            if ax_h == 1:
                u = (hi[1] - p[1]) / (hi[1] - lo[1])            # robot -y = right
                v = 1 - (p[0] - lo[0]) / (hi[0] - lo[0])         # forward = up on the map
            else:
                u = (p[0] - lo[0]) / (hi[0] - lo[0])             # forward = right
                v = 1 - (p[2] - lo[2]) / (hi[2] - lo[2])         # up = up
            return int(x0 + min(max(u, -0.1), 1.1) * size), int(y0 + min(max(v, -0.1), 1.1) * size)
        cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (200, 200, 200), 1)
        if st.get("target"):
            cv2.circle(frame, px(st["target"]), 6, (0, 0, 255) if st.get("clipped") else (0, 220, 0), 2)
        if st.get("tcp"):
            cv2.circle(frame, px(st["tcp"]), 4, (255, 255, 255), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--side", choices=SIDES, default="right")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--image", help="test on one image instead of the camera (prints, sends nothing)")
    ap.add_argument("--mirror", action="store_true", help="flip left/right (webcams that mirror the image); M toggles")
    ap.add_argument("--record", help="append every frame's landmarks to this .jsonl (for debugging the mapping)")
    args = ap.parse_args()

    if args.image:
        pose_res, hand_res = Tracker(video=False)(cv2.imread(args.image))
        fields, openness, _ = extract(pose_res, hand_res, args.side)
        print(json.dumps({"pose_found": fields is not None, "hand_found": bool(hand_res.hand_landmarks),
                          **(fields or {})}, indent=1))
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    addr = (args.host, args.port)
    status, robot = "press F to follow", {}
    mirror, gate = args.mirror, JumpGate()
    rec = open(args.record, "a") if args.record else None

    def say(text):
        sock.sendto(json.dumps({"kind": "utterance", "t": time.time(), "text": text}).encode(), addr)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"camera {args.camera} did not open (try --camera 1)")
    track = Tracker(video=True)
    last_send, fps_t, frames, fps = 0.0, time.time(), 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pose_res, hand_res = track(frame)
        fields, openness, hand = extract(pose_res, hand_res, args.side, mirror)
        visible = bool(fields and fields["visible"])
        now = time.time()
        if fields and visible:
            rel = [a - b for a, b in zip(fields["wrist"], fields["shoulder"])]
            if not gate.ok(rel, now):
                visible = False  # glitch: skip this frame, the robot holds for a moment
        if rec and fields:
            rec.write(json.dumps({"t": now, "mirror": mirror, **fields}) + "\n")
        if fields and visible and now - last_send >= 1.0 / args.hz:
            last_send = now
            sock.sendto(json.dumps({"kind": "body", "t": now, **fields}).encode(), addr)
        if pose_res.pose_landmarks:
            draw(frame, pose_res.pose_landmarks[0], vision.PoseLandmarksConnections.POSE_LANDMARKS, (0, 200, 0))
        if hand is not None:
            draw(frame, hand, vision.HandLandmarksConnections.HAND_CONNECTIONS, (255, 128, 0))
        while True:  # replies (to F/S/R) and 10 Hz robot status for the mini-map
            try:
                msg = json.loads(sock.recvfrom(65536)[0])
            except (BlockingIOError, ConnectionResetError, OSError, ValueError):
                break
            if msg.get("kind") == "status":
                robot = msg
            else:
                status = msg.get("say", status)
        minimap(frame, robot)
        frames += 1
        if now - fps_t >= 1.0:
            fps, frames, fps_t = frames / (now - fps_t), 0, now
        grip = "no hand" if openness is None else "FIST" if openness < 0.35 else "open" if openness > 0.6 else "..."
        cv2.putText(frame, f"{args.side} arm {'TRACKED' if visible else 'NOT VISIBLE'}  hand:{grip}  "
                           f"{fps:.0f} fps{'  MIRRORED' if mirror else ''}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 200, 0) if visible else (0, 0, 255), 2)
        cv2.putText(frame, f"robot: {status}  [{robot.get('mode', '?')}]", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2)
        if robot.get("clipped"):
            cv2.putText(frame, "OUT OF REACH - come back toward the box", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
        cv2.putText(frame, "F follow | S stop | R reset | M flip L/R | Q quit", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("rover body client", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("f"):
            say("follow me")
        elif k == ord("s"):
            say("stop")
        elif k == ord("r"):
            say("reset")
        elif k == ord("m"):
            mirror = not mirror
            say("follow me")  # re-clutch so flipping doesn't make the arm jump
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
