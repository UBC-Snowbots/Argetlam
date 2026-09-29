"""LAPTOP body client: webcam -> MediaPipe Pose + Hand landmarkers -> wrist/shoulder/hand-open -> robot (UDP, tailnet).

Standalone: needs only this file. Uses the MediaPipe *Tasks* API (mediapipe >= 0.10.x; 1.0 removed the old
`mp.solutions`). The two model files (~17 MB) download next to this script on first run.

    python -m venv rover-body
    rover-body\\Scripts\\activate           (Windows)   |   source rover-body/bin/activate   (mac/Linux)
    pip install mediapipe opencv-python numpy
    python body_client.py [--camera 0] [--side right]

In the preview window:  F = "follow me" (the arm stays put; your next pose is neutral)
                        S = stop        R = reset        Q / Esc = quit
Make a fist to close the gripper, open your hand to release (stand-in until EEG is live).
Stand ~1.5-2 m from the camera with your shoulder, elbow and wrist in view.
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
            num_hands=1))
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


def extract(pose_res, hand_res, side):
    """-> (message fields or None, openness)"""
    sh_i, wr_i = SIDES[side]
    openness = hand_openness(hand_res.hand_landmarks[0]) if hand_res.hand_landmarks else 1.0
    if not pose_res.pose_world_landmarks:
        return None, openness
    w = pose_res.pose_world_landmarks[0]
    vis = min(w[sh_i].visibility or 0.0, w[wr_i].visibility or 0.0) > 0.5
    return {"wrist": [w[wr_i].x, w[wr_i].y, w[wr_i].z], "shoulder": [w[sh_i].x, w[sh_i].y, w[sh_i].z],
            "hand_open": openness, "visible": vis}, openness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--side", choices=SIDES, default="right")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--image", help="test on one image instead of the camera (prints, sends nothing)")
    args = ap.parse_args()

    if args.image:
        pose_res, hand_res = Tracker(video=False)(cv2.imread(args.image))
        fields, openness = extract(pose_res, hand_res, args.side)
        print(json.dumps({"pose_found": fields is not None, "hand_found": bool(hand_res.hand_landmarks),
                          **(fields or {})}, indent=1))
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    addr = (args.host, args.port)
    status = "press F to follow"

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
        fields, openness = extract(pose_res, hand_res, args.side)
        visible = bool(fields and fields["visible"])
        now = time.time()
        if fields and now - last_send >= 1.0 / args.hz:
            last_send = now
            sock.sendto(json.dumps({"kind": "body", "t": now, **fields}).encode(), addr)
        if pose_res.pose_landmarks:
            draw(frame, pose_res.pose_landmarks[0], vision.PoseLandmarksConnections.POSE_LANDMARKS, (0, 200, 0))
        if hand_res.hand_landmarks:
            draw(frame, hand_res.hand_landmarks[0], vision.HandLandmarksConnections.HAND_CONNECTIONS, (255, 128, 0))
        try:  # replies from the robot (to F/S/R)
            data, _ = sock.recvfrom(65536)
            status = json.loads(data).get("say", status)
        except (BlockingIOError, ConnectionResetError, OSError):
            pass
        frames += 1
        if now - fps_t >= 1.0:
            fps, frames, fps_t = frames / (now - fps_t), 0, now
        grip = "FIST" if openness < 0.35 else "open" if openness > 0.6 else "..."
        cv2.putText(frame, f"{args.side} arm {'TRACKED' if visible else 'NOT VISIBLE'}  hand:{grip} ({openness:.2f})  "
                           f"{fps:.0f} fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 200, 0) if visible else (0, 0, 255), 2)
        cv2.putText(frame, f"robot: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "F follow | S stop | R reset | Q quit", (10, frame.shape[0] - 15),
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
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
