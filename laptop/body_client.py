"""LAPTOP body client: webcam -> MediaPipe Pose + Hand landmarkers -> wrist/shoulder/hand-open -> robot (UDP, tailnet).

Standalone: needs only this file. Uses the MediaPipe *Tasks* API (mediapipe >= 0.10.x; 1.0 removed the old
`mp.solutions`). The two model files (~17 MB) download next to this script on first run.

    python -m venv rover-body
    rover-body\\Scripts\\activate           (Windows)   |   source rover-body/bin/activate   (mac/Linux)
    pip install mediapipe opencv-python numpy
    python body_client.py [--camera 0] [--side right]

In the preview window:  F = clutch (hold still 2 s to calibrate, then the arm follows; F again = freeze)
                        C = calibrate (about 30 s): stand on the green guide, follow the prompts, then touch
                            4 magenta targets. X = redo only the axis that failed. The profile is saved and reused.
                        S = stop   R = reset   M = flip left/right (if the robot goes the wrong way)   Q/Esc = quit
Hands-free clutch: hold an open palm UP (above your elbow) for 1 s, or keep your LEFT hand raised above your
shoulder (hold-to-follow: lower it and the arm freezes, so you can reposition yourself).
Make a fist to close the gripper, open your hand to release (stand-in until EEG is live).
Stand ~1.5-2 m from the camera with your shoulders, elbow, wrist and hips in view. Gesture mode (default on the VM):
  swing your hand left/right -> the arm rotates left/right
  raise/lower your hand      -> the gripper goes up/down
  straighten/bend your elbow -> the gripper reaches farther/closer
The green ghost ball in Isaac shows where the gripper is heading (red = out of reach, clamped).
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
SKEL = [11, 12, 13, 14, 15, 16, 23, 24, 0]      # shoulders, elbows, wrists, hips, nose -> HUD skeleton
ELBOW = {"right": 14, "left": 13}
OTHER_SHOULDER = {"right": 11, "left": 12}
HIPS = (23, 24)
MODELS = {
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                                 "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                            "hand_landmarker/float16/latest/hand_landmarker.task",
}


class Link:
    """UDP straight to the VM, or (--tunnel host:port) newline-JSON over TCP through an SSH tunnel to
    scripts/tcp_relay.py on the VM. Same messages either way."""

    def __init__(self, host, port, tunnel=None):
        self.tunnel = tunnel
        if tunnel:
            th, tp = tunnel.rsplit(":", 1)
            self.sock = socket.create_connection((th, int(tp)), timeout=5)
            self.sock.setblocking(False)
            self._buf = b""
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setblocking(False)
            self.addr = (host, port)

    def send(self, msg: dict):
        data = json.dumps(msg).encode()
        if self.tunnel:
            self.sock.setblocking(True)
            self.sock.sendall(data + b"\n")
            self.sock.setblocking(False)
        else:
            self.sock.sendto(data, self.addr)

    def poll(self):
        """All messages received so far (non-blocking)."""
        out = []
        while True:
            try:
                data = self.sock.recv(65536) if self.tunnel else self.sock.recvfrom(65536)[0]
            except (BlockingIOError, ConnectionResetError, OSError):
                break
            if self.tunnel:
                if not data:
                    break
                self._buf += data
                *lines, self._buf = self._buf.split(b"\n")
                out += [json.loads(x) for x in lines if x.strip()]
            else:
                try:
                    out.append(json.loads(data))
                except ValueError:
                    pass
        return out

    def wait(self, timeout, want=lambda m: True):
        """Block up to `timeout` s for a message matching `want`."""
        import time as _t
        end = _t.time() + timeout
        while _t.time() < end:
            for m in self.poll():
                if want(m):
                    return m
            _t.sleep(0.02)
        return None


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
    """-> (message fields or None, openness or None, chosen hand landmarks or None).
    mirror: the webcam flips the image, so MediaPipe labels your right arm "left": track the other side's landmarks
    and flip x back, which restores your real right arm in un-mirrored coordinates."""
    track = side if not mirror else ("left" if side == "right" else "right")
    sh_i, wr_i = SIDES[track]
    hand = pick_hand(pose_res, hand_res, wr_i)
    openness = hand_openness(hand) if hand is not None else None  # None = not found (NOT "open")
    if not pose_res.pose_world_landmarks:
        return None, openness, hand
    w = pose_res.pose_world_landmarks[0]
    vis = min(w[sh_i].visibility or 0.0, w[wr_i].visibility or 0.0) > 0.5
    sx = -1.0 if mirror else 1.0
    pt = lambda i: [sx * w[i].x, w[i].y, w[i].z]
    hips = [(pt(HIPS[0])[k] + pt(HIPS[1])[k]) / 2 for k in range(3)]
    other_wr = SIDES["left" if track == "right" else "right"][1]
    img = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None
    skel = [[round(1 - img[i].x if mirror else img[i].x, 3), round(img[i].y, 3)] for i in SKEL] if img else None
    other_ok = (w[other_wr].visibility or 0.0) > 0.5   # MediaPipe guesses unseen wrists: don't trust those
    return {"wrist": pt(wr_i), "shoulder": pt(sh_i), "elbow": pt(ELBOW[track]),
            "shoulder_other": pt(OTHER_SHOULDER[track]), "wrist_other": pt(other_wr) if other_ok else None,
            "hips": hips, "side": side,
            "skel2d": skel, "hand_open": openness, "visible": vis}, openness, hand


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


GUIDE = {"width": (0.18, 0.30), "cx": (0.42, 0.58), "cy": (0.22, 0.45)}   # shoulders: width, centre x, height


def stand_guide(frame, pose_res):
    """Stand-here guide: a target outline for your shoulders; green when you're at the right distance / place.
    Returns (in_position, hint)."""
    h, w = frame.shape[:2]
    gx0, gx1 = 0.5 - sum(GUIDE["width"]) / 4, 0.5 + sum(GUIDE["width"]) / 4
    gy = sum(GUIDE["cy"]) / 2
    ok, hint = False, "step into view"
    if pose_res.pose_landmarks:
        lm = pose_res.pose_landmarks[0]
        l, r = lm[11], lm[12]
        width, cx, cy = abs(l.x - r.x), (l.x + r.x) / 2, (l.y + r.y) / 2
        checks = [(width < GUIDE["width"][0], "step CLOSER"), (width > GUIDE["width"][1], "step BACK"),
                  (cx < GUIDE["cx"][0], "move -> (on screen)"), (cx > GUIDE["cx"][1], "move <- (on screen)"),
                  (cy < GUIDE["cy"][0], "lower the camera / sit up less"), (cy > GUIDE["cy"][1], "raise the camera / stand up")]
        bad = [t for c, t in checks if c]
        ok, hint = not bad, (bad[0] if bad else "IN POSITION")
        cv2.line(frame, (int(l.x * w), int(l.y * h)), (int(r.x * w), int(r.y * h)), (0, 220, 0) if ok else (0, 0, 255), 3)
    col = (0, 220, 0) if ok else (0, 0, 255)
    y = int(gy * h)
    cv2.rectangle(frame, (int(gx0 * w), y - 30), (int(gx1 * w), y + 30), col, 2)          # shoulder target
    cv2.ellipse(frame, (w // 2, y - 90), (45, 55), 0, 0, 360, col, 2)                      # head outline
    cv2.putText(frame, hint, (int(gx0 * w), y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    return ok, hint


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
    ap.add_argument("--tunnel", help="host:port of an SSH tunnel to scripts/tcp_relay.py (when UDP can't reach the VM)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--side", choices=SIDES, default="right")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--image", help="test on one image instead of the camera (prints, sends nothing)")
    ap.add_argument("--mirror", action="store_true", help="flip left/right (webcams that mirror the image); M toggles")
    ap.add_argument("--record", help="append every frame's landmarks to this .jsonl (for debugging the mapping)")
    ap.add_argument("--minimap", action="store_true", help="show the old top/side mini-map (the ghost in Isaac replaces it)")
    args = ap.parse_args()

    if args.image:
        pose_res, hand_res = Tracker(video=False)(cv2.imread(args.image))
        fields, openness, _ = extract(pose_res, hand_res, args.side)
        print(json.dumps({"pose_found": fields is not None, "hand_found": bool(hand_res.hand_landmarks),
                          **(fields or {})}, indent=1))
        return

    link = Link(args.host, args.port, args.tunnel)
    print("link:", f"tunnel {args.tunnel}" if args.tunnel else f"udp {args.host}:{args.port}")
    status, robot = "press F to follow", {}
    mirror, gate = args.mirror, JumpGate()
    seq, sent_t, rtt_ms = 0, {}, None
    rec = open(args.record, "a") if args.record else None

    def say(text):
        link.send({"kind": "utterance", "t": time.time(), "text": text})

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
        in_pos, _hint = stand_guide(frame, pose_res)
        if fields:
            fields["in_position"] = in_pos
        if fields and now - last_send >= 1.0 / args.hz:
            last_send, seq = now, seq + 1
            sent_t[seq] = now
            if len(sent_t) > 300:
                sent_t.pop(min(sent_t))
            fields["visible"] = visible
            link.send({"kind": "body", "t": now, "seq": seq, "rtt_ms": rtt_ms, **fields})
        if pose_res.pose_landmarks:
            draw(frame, pose_res.pose_landmarks[0], vision.PoseLandmarksConnections.POSE_LANDMARKS, (0, 200, 0))
        if hand is not None:
            draw(frame, hand, vision.HandLandmarksConnections.HAND_CONNECTIONS, (255, 128, 0))
        for msg in link.poll():  # replies (to F/S/R) and 10 Hz robot status
            if msg.get("kind") == "status":
                robot = msg
                if msg.get("set_mirror") and not mirror:   # warm-up detected a mirrored webcam
                    mirror = True
                    status = "mirrored webcam detected - flipped left/right"
                es = msg.get("echo_seq")
                if es in sent_t and msg.get("echo_age") is not None:  # RTT minus the time the app held it
                    r = (time.time() - sent_t[es] - msg["echo_age"]) * 1000
                    rtt_ms = r if rtt_ms is None else 0.8 * rtt_ms + 0.2 * r
            else:
                status = msg.get("say", status)
        if args.minimap:
            minimap(frame, robot)
        frames += 1
        if now - fps_t >= 1.0:
            fps, frames, fps_t = frames / (now - fps_t), 0, now
        grip = "no hand" if openness is None else "FIST" if openness < 0.35 else "open" if openness > 0.6 else "..."
        cv2.putText(frame, f"{args.side} arm {'TRACKED' if visible else 'NOT VISIBLE'}  hand:{grip}  "
                           f"{fps:.0f} fps{'  MIRRORED' if mirror else ''}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 200, 0) if visible else (0, 0, 255), 2)
        cl = robot.get("clutch", "?")
        why = f" ({robot['release_reason']})" if robot.get("release_reason") else ""
        cl_txt = {"idle": f"FROZEN{why} - press F, hold an open palm up 1 s, or raise your LEFT hand",
                  "calibrating": f"RE-CENTRE: hold still... {int(100 * robot.get('clutch_progress', 0))}%",
                  "following": f"FOLLOWING ({robot.get('trigger')})"}.get(cl, cl)
        cv2.putText(frame, cl_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    {"following": (0, 220, 0), "calibrating": (255, 160, 60)}.get(cl, (200, 200, 200)), 2)
        lat = f"rtt {rtt_ms:.0f} ms" if rtt_ms is not None else "rtt ?"
        cv2.putText(frame, f"robot: {status}   {lat}   loss {100 * robot.get('loss', 0):.0f}%   "
                           f"ghost {robot.get('ghost') or '-'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cal = robot.get("calib")
        if cal and cal.get("prompt"):
            cv2.putText(frame, f"CALIBRATION: {cal['prompt']}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            if cal.get("progress") is not None:
                cv2.rectangle(frame, (10, 160), (10 + int(300 * cal["progress"]), 170), (0, 255, 255), -1)
            if cal.get("note"):
                cv2.putText(frame, cal["note"], (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        elif cal and cal.get("result"):
            r = cal["result"]
            txt = (f"calibration {'ACCEPTED' if r.get('accepted') else 'NOT accepted'}: {r.get('errors_cm')} cm, "
                   f"fit {r.get('residual_deg')} deg" + (f", redo {r['bad_axis']} (X)" if r.get("bad_axis") else ""))
            cv2.putText(frame, txt, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        if robot.get("ghost") == "clamped":
            cv2.putText(frame, "OUT OF REACH - the gripper is clamped at the edge", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(frame, "C calibrate | X redo bad axis | F clutch | S stop | R reset | M flip | Q quit",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("rover body client", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("f"):
            link.send({"kind": "clutch", "t": time.time(), "action": "toggle"})
        elif k == ord("s"):
            say("stop")
        elif k == ord("r"):
            say("reset")
        elif k in (ord("c"), ord("w")):
            link.send({"kind": "clutch", "t": time.time(), "action": "calibrate"})
        elif k == ord("x"):
            link.send({"kind": "clutch", "t": time.time(), "action": "redo_axis"})
        elif k == ord("m"):
            mirror = not mirror
            if robot.get("clutch") == "following":  # re-calibrate so flipping doesn't make the arm jump
                link.send({"kind": "clutch", "t": time.time(), "action": "toggle"})
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
