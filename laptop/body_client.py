"""LAPTOP body client: webcam -> MediaPipe Pose + Hand landmarkers -> wrist/shoulder/hand-open -> robot (UDP, tailnet).

Standalone: needs only this file. Uses the MediaPipe *Tasks* API (mediapipe >= 0.10.x; 1.0 removed the old
`mp.solutions`). The two model files (~17 MB) download next to this script on first run.

    python -m venv rover-body
    rover-body\\Scripts\\activate           (Windows)   |   source rover-body/bin/activate   (mac/Linux)
    pip install mediapipe opencv-python numpy
    python body_client.py [--camera 0] [--side right]

In the preview window:  F = clutch (hold still 2 s to re-centre, then the arm follows; F again = freeze)
                        S = stop   R = reset   M = flip left/right (if the robot goes the wrong way)   Q/Esc = quit
Hands-free clutch: hold an open palm UP (above your elbow) for 1 s, or keep your LEFT hand raised above your
shoulder (hold-to-follow: lower it and the arm freezes, so you can reposition yourself).
Grab assist: steer the gripper roughly above an object (it turns magenta in Isaac), then hold a FIST ~0.5 s: the arm
lines up, grabs and lifts it, then hands control back. Hold an OPEN hand ~0.5 s to let go (stand-in until EEG is live).
Stand ~1.5-2 m from the camera with your shoulders, elbow, wrist and hips in view. Two body teleop modes (set by
body.mode on the VM):
  plane      - image-plane teleop (default): your right hand drives the gripper left/right + up/down in the camera
               plane, your left hand height drives reach (depth is ignored). See docs/BODY_TELEOP.md.
  arm_mirror - full-arm joint mirroring: your whole arm's pose drives the robot joints directly (no IK).
The green ghost ball in Isaac shows where the gripper is heading (red = out of reach, clamped).
Only the chosen arm's hand is used (--side, default right); the other hand is ignored.
"""
import argparse
import base64
import json
import os
import socket
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

DEFAULT_HOST = os.environ.get("ROVER_HOST", "127.0.0.1")  # the VM's address (tailnet IP or hostname); or pass --host
PORT = 47100
SIDES = {"right": (12, 16), "left": (11, 15)}  # (shoulder, wrist) pose landmark ids
SKEL = [11, 12, 13, 14, 15, 16, 23, 24, 0]      # shoulders, elbows, wrists, hips, nose -> HUD skeleton
ELBOW = {"right": 14, "left": 13}
OTHER_SHOULDER = {"right": 11, "left": 12}
HIPS = (23, 24)
HANDPTS = {"right": (20, 18), "left": (19, 17)}  # (index, pinky) pose ids -> hand midpoint (arm_mirror wrist bend)
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
        self.reconnects = 0           # tunnel drops we recovered from (shown in the 1 Hz log line)
        self._th = self._tp = None
        if tunnel:
            th, tp = tunnel.rsplit(":", 1)
            self._th, self._tp = th, int(tp)
            self.sock = socket.create_connection((self._th, self._tp), timeout=5)
            self.sock.setblocking(False)
            self._buf = b""
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setblocking(False)
            self.addr = (host, port)

    def _reconnect(self):
        """Tunnel only: re-open a dropped TCP connection to the relay (the SSH tunnel bounced, the relay restarted)."""
        if not self.tunnel:
            return False
        try:
            self.sock.close()
        except OSError:
            pass
        try:
            self.sock = socket.create_connection((self._th, self._tp), timeout=5)
            self.sock.setblocking(False)
            self._buf = b""
            self.reconnects += 1
            return True
        except OSError:
            return False

    def send(self, msg: dict):
        data = json.dumps(msg).encode()
        if self.tunnel:
            try:
                self.sock.setblocking(True)
                self.sock.sendall(data + b"\n")
                self.sock.setblocking(False)
            except OSError:
                self._reconnect()
        else:
            self.sock.sendto(data, self.addr)

    def poll(self):
        """All messages received so far (non-blocking)."""
        out = []
        while True:
            try:
                data = self.sock.recv(65536) if self.tunnel else self.sock.recvfrom(65536)[0]
            except BlockingIOError:
                break
            except (ConnectionResetError, OSError):
                if self.tunnel:
                    self._reconnect()
                break
            if self.tunnel:
                if not data:          # peer closed: relay or tunnel went away
                    self._reconnect()
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


def encode_video_frame(frame, width: int, height: int, quality: int):
    """Downscale `frame` and JPEG-encode it. Returns (base64 str, w, h, jpeg byte count)."""
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    data = buf.tobytes() if hasattr(buf, "tobytes") else bytes(buf)
    return base64.b64encode(data).decode(), width, height, len(data)


class VideoStreamer:
    """Turns the annotated webcam frame into throttled `camera_frame` messages for the HUD video panel.

    Rate-limited to `hz`: frames in between are DROPPED, never queued, so the pose packets' latency is untouched.
    Over plain UDP a frame larger than `max_udp_bytes` would IP-fragment and likely drop, so it warns once and tells
    the operator to lower the quality/size or use --tunnel (the TCP relay has no datagram limit)."""

    def __init__(self, width: int, height: int, quality: int, hz: float, tunnel: bool = False,
                 max_udp_bytes: int = 60000):
        self.width, self.height, self.quality = int(width), int(height), int(quality)
        self.min_dt = 1.0 / max(hz, 0.1)
        self.tunnel, self.max_udp_bytes = bool(tunnel), int(max_udp_bytes)
        self.last = 0.0
        self.seq = self.sent = self.dropped = 0
        self.last_bytes = 0
        self._warned_big = False

    def maybe(self, frame, now: float):
        """A camera_frame dict to send, or None to drop this frame (rate limit)."""
        if now - self.last < self.min_dt:
            self.dropped += 1
            return None
        b64, w, h, nbytes = encode_video_frame(frame, self.width, self.height, self.quality)
        msg = {"kind": "camera_frame", "t": now, "jpeg_b64": b64, "width": w, "height": h,
               "seq": self.seq, "source": "body_cam"}
        approx = len(b64) + 80  # base64 payload plus the small JSON envelope
        if not self.tunnel and approx > self.max_udp_bytes and not self._warned_big:
            self._warned_big = True
            print(f"WARNING: video frame ~{approx} B > UDP safe size; lower --video-quality/--video-width "
                  f"or use --tunnel", flush=True)
        self.last, self.last_bytes = now, nbytes
        self.seq += 1
        self.sent += 1
        return msg


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
            "hips": hips, "side": side, "hand": hand_mid(w, track, pt),
            "skel2d": skel, "hand_open": openness, "visible": vis}, openness, hand


def hand_mid(world, track, pt):
    """World-landmark midpoint of the index + pinky knuckles for the tracked arm (arm_mirror wrist bend), or None
    if either is not clearly visible. `pt(i)` is the extract()-local accessor (applies the mirror x flip).
    Kept separate so the WIP cam-in-sim branch (which also edits body_client) merges cleanly."""
    ix, pk = HANDPTS[track]
    if min(world[ix].visibility or 0.0, world[pk].visibility or 0.0) <= 0.5:
        return None
    return [(pt(ix)[k] + pt(pk)[k]) / 2 for k in range(3)]


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


def plane_tune_msg(op: str, now: float) -> dict:
    """A plane live-tuning command to the app (keys + - [ ] , . g o 0). Handled only when the app is in plane mode."""
    return {"kind": "plane_tune", "t": now, "op": op}


# key -> plane-tune op (reach speed +/-, sensitivity [/], reach thresholds ,/. sooner, g check, o save, 0 reset).
# "+" and "-" accept their shifted twins "=" / "_" so no shift gymnastics are needed.
PLANE_TUNE_KEYS = {ord("+"): "reach_faster", ord("="): "reach_faster",
                   ord("-"): "reach_slower", ord("_"): "reach_slower",
                   ord("["): "less_sensitive", ord("]"): "more_sensitive",
                   ord(","): "reach_back_sooner", ord("."): "reach_fwd_sooner",
                   ord("g"): "tune_check", ord("o"): "save", ord("0"): "reset"}


def _strip0(s: str) -> str:
    """Drop a leading 0 before a decimal point for the compact status line (0.20 -> .20, -0.20 -> -.20)."""
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def reach_status_line(robot: dict) -> str:
    """Compact plane-tuning status line, e.g. 'REACH >> | spd4.0 lat.20 vrt.15 db.015'. '' when not plane mode."""
    pt = robot.get("plane_tune")
    if not pt:
        return ""
    arrow = {">>": ">>", "<<": "<<", "hold": "--"}.get(pt.get("reach_state"), "--")
    lat = _strip0("{:.2f}".format(pt["lateral"]))
    vrt = _strip0("{:.2f}".format(pt["vertical"]))
    db = _strip0("{:.3f}".format(pt["deadband"]))
    return "REACH {} | spd{:.1f} lat{} vrt{} db{}".format(arrow, pt["max_speed"] * 100, lat, vrt, db)


class ButtonPanel:
    """Clickable buttons on the right edge of the window: STOP, then the presets the app reports in its status
    ("return", profile presets, "home_hw" when a real arm is configured). HOME needs a second click within
    CONFIRM_S. The app does all the safety checks (clutch idle, joint limits, workspace box, speed limits)."""
    CONFIRM_S = 3.0
    W, H, GAP = 170, 44, 8

    def __init__(self, margin: int = 10):
        self.names: list[str] = []
        self.rects: list[tuple[str, int, int, int, int]] = []
        self.clicks: list[tuple[int, int]] = []
        self.home_armed_until = 0.0
        self.note = ""
        self.margin = margin   # gap from the right edge; bigger keeps the buttons off the screen edge

    @staticmethod
    def label(name: str) -> str:
        return {"return": "RETURN", "home_hw": "HOME (arm)", "stop": "STOP"}.get(name, name.upper())

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicks.append((x, y))

    def layout(self, width: int):
        x0 = width - self.W - self.margin
        self.rects = [(n, x0, 40 + i * (self.H + self.GAP), self.W, self.H)
                      for i, n in enumerate(["stop"] + self.names)]

    def hit(self, x: int, y: int) -> str | None:
        for n, rx, ry, rw, rh in self.rects:
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return n
        return None

    def press(self, name: str, now: float) -> dict | None:
        """The message for a click on `name` (None = nothing to send yet)."""
        if name == "stop":
            self.note = "STOP sent"
            return {"kind": "utterance", "t": now, "text": "stop"}
        if name == "home_hw":
            if now > self.home_armed_until:
                self.home_armed_until = now + self.CONFIRM_S
                self.note = "click HOME again within 3 s to confirm (all axes drive to their limit switches)"
                return None
            self.home_armed_until = 0.0
            self.note = "HOME confirmed"
            return {"kind": "preset", "t": now, "name": "home_hw", "confirm": True}
        self.note = f"{self.label(name)} requested"
        return {"kind": "preset", "t": now, "name": name, "confirm": False}

    def take(self, now: float) -> list[dict]:
        out = []
        while self.clicks:
            n = self.hit(*self.clicks.pop(0))
            if n is not None:
                msg = self.press(n, now)
                if msg:
                    out.append(msg)
        return out

    def draw(self, frame, robot: dict, now: float):
        names = [n for n in robot.get("presets", []) if isinstance(n, str)]
        if names != self.names:
            self.names = names
        self.layout(frame.shape[1])
        busy = robot.get("hw_busy") or 0.0
        active = robot.get("preset")
        for n, x, y, w, h in self.rects:
            if n == "stop":
                col = (0, 0, 200)
            elif n == "home_hw":
                col = (0, 140, 255) if now <= self.home_armed_until else (60, 60, 140)
            else:
                col = (0, 160, 0) if n == active else (90, 90, 90) if busy else (120, 80, 40)
            cv2.rectangle(frame, (x, y), (x + w, y + h), col, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
            txt = "CONFIRM HOME" if n == "home_hw" and now <= self.home_armed_until else self.label(n)
            cv2.putText(frame, txt, (x + 8, y + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y = 40 + len(self.rects) * (self.H + self.GAP) + 10
        x = frame.shape[1] - self.W - self.margin
        if busy:
            cv2.putText(frame, f"ARM HOMING {busy:.0f}s", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
        elif active:
            cv2.putText(frame, f"moving: {self.label(active)}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        if self.note:
            cv2.putText(frame, self.note[:60], (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)


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
    ap.add_argument("--stream-video", dest="stream_video", action="store_true", default=True,
                    help="send the annotated webcam frame to the HUD video panel in Isaac (default ON)")
    ap.add_argument("--no-video", dest="stream_video", action="store_false", help="do not send the video stream")
    ap.add_argument("--video-width", type=int, default=480)
    ap.add_argument("--video-height", type=int, default=360)
    ap.add_argument("--video-quality", type=int, default=60, help="JPEG quality for the stream (lower -> smaller)")
    ap.add_argument("--video-hz", type=float, default=10.0, help="stream frame rate (separate from the pose rate)")
    ap.add_argument("--window-width", type=int, default=960, help="client window width (smaller keeps it off the screen edge)")
    ap.add_argument("--window-height", type=int, default=540)
    ap.add_argument("--panel-margin", type=int, default=40, help="gap from the right edge to the button panel")
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
    streamer = (VideoStreamer(args.video_width, args.video_height, args.video_quality, args.video_hz,
                              tunnel=bool(args.tunnel)) if args.stream_video else None)
    toast_text, toast_until, toast_ok = "", 0.0, True   # transient ack/refusal banner from the app's reply

    def say(text):
        link.send({"kind": "utterance", "t": time.time(), "text": text})

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"camera {args.camera} did not open (try --camera 1)")
    track = Tracker(video=True)
    panel = ButtonPanel(margin=args.panel_margin)
    cv2.namedWindow("rover body client")
    cv2.setMouseCallback("rover body client", panel.on_mouse)
    last_send, fps_t, frames, fps = 0.0, time.time(), 0, 0.0
    tracked, last_f = 0, 0.0          # frames with the arm visible this second; last F toggle (for debounce)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pose_res, hand_res = track(frame)        # track on the full-res camera frame (best accuracy)
        frame = cv2.resize(frame, (args.window_width, args.window_height))  # smaller display / overlay / stream canvas
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
        if streamer is not None:  # separate, rate-limited, dropped-not-queued: pose latency is never affected
            vmsg = streamer.maybe(frame, now)
            if vmsg is not None:
                link.send(vmsg)
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
                if msg.get("kind") == "reply" and msg.get("say"):  # ack / refusal toast for a button or voice action
                    toast_text = msg["say"]
                    toast_until = time.time() + 3.0
                    toast_ok = bool(msg.get("ok", True)) and not msg.get("busy")
        if args.minimap:
            minimap(frame, robot)
        frames += 1
        tracked += 1 if visible else 0
        if now - fps_t >= 1.0:
            fps = frames / (now - fps_t)
            lat = f"{rtt_ms:.0f} ms" if rtt_ms is not None else "?"
            print(f"[body] {fps:4.1f} fps  tracking {100 * tracked / max(frames, 1):3.0f}%  "
                  f"reconnects {link.reconnects}  clutch {robot.get('clutch', '?')}  rtt {lat}", flush=True)
            frames, tracked, fps_t = 0, 0, now
        grip = "no hand" if openness is None else "FIST" if openness < 0.30 else "open" if openness > 0.65 else "..."
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
        rl = reach_status_line(robot)   # plane tuning: compact "REACH >> | spd4.0 lat.20 vrt.15 db.015"
        if rl:
            col = (0, 220, 0) if robot.get("plane_tune", {}).get("reach_state") in (">>", "<<") else (200, 200, 200)
            cv2.putText(frame, rl, (10, frame.shape[0] - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.putText(frame, "+/- reach  [/] sens  ,/. sooner  g check  o save  0 reset",
                        (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        tune = robot.get("tune")   # guided reach-threshold check prompt
        if tune and tune.get("active"):
            cv2.putText(frame, f"TUNE CHECK: {tune.get('prompt', '')}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 255), 2)
            if tune.get("progress") is not None:
                cv2.rectangle(frame, (10, 160), (10 + int(300 * tune["progress"]), 170), (0, 255, 255), -1)
        if robot.get("assist"):
            cv2.putText(frame, "ASSIST: grabbing / releasing...", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 80, 220), 2)
        elif robot.get("grab_ready"):
            cv2.putText(frame, f"make a FIST to grab the {robot['grab_ready']}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 80, 220), 2)
        if robot.get("ghost") == "clamped":
            cv2.putText(frame, "OUT OF REACH - the gripper is clamped at the edge", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(frame, "F clutch | S stop | R reset | M flip | Q quit",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # active preset banner + progress (clear feedback that a move is running; status carries preset/preset_progress)
        left_w = frame.shape[1] - ButtonPanel.W - panel.margin - 20
        active = robot.get("preset")
        if active:
            prog = robot.get("preset_progress")
            txt = f"MOVING: {ButtonPanel.label(active)}" + (f"   {int(100 * prog)}%" if prog is not None else "")
            cv2.rectangle(frame, (10, 100), (left_w, 138), (0, 120, 0), -1)
            cv2.rectangle(frame, (10, 100), (left_w, 138), (255, 255, 255), 1)
            cv2.putText(frame, txt, (18, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if prog is not None:
                cv2.rectangle(frame, (10, 141), (10 + int(max(0, left_w - 20) * prog), 149), (0, 220, 0), -1)
        # real arm following? None = no follower ever seen (pure sim); red = it stopped polling (not following)
        rf = robot.get("real_following")
        if rf is not None:
            yb = frame.shape[0] - 48
            if rf:
                cv2.putText(frame, "real arm: FOLLOWING", (10, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
            else:
                cv2.rectangle(frame, (6, yb - 22), (left_w, yb + 8), (0, 0, 200), -1)
                cv2.putText(frame, "REAL ARM NOT FOLLOWING - start the follower", (12, yb),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for msg in panel.take(time.time()):  # button clicks -> app (it does the safety checks)
            link.send(msg)
        panel.draw(frame, robot, time.time())
        if time.time() < toast_until and toast_text:   # transient ack / refusal, drawn on top
            col = (0, 160, 0) if toast_ok else (0, 90, 220)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), col, -1)
            cv2.putText(frame, toast_text[:70], (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("rover body client", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("f"):
            if time.time() - last_f >= 0.3:   # debounce: a key repeat / double tap must not toggle twice
                last_f = time.time()
                link.send({"kind": "clutch", "t": time.time(), "action": "toggle"})
        elif k == ord("s"):
            say("stop")
        elif k == ord("r"):
            say("reset")
        elif k == ord("m"):
            mirror = not mirror   # flip left/right for a mirrored webcam (plane / arm_mirror re-anchor on engage)
        elif k in PLANE_TUNE_KEYS:   # plane live-tuning: + - reach spd, [ ] sens, , . sooner, g check, o save, 0 reset
            link.send(plane_tune_msg(PLANE_TUNE_KEYS[k], time.time()))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
