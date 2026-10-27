"""Live viewer for the rover camera (SIM or REAL) on the VM screen: "see what it sees".

Shows the camera's JPEG stream with overlays so the perception error is visible at a glance:
  * a banner with the source: "SIM CAMERA (rover_front)" or "REAL CAMERA";
  * GROUND-TRUTH 2D boxes carried in the frame packet (green) - the sim bridge fills these from the
    instance segmentation; absent on a real camera;
  * the DETECTED object positions (magenta): the app's current scene positions (from the Query status)
    projected back through the camera model, with the label and the estimated 3D (x, y, z) and the
    source (ground_truth / aruco / owlv2); stale objects are dimmed.

It reuses the HUD video path: the app forwards every CameraFrame to net.hud_video_port, so point this
viewer at that port (or run it instead of scripts/hud.py). The SAME viewer serves the real camera when
it arrives (its frames ride the identical CameraFrame path) - switching sim <-> real is a config flip.

    PYTHONPATH=src python scripts/sim_cam_viewer.py --arm dev_arm --video-port 47231 --app 127.0.0.1:47200

Needs tkinter (stdlib) + Pillow. Isaac is NOT needed (it consumes frames, it does not render them).
The overlay maths (annotate_frame) is pure and unit-tested; the tkinter loop is a thin, paced wrapper.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rover_intent.config import load_config  # noqa: E402
from rover_intent.perception.camera import camera_from_config  # noqa: E402

_GREEN = (60, 220, 90)
_MAGENTA = (235, 60, 200)
_DIM = (120, 120, 120)


def source_banner(source: str | None) -> str:
    s = (source or "").lower()
    if s.startswith("sim"):
        where = source.split(":", 1)[1] if ":" in source else ""
        return f"SIM CAMERA ({where})" if where else "SIM CAMERA"
    if "real" in s or s in ("rover_cam", "pi_cam", "jetson"):
        return f"REAL CAMERA ({source})"
    return f"CAMERA ({source})" if source else "CAMERA"


def annotate_frame(rgb: np.ndarray, camera, boxes: dict | None, detections: list[dict] | None,
                   source: str | None = None):
    """Draw the overlays on an RGB frame and return a PIL.Image.

    `boxes` = {name: [u0,v0,u1,v1]} ground-truth 2D boxes (green). `detections` = [{name, xyz,
    src?, seen?}] detected object positions (projected to pixels, magenta; dimmed if seen is False).
    """
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.asarray(rgb, np.uint8)[:, :, :3].copy())
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width - 1, 14], fill=(0, 0, 0))
    d.text((3, 3), source_banner(source), fill=(230, 230, 230))
    for name, b in (boxes or {}).items():
        d.rectangle([b[0], b[1], b[2], b[3]], outline=_GREEN)
        d.text((b[0] + 2, b[1] + 2), f"gt:{name}", fill=_GREEN)
    for det in (detections or []):
        xyz = np.asarray(det["xyz"], float)
        uv = camera.project(xyz)[0]
        if not (0 <= uv[0] < im.width and 0 <= uv[1] < im.height):
            continue
        col = _MAGENTA if det.get("seen", True) else _DIM
        r = 4
        d.ellipse([uv[0] - r, uv[1] - r, uv[0] + r, uv[1] + r], outline=col, width=2)
        label = f"{det['name']} ({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f}) {det.get('src', '')}".strip()
        d.text((uv[0] + 6, uv[1] - 4), label, fill=col)
    return im


class _AppQuery:
    """Tiny client that asks the app for the current scene (object positions + provenance)."""

    def __init__(self, host, port):
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.2)

    def poll(self):
        try:
            self.sock.sendto(json.dumps({"kind": "query", "t": time.time()}).encode(), self.addr)
            for _ in range(5):
                m = json.loads(self.sock.recvfrom(65536)[0])
                if m.get("kind") == "state":
                    return m
        except OSError:
            return None
        return None


def _detections_from_state(state: dict) -> list[dict]:
    out = []
    for name, o in (state or {}).get("objects", {}).items():
        pos = o["pos"]
        bz = o.get("bottom_z", pos[2])
        out.append({"name": name, "xyz": [pos[0], pos[1], bz], "src": o.get("src", ""),
                    "seen": o.get("seen", True)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="dev_arm")
    ap.add_argument("--video-port", type=int, default=47231, help="UDP port the app forwards CameraFrames to")
    ap.add_argument("--app", default="127.0.0.1:47200", help="host:port of the app (for the Query status)")
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    import tkinter as tk
    from PIL import Image, ImageTk

    cfg = load_config(arm=args.arm)
    camera, _ = camera_from_config(cfg["camera"], z_table_default=cfg.get("table", {}).get("top", 0.0))
    ah, ap_ = args.app.rsplit(":", 1)
    query = _AppQuery(ah, int(ap_))

    vsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    vsock.bind(("127.0.0.1", args.video_port))
    vsock.setblocking(False)
    latest = {"frame": None}

    root = tk.Tk()
    root.title("rover camera viewer")
    W = int(camera.width * args.scale) or 640
    H = int(camera.height * args.scale) or 480
    canvas = tk.Canvas(root, width=W, height=H, bg="#101418", highlightthickness=0)
    canvas.pack()
    imgref = {"tk": None}

    def drain():
        newest = None
        while True:
            try:
                newest = json.loads(vsock.recv(200000))
            except (BlockingIOError, OSError):
                break
        if newest and newest.get("kind") == "camera_frame":
            latest["frame"] = newest

    def tick():
        drain()
        fr = latest["frame"]
        if fr is not None:
            rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(fr["jpeg_b64"]))).convert("RGB"))
            dets = _detections_from_state(query.poll())
            im = annotate_frame(rgb, camera, fr.get("boxes"), dets, fr.get("source"))
            if args.scale != 1.0:
                im = im.resize((W, H))
            imgref["tk"] = ImageTk.PhotoImage(im)
            canvas.create_image(0, 0, anchor="nw", image=imgref["tk"])
        root.after(100, tick)   # 10 Hz paced redraw (timer callback, not a busy loop)

    root.after(100, tick)
    root.mainloop()


if __name__ == "__main__":
    main()
