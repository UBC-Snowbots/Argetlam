"""Rover camera sender: runs ON THE CAMERA'S DEVICE (a Pi / Jetson / the rover's onboard computer, or
any Linux/Windows box) and streams frames to the VM as the SAME `CameraFrame` message the sim bridge
and the laptop clients send, so the sim camera and the real camera share one downstream path
(detector -> table-plane projection -> scene -> viewer); sim <-> real is a config flip.

Pure Python + OpenCV + stdlib, no repo imports, so it can be scp'd to the device on its own:
    scp src/rover_intent/perception/camera_tx.py pi@rover:~/
    python3 camera_tx.py --host <VM-tailscale-ip> --port 47100 --source 0 --width 640 --height 480 --fps 5

`--source` is a local camera index (0), a device path (/dev/video0), OR an mjpeg/rtsp URL the device
already exposes (http://.../stream.mjpg, rtsp://...); the VM can also run this same script to INGEST
such a URL (cv2.VideoCapture(url)) when the device only serves a stream.  Reconnects on failure and
reopens the source if no frame arrives within the watchdog window.  Blocking, paced send - no busy loop.

Calibration for a remote device: capture the ChArUco board frames ON the device (--save-dir writes the
raw frames; or grab stills any way), copy them to the VM, and calibrate there with
scripts/perception_calibrate.py (docs/PERCEPTION.md).  Point `camera.calib` at the result.

Nothing here is run by the dev agent; these are the commands for Pranav / the rover. The VM app must
have `perception.source: camera` for the frames to be used.
"""
from __future__ import annotations

import argparse
import base64
import json
import socket
import time


def parse_source(src: str):
    """'0' -> 0 (camera index); '/dev/video1' or a URL -> the string unchanged (cv2 handles both)."""
    return int(src) if str(src).isdigit() else src


def open_capture(source, width=0, height=0, fps=0):
    import cv2
    cap = cv2.VideoCapture(parse_source(source))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def build_camera_frame(frame_bgr, seq: int, source_label: str = "rover_cam", quality: int = 70,
                       width: int = 0, height: int = 0) -> tuple[dict, int]:
    """BGR frame (as OpenCV reads it) -> (CameraFrame dict, jpeg byte size). Resizes if width/height set."""
    import cv2
    if width and height:
        frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    h, w = frame_bgr.shape[:2]
    frame = {"kind": "camera_frame", "t": time.time(), "jpeg_b64": base64.b64encode(buf).decode(),
             "width": int(w), "height": int(h), "seq": int(seq), "source": source_label}
    return frame, int(buf.nbytes)


class UdpLink:
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (host, int(port))

    def send(self, msg: dict):
        data = json.dumps(msg).encode()
        if len(data) > 65000:
            print(f"WARNING: frame {len(data)} B > UDP datagram limit; lower --quality/--width/--height")
        self.sock.sendto(data, self.addr)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="VM address (Tailscale IP / hostname)")
    ap.add_argument("--port", type=int, default=47100, help="VM app listen_port (CameraFrame receiver)")
    ap.add_argument("--source", default="0", help="camera index (0), /dev/videoN, or an mjpeg/rtsp URL")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=5.0, help="frames per second to send (2-10)")
    ap.add_argument("--quality", type=int, default=70)
    ap.add_argument("--label", default="rover_cam", help="CameraFrame source label (shown in the viewer)")
    ap.add_argument("--reconnect-s", type=float, default=2.0)
    ap.add_argument("--watchdog-s", type=float, default=5.0, help="reopen the source if no frame for this long")
    args = ap.parse_args(argv)

    link = UdpLink(args.host, args.port)
    print(f"camera_tx: source={args.source} -> udp {args.host}:{args.port} @ {args.fps} Hz ({args.label})",
          flush=True)
    period = 1.0 / max(args.fps, 0.1)
    cap = None
    seq, last_log, last_ok = 0, 0.0, time.time()
    try:
        while True:
            t0 = time.time()
            if cap is None or not cap.isOpened():
                cap = open_capture(args.source, args.width, args.height, args.fps)
                if not cap.isOpened():
                    print(f"could not open {args.source}; retry in {args.reconnect_s}s", flush=True)
                    time.sleep(args.reconnect_s)
                    continue
                last_ok = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                if time.time() - last_ok > args.watchdog_s:
                    print("no frames past the watchdog; reopening the source", flush=True)
                    cap.release()
                    cap = None
                else:
                    time.sleep(min(args.reconnect_s, period))
                continue
            last_ok = time.time()
            msg, nbytes = build_camera_frame(frame, seq, args.label, args.quality, args.width, args.height)
            link.send(msg)
            seq += 1
            if t0 - last_log > 2.0:
                last_log = t0
                print(f"seq {seq}  {msg['width']}x{msg['height']}  jpeg {nbytes} B", flush=True)
            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    main()
