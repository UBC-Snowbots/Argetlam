"""Table camera client (runs on the operator laptop, e.g. "bench-laptop" at the bench).

Captures frames from a plain USB/RGB webcam and sends them to the VM as `CameraFrame` messages
(base64 JPEG inside the same JSON transport every other laptop client uses: UDP on the tailnet, or
newline-JSON over an SSH tunnel to scripts/tcp_relay.py with --tunnel).  Detection + table-plane
projection run on the VM GPU; this script only captures and sends.

The camera is FIXED, looking at the table; mount and calibration are in docs/TABLE_CAM.md.

Run (direct UDP over the tailnet):
    python table_cam.py --host rover-vm --hz 2
Run (SSH tunnel, when UDP can't reach the VM):
    ssh -N -L 47120:127.0.0.1:47120 rover-vm         # leave running
    python table_cam.py --tunnel 127.0.0.1:47120 --hz 2
Test one frame from a file (prints size, sends nothing):
    python table_cam.py --image bench.jpg --dry-run

Nothing here is run by the dev agent; these are the commands for Pranav. The VM app must have
`perception.source: camera` (configs: set perception.source) for the frames to be used.
"""
import argparse
import base64
import json
import socket
import time

import cv2

DEFAULT_HOST = "rover-vm"   # rover-vm on the tailnet (same as body_client.py)
PORT = 47100


class Link:
    """UDP straight to the VM, or (--tunnel host:port) newline-JSON over TCP through the relay.
    Same message either way (copied from laptop/body_client.py's transport so this stays standalone)."""

    def __init__(self, host, port, tunnel=None):
        self.tunnel = tunnel
        self.reconnects = 0
        if tunnel:
            th, tp = tunnel.rsplit(":", 1)
            self._th, self._tp = th, int(tp)
            self.sock = socket.create_connection((self._th, self._tp), timeout=5)
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.addr = (host, port)

    def _reconnect(self):
        try:
            self.sock.close()
        except OSError:
            pass
        try:
            self.sock = socket.create_connection((self._th, self._tp), timeout=5)
            self.reconnects += 1
            return True
        except OSError:
            return False

    def send(self, msg: dict):
        data = json.dumps(msg).encode()
        if self.tunnel:
            try:
                self.sock.sendall(data + b"\n")
            except OSError:
                self._reconnect()
        else:
            if len(data) > 65000:
                # a UDP datagram this large will IP-fragment and likely drop; use --tunnel instead
                print(f"WARNING: frame {len(data)} B > UDP limit; lower --quality/--width or use --tunnel")
            self.sock.sendto(data, self.addr)


def encode_frame(frame, width, height, quality):
    if width and height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    h, w = frame.shape[:2]
    return base64.b64encode(buf).decode(), w, h, len(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--tunnel", help="host:port of an SSH tunnel to scripts/tcp_relay.py (TCP; no UDP size limit)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--hz", type=float, default=2.0, help="frames per second to send (table is static; 1-2 is plenty)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--quality", type=int, default=70, help="JPEG quality (lower -> smaller packets)")
    ap.add_argument("--image", help="send/encode one image file instead of the live camera")
    ap.add_argument("--dry-run", action="store_true", help="encode and print size, do not send")
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"could not read {args.image}")
        b64, w, h, nbytes = encode_frame(frame, args.width, args.height, args.quality)
        print(f"encoded {w}x{h} jpeg {nbytes} B (base64 {len(b64)} B)")
        if not args.dry_run:
            link = Link(args.host, args.port, args.tunnel)
            link.send({"kind": "camera_frame", "t": time.time(), "jpeg_b64": b64,
                       "width": w, "height": h, "seq": 0, "source": "table_cam"})
            print("sent one frame")
        return

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")
    link = Link(args.host, args.port, args.tunnel)
    print("link:", f"tunnel {args.tunnel}" if args.tunnel else f"udp {args.host}:{args.port}")
    period = 1.0 / max(args.hz, 0.1)
    seq, last_log = 0, 0.0
    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                print("camera read failed"); time.sleep(0.5); continue
            b64, w, h, nbytes = encode_frame(frame, args.width, args.height, args.quality)
            link.send({"kind": "camera_frame", "t": time.time(), "jpeg_b64": b64,
                       "width": w, "height": h, "seq": seq, "source": "table_cam"})
            seq += 1
            if t0 - last_log > 2.0:
                last_log = t0
                print(f"seq {seq}  {w}x{h}  jpeg {nbytes} B  reconnects {link.reconnects}", flush=True)
            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main()
