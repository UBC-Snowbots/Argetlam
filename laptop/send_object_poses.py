"""Send object_poses to the rover app: the wire format for a remote perception device (the laptop's RealSense D415).

Stdlib only, so it runs on the laptop as-is. Use it two ways:
  * test the link before detection exists (fixed positions, 3 Hz until Ctrl+C):
        python laptop/send_object_poses.py --host rover-vm --obj cup 0.45 0.20 0.0 --obj bottle 0.45 -0.20 0.0
  * import it from the RealSense detector and call send() a few times a second:
        from send_object_poses import ObjectPoseSender
        tx = ObjectPoseSender("rover-vm")
        tx.send({"cup": {"pos": [x, y, z_bottom], "conf": 0.9, "err_cm": 1.0, "desc": "red cup"}})

Positions are each object's BOTTOM centre in the ARM BASE frame, metres (camera->base calibration happens on the
laptop). Names must match the VM's scene objects (cup, bottle, block today); unknown names are ignored there. The VM
uses them only when perception.source is "external" (and then moves the Isaac objects to match: mirror_to_sim).
Send only what the camera sees NOW: an object not sent for perception.stale_s (1 s) counts as unseen and a voice
skill refuses it ("I don't see the bottle"). docs/PERCEPTION.md "External perception".
"""
import argparse
import json
import socket
import time


class ObjectPoseSender:
    def __init__(self, host: str, port: int = 47100, source: str = "realsense"):
        self.addr, self.source, self.seq = (host, port), source, 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, objects: dict) -> None:
        """objects: {name: {"pos": [x, y, z_bottom], "conf": 0..1, "err_cm": float, "desc": str}} (pos required)."""
        self.seq += 1
        msg = {"kind": "object_poses", "t": time.time(), "seq": self.seq, "source": self.source, "objects": objects}
        self.sock.sendto(json.dumps(msg).encode(), self.addr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="rover-vm", help="VM (Tailscale) address")
    ap.add_argument("--port", type=int, default=47100)
    ap.add_argument("--hz", type=float, default=3.0)
    ap.add_argument("--obj", nargs=4, action="append", metavar=("NAME", "X", "Y", "Z_BOTTOM"), required=True)
    a = ap.parse_args()
    objs = {n: {"pos": [float(x), float(y), float(z)], "conf": 1.0} for n, x, y, z in a.obj}
    tx = ObjectPoseSender(a.host, a.port, source="test")
    print(f"sending {sorted(objs)} to {a.host}:{a.port} at {a.hz} Hz (Ctrl+C to stop)")
    try:
        while True:
            tx.send(objs)
            time.sleep(1.0 / a.hz)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
