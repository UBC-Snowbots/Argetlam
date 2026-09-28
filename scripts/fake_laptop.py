"""Dry-run input: stands in for the laptop so the VM loop can be tested with no camera/mic/EEG.

    python scripts/fake_laptop.py [--host 127.0.0.1] --say "follow me" --circle 10 --eeg-grasp-at 5
    python scripts/fake_laptop.py --say "move the cup beside the bottle"
"""
import argparse
import math
import time

from rover_intent.transport.udp import Sender
from rover_intent.types import BodyPose, EEGState, Utterance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47100)
    ap.add_argument("--say", action="append", default=[])
    ap.add_argument("--circle", type=float, default=0.0, help="seconds of a fake wrist circle")
    ap.add_argument("--eeg-grasp-at", type=float, default=None, help="send p_grasp=0.9 after N s of circling")
    args = ap.parse_args()
    tx = Sender(args.host, args.port)
    for s in args.say:
        tx.send(Utterance(t=time.time(), text=s))
        time.sleep(0.2)
    t0 = time.time()
    while time.time() - t0 < args.circle:
        t = time.time() - t0
        wrist = (0.1 * math.cos(t), 0.0, -0.35 + 0.1 * math.sin(t))
        tx.send(BodyPose(t=time.time(), wrist=wrist, shoulder=(0.0, 0.0, 0.0)))
        if args.eeg_grasp_at is not None:
            tx.send(EEGState(t=time.time(), p_grasp=0.9 if t > args.eeg_grasp_at else 0.1))
        time.sleep(1 / 30)


if __name__ == "__main__":
    main()
