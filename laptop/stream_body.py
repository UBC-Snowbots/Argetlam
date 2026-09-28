"""LAPTOP: webcam -> MediaPipe Pose (world landmarks) -> BodyPose over UDP to the VM.

    pip install -e ".[laptop]"
    python laptop/stream_body.py --host <vm-tailnet-ip> [--camera 0] [--side right] [--show]

Uses the right arm by default (MediaPipe landmarks 12 shoulder, 16 wrist; left = 11, 15).
Hand openness comes from MediaPipe Hands (fingertip-to-wrist spread), for the video's "hand stays open" shot.
"""
import argparse
import time

import cv2
import mediapipe as mp

from rover_intent.transport.udp import Sender
from rover_intent.types import BodyPose

SIDES = {"right": (12, 16), "left": (11, 15)}


def hand_openness(hand) -> float:
    lm = hand.landmark
    w = lm[0]
    tips = [lm[i] for i in (8, 12, 16, 20)]
    spread = sum(((t.x - w.x) ** 2 + (t.y - w.y) ** 2) ** 0.5 for t in tips) / 4
    palm = ((lm[9].x - w.x) ** 2 + (lm[9].y - w.y) ** 2) ** 0.5 + 1e-6
    return max(0.0, min(1.0, (spread / palm - 1.0) / 0.9))  # ~1.0 closed fist .. ~1.9 open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=47100)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--side", choices=SIDES, default="right")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    tx = Sender(args.host, args.port)
    sh_i, wr_i = SIDES[args.side]
    cap = cv2.VideoCapture(args.camera)
    with mp.solutions.pose.Pose(model_complexity=1) as pose, mp.solutions.hands.Hands(max_num_hands=1) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r = pose.process(rgb)
            h = hands.process(rgb)
            if r.pose_world_landmarks:
                w = r.pose_world_landmarks.landmark
                vis = min(w[sh_i].visibility, w[wr_i].visibility) > 0.5
                openness = hand_openness(h.multi_hand_landmarks[0]) if h.multi_hand_landmarks else 1.0
                tx.send(BodyPose(t=time.time(), wrist=(w[wr_i].x, w[wr_i].y, w[wr_i].z),
                                 shoulder=(w[sh_i].x, w[sh_i].y, w[sh_i].z), hand_open=openness, visible=vis))
            if args.show:
                if r.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, r.pose_landmarks,
                                                              mp.solutions.pose.POSE_CONNECTIONS)
                cv2.imshow("body", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    cap.release()


if __name__ == "__main__":
    main()
