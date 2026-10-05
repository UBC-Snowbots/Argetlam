"""Automated grasp check against the running app + Isaac bridge (sim ground truth).

    python scripts/check_grasps.py [--host 127.0.0.1] [--moves]

For every object: reset -> "pick up the X" (offline keyword parser: deterministic, no Nemotron) -> wait for the plan
to finish -> PASS if X is held and lifted >= 5 cm off the table and no other object moved > 2 cm or fell over.
--moves also runs "move the X beside the Y" for every pair: PASS if X ends upright on the table within 3 cm of the
drop-off point.
--assist runs the body-teleop GRAB ASSIST with a scripted operator (fake body poses through the real app, position
mode: needs no calibration profile loaded): clutch in, steer roughly above X (3 cm / 2 cm off, 7 cm high), hold a fist
-> PASS if X is held, lifted >= 4 cm, the arm is back in FOLLOW, nothing else moved; then steer down to just above the
table, open the hand -> PASS if X is released upright within 2 cm of where it started.
Exit code 0 only if everything passes.
"""
import argparse
import itertools
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

BESIDE = np.array([0.0, 0.12, 0.0])  # planner/skills.py _OFFSETS["beside"]


class App:
    def __init__(self, host, port):
        self.addr = (host, port)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.settimeout(5)

    def _ask(self, msg, kind):
        self.s.sendto(json.dumps(msg).encode(), self.addr)
        end = time.time() + 5
        while time.time() < end:
            r = json.loads(self.s.recvfrom(65536)[0])
            if r.get("kind") == kind:
                return r
        raise TimeoutError(kind)

    def say(self, text):
        return self._ask({"kind": "utterance", "t": time.time(), "text": text, "offline": True}, "reply")

    def state(self):
        return self._ask({"kind": "query", "t": time.time()}, "state")

    def run(self, text, timeout=40):
        r = self.say(text)
        t0 = time.time()
        time.sleep(1.0)
        while time.time() - t0 < timeout:
            st = self.state()
            if st["plan_left"] == 0 and st["mode"] != "auto":
                return r, st
            time.sleep(0.5)
        return r, self.state()


class Operator:
    """Scripted body client: wrist/shoulder poses mapped through the app's position-mode Retargeter
    (robot delta = scale * axes @ (rel - neutral))."""

    def __init__(self, app, axes, scale):
        self.app, self.A, self.g = app, np.asarray(axes, float), np.asarray(scale, float)
        self.seq, self.neutral, self.rel = 0, np.array([0.05, 0.10, -0.30]), None

    def send(self, rel, hand_open):
        self.seq += 1
        msg = {"kind": "body", "t": time.time(), "seq": self.seq, "wrist": list(map(float, rel)), "shoulder": [0.0, 0.0, 0.0],
               "visible": True, "in_position": True, "side": "right", "hand_open": hand_open}
        self.app.s.sendto(json.dumps(msg).encode(), self.app.addr)

    def stream(self, seconds, hand_open=0.9, to=None, until=None):
        """Hold (or glide linearly to robot-frame offset `to` from the engage point) at 30 Hz; stop early on until(st)."""
        start, t0 = self.rel.copy(), time.time()
        goal = start if to is None else self.neutral + self.A.T @ (np.asarray(to, float) / self.g)
        while time.time() - t0 < seconds:
            f = min(1.0, (time.time() - t0) / max(seconds * 0.6, 1e-3))
            self.rel = start + f * (goal - start)
            self.send(self.rel, hand_open)
            time.sleep(1 / 30)
            if until is not None and self.seq % 15 == 0 and until(self.app.state()):
                return True
        return until is None

    def engage(self):
        self.rel = self.neutral.copy()
        self.app.s.sendto(json.dumps({"kind": "clutch", "t": time.time(), "action": "toggle"}).encode(), self.app.addr)
        return self.stream(6.0, until=lambda st: st.get("clutch") == "following")


def run_assist(app, name, axes, scale):
    before = settle(app)
    op = Operator(app, axes, scale)
    if not op.engage():
        return False, "clutch never engaged"
    tcp0 = np.array(app.state()["tcp_meas"])
    o = np.array(before["objects"][name]["pos"])
    above = o + [0.03, 0.02, 0.07]                       # rough, like a person would steer
    op.stream(4.0, to=above - tcp0)
    got = op.stream(30.0, hand_open=0.1, until=lambda st: st["held"] == name and not st["assist"])
    st = app.state()
    lift = st["objects"][name]["bottom_z"] - before["table_top"]
    probs = moved(before, st, {name})
    ok1 = got and st["mode"] == "follow" and lift >= 0.04 and not probs
    msg = f"grab: held={st['held']} mode={st['mode']} lifted {lift * 100:4.1f} cm {'; '.join(probs)}"
    if not ok1:
        return False, msg
    tcp1 = np.array(st["tcp_meas"])
    op.neutral = op.rel.copy()      # after the hand-back the app re-anchors: this pose now maps to the arm's position
    op.stream(4.0, hand_open=0.1, to=o + [0, 0, 0.005] - tcp1)   # steer back down to just above the table
    rel_ok = op.stream(20.0, hand_open=0.9, until=lambda st: st["held"] is None and not st["assist"])
    op.stream(1.5)
    st = app.state()
    d = np.linalg.norm(np.array(st["objects"][name]["pos"][:2]) - o[:2])
    upright = abs(st["objects"][name]["bottom_z"] - before["table_top"]) < 0.01
    ok2 = rel_ok and upright and d < 0.02 and st["mode"] == "follow"
    app.s.sendto(json.dumps({"kind": "clutch", "t": time.time(), "action": "toggle"}).encode(), app.addr)
    return ok1 and ok2, msg + f" | release: upright={upright} moved {d * 100:.1f} cm mode={st['mode']}"


def settle(app):
    app.say("reset")
    time.sleep(3.0)  # let the objects settle on the table
    return app.state()


def moved(before, after, skip):
    bad = []
    for n, o in before["objects"].items():
        if n in skip:
            continue
        a = after["objects"][n]
        d = np.linalg.norm(np.array(a["pos"][:2]) - np.array(o["pos"][:2]))
        if d > 0.02 or abs(a["bottom_z"] - before["table_top"]) > 0.01:
            bad.append(f"{n} disturbed (moved {d * 100:.1f} cm, bottom z {a['bottom_z']:.3f})")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47100)
    ap.add_argument("--moves", action="store_true")
    ap.add_argument("--assist", action="store_true", help="also run the body-teleop grab assist with a scripted operator")
    ap.add_argument("--only-assist", action="store_true")
    args = ap.parse_args()
    app = App(args.host, args.port)
    results = []
    names = list(settle(app)["objects"])
    if args.assist or args.only_assist:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from rover_intent.config import load_config
        b = load_config()["body"]
        if app.state().get("mapping") != "Retargeter":
            print("SKIP assist: the app uses a calibration profile (needs position mode)")
        else:
            for n in names:
                ok, msg = run_assist(app, n, b["axes"], b["scale"])
                results.append(ok)
                print(f"{'PASS' if ok else 'FAIL'}  assist {n:6}  {msg}", flush=True)
    if args.only_assist:
        settle(app)
        print(f"\n{sum(results)}/{len(results)} passed")
        sys.exit(0 if all(results) else 1)
    for n in names:
        before = settle(app)
        reply, after = app.run(f"pick up the {n}")
        lift = after["objects"][n]["bottom_z"] - before["table_top"]
        problems = moved(before, after, {n})
        ok = after["held"] == n and lift >= 0.05 and not problems
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  pick {n:6}  held={after['held']}  lifted {lift * 100:5.1f} cm  "
              f"{'; '.join(problems)}  [{reply.get('say', '')}]", flush=True)
    if args.moves:
        for a, b in itertools.permutations(names, 2):
            before = settle(app)
            goal = np.array(before["objects"][b]["pos"]) + BESIDE
            reply, after = app.run(f"move the {a} beside the {b}", timeout=60)
            time.sleep(1.5)
            after = app.state()
            pa = after["objects"][a]
            err = np.linalg.norm(np.array(pa["pos"][:2]) - goal[:2])
            upright = abs(pa["bottom_z"] - before["table_top"]) < 0.01
            problems = moved(before, after, {a})
            ok = err < 0.03 and upright and after["held"] is None and not problems
            results.append(ok)
            print(f"{'PASS' if ok else 'FAIL'}  move {a:6} beside {b:6}  error {err * 100:4.1f} cm  upright={upright}  "
                  f"{'; '.join(problems)}", flush=True)
    settle(app)
    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
