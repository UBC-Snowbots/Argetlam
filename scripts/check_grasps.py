"""Automated grasp check against the running app + Isaac bridge (sim ground truth).

    python scripts/check_grasps.py [--host 127.0.0.1] [--moves]

For every object: reset -> "pick up the X" (offline keyword parser: deterministic, no Nemotron) -> wait for the plan
to finish -> PASS if X is held and lifted >= 5 cm off the table and no other object moved > 2 cm or fell over.
--moves also runs "move the X beside the Y" for every pair: PASS if X ends upright on the table within 3 cm of the
drop-off point.
Exit code 0 only if everything passes.

(The body-teleop grab assist check was removed with the legacy position mode: the two surviving teleop modes, plane
and arm_mirror, use a plain open/close gripper with no AUTO grab-assist, so there is nothing for it to drive.)
"""
import argparse
import itertools
import json
import socket
import sys
import time

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
    args = ap.parse_args()
    app = App(args.host, args.port)
    results = []
    names = list(settle(app)["objects"])
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
