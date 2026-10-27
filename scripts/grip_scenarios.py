"""Seeded gripper scenario generator for the dual-source gripper (docs/GRIPPER.md).

Synthesizes WILD body teleop (fast right-hand sweeps, reversals, left reach up/down, brief tracking loss) at the
client's real rate, engages the clutch through the normal palm-up path, then injects gripper commands from:
  (1) Muse  - GripToggle only (hand not found, so no hand edges),
  (2) hand  - fist/open edges only (no Muse),
  (3) both  - a hand edge and a Muse toggle with a timing offset {0, 0.1, 0.3, 0.6, 1.0 s}, same-intent and
              opposite-intent, to exercise the coalescing CLASH policy.
Each command has an EXPECTED gripper state under the policy. The app is driven over UDP; per command the harness
checks the app gripper reached the expectation within 0.5 s (and, with --backend isaac, that the SIM FINGERS
reached it within ~2 s via the bridge-reported gripper); it also checks the arm TCP moved (teleop sanity). It runs
N seeds per family and prints a table.

Mock validation (no GPU):
  .venv/bin/python scripts/grip_scenarios.py --backend mock --seeds 5
The live Isaac run (the coordinator runs this in a GPU window on the live stack; do NOT run it here):
  .venv/bin/python scripts/grip_scenarios.py --backend isaac --seeds 20 --no-spawn \
      --host 127.0.0.1 --port 47100
NO GPU and NO Isaac is started by this script; --no-spawn drives an already-running app (e.g. the live Isaac stack).
"""
from __future__ import annotations

import argparse
import socket
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rover_intent import griptest as G   # noqa: E402
from rover_intent.control.arbiter import Arbiter, HAND_HOLD_S as G_HAND_HOLD   # noqa: E402
from rover_intent.planner.scene import Scene   # noqa: E402
from rover_intent.types import Action, Intent   # noqa: E402

COALESCE_S = 0.6
SETTLE_S = 1.3           # time allowed for a command to settle (covers the 0.6 s hand debounce + margin)
MATCH_WINDOW_S = 0.5     # the app gripper must match within this of the trigger (brief)
FINGER_WINDOW_S = 2.0    # Isaac: the sim fingers must reach the state within this
OFFSETS = [0.0, 0.1, 0.3, 0.6, 1.0]


class Player:
    """Streams body frames at the client rate while applying seeded wild motion, and sends gripper commands at
    recorded wall-clock trigger times. Collects the list of expected commands for evaluation."""

    def __init__(self, send, rng, stream_hz=18.0):
        self.send, self.rng, self.dt = send, rng, 1.0 / stream_hz
        self.seq = 1
        self.hand_open = 0.9          # current hand openness streamed in each frame (None = hand not found)
        self.commands: list[dict] = []
        # a SHADOW arbiter fed the exact same gripper inputs (hand openness + Muse toggles) at the same wall-clock
        # times, as the ground-truth ORACLE for the expected gripper state + clash under the policy. The dual-source
        # policy itself is proven by the unit tests; here it lets the scenario check the full UDP + app + plane +
        # backend pipeline against it. tcp=None (plain gripper), coalesce_s matches the app default.
        self.oracle = Arbiter(Scene({"cup": {"pos": [0.45, 0.20, 0.055]}}), coalesce_s=COALESCE_S)
        self._oracle_on = False
        # wild-motion state
        self._cx, self._cy, self._ry = 0.0, 0.0, 0.0
        self._vx, self._vy = 0.0, 0.0
        self._loss_until = 0.0

    def reset_oracle(self):
        self.oracle.on_intent(Intent(action=Action.reset))
        self.oracle.on_intent(Intent(action=Action.follow))
        self._oracle_on = False

    def _wild_step(self, now):
        # fast sweeps with sudden reversals, bounded so the control wrist stays above the hip and in frame
        if self.rng.random() < 0.06:
            self._vx = self.rng.uniform(-0.9, 0.9)
            self._vy = self.rng.uniform(-0.7, 0.7)
        self._cx = float(np.clip(self._cx + self._vx * self.dt, -0.25, 0.25))
        self._cy = float(np.clip(self._cy + self._vy * self.dt, -0.28, 0.28))
        self._ry = float(np.clip(self._ry + self.rng.uniform(-0.05, 0.05), -0.25, 0.25))  # left reach up/down
        if self._cx in (-0.25, 0.25):
            self._vx = -self._vx
        if self._cy in (-0.28, 0.28):
            self._vy = -self._vy
        visible = now >= self._loss_until
        if visible and self.rng.random() < 0.01:       # occasional brief tracking loss (< watchdog 1 s)
            self._loss_until = now + self.rng.uniform(0.15, 0.4)
            visible = False
        return visible

    def _frame(self, now, hand_open):
        visible = self._wild_step(now)
        self.send(G.make_body(now, self.seq, ctrl_dx=self._cx, ctrl_dy=self._cy, reach_dy=self._ry,
                              hand_open=hand_open, visible=visible))
        self.seq += 1
        if self._oracle_on and visible and hand_open is not None:
            self.oracle.on_hand(hand_open, now=now, tcp=None)   # feed the shadow arbiter the same hand input

    def stream(self, dur, hand_open="keep", on_tick=None):
        """Stream frames for `dur` seconds. hand_open 'keep' leaves the current value; a float/None sets it. on_tick
        (now, elapsed) may fire one-shot injections (returns True once fired to stop being called)."""
        if hand_open != "keep":
            self.hand_open = hand_open
        t0 = time.time()
        fired = False
        while True:
            now = time.time()
            el = now - t0
            if el >= dur:
                break
            if on_tick is not None and not fired and on_tick(now, el):
                fired = True
            self._frame(now, self.hand_open)
            time.sleep(max(0.0, self.dt - (time.time() - now)))

    def muse(self):
        t = time.time()
        self.send({"kind": "grip_toggle", "t": t, "source": "clench"})
        if self._oracle_on:
            self.oracle.on_grip_toggle(None, now=t)
        return t

    def record(self, family, kind, trigger_t, clash0, **extra):
        """Snapshot the ORACLE's settled gripper + clash as the expectation for one command."""
        c = {"family": family, "kind": kind, "trigger_t": trigger_t,
             "expected": int(self.oracle.gripper > 0.5),
             "expect_clash": self.oracle.grip_clashes > clash0, "source": kind}
        c.update(extra)
        self.commands.append(c)

    def engage(self):
        rows, _, seq = G.engage_frames(time.time() + 0.05)
        for st, msg in rows:
            d = st - time.time()
            if d > 0:
                time.sleep(d)
            self.send(msg)
        self.seq = seq
        self.oracle.on_intent(Intent(action=Action.follow))   # the app is now following: shadow it from here
        self._oracle_on = True
        self.stream(0.6, hand_open=0.9)   # settle in FOLLOW


# --- families -----------------------------------------------------------------------------------------------------
def run_muse(p: Player, n=5):
    """Muse toggles only; hand not found so no hand edges. Each toggle flips the gripper."""
    p.stream(1.2, hand_open=None)
    for i in range(n):
        c0 = p.oracle.grip_clashes
        t = p.muse()
        p.stream(1.5, hand_open=None)
        p.record("muse", "muse", t, c0)


def run_hand(p: Player, n=6):
    """Hand edges only. Alternate fist (CLOSE) and open (OPEN) postures, each held past the debounce."""
    p.stream(1.0, hand_open=0.95)    # establish OPEN baseline
    for i in range(n):
        want_fist = (i % 2 == 0)     # first command is a fist -> CLOSE
        c0 = p.oracle.grip_clashes
        t = time.time()
        p.stream(1.3, hand_open=0.03 if want_fist else 0.97)   # edge fires ~HAND_HOLD_S after this starts
        p.record("hand", "hand", t, c0)
        p.stream(0.4, hand_open="keep")


def run_both(p: Player):
    """A hand edge and a Muse toggle near each other, same-intent and opposite-intent, over the offset set. A Muse
    toggle lands `offset` before the hand fist CLOSE edge confirms. same: gripper is OPEN when the Muse fires, so it
    also closes (one CLOSE, no clash). opposite: gripper is CLOSED (a Muse closed it earlier) so the trial Muse
    OPENS while the hand edge CLOSES - a clash inside the window, the later (hand) event wins. The oracle gives the
    exact expected state + clash for each."""
    for offset in OFFSETS:
        _both_trial(p, offset, opposite=False)
        _both_trial(p, offset, opposite=True)


def _both_trial(p: Player, offset, opposite):
    p.stream(0.9, hand_open=0.97)      # establish an OPEN posture (open edge -> gripper 0, hand state "open")
    p.stream(0.3, hand_open="keep")
    if opposite:
        p.muse()                        # Muse CLOSE -> gripper 1, well before the trial edge (not coalesced)
        p.stream(1.0, hand_open="keep")
    c0 = p.oracle.grip_clashes
    trigger = time.time()
    fire_at = max(0.0, G_HAND_HOLD - offset)   # fire the Muse `offset` before the hand edge confirms

    def on_tick(now, el):
        if el >= fire_at:
            p.muse()
            return True
        return False

    p.stream(G_HAND_HOLD + 0.4, hand_open=0.03, on_tick=on_tick)   # open->fist CLOSE edge + the Muse at offset
    # at offset 0 (truly simultaneous) and at the window boundary (== coalesce_s) the event ORDER, hence the winner,
    # is inherently undefined; mark those opposite trials "boundary" so the check requires a clash + a valid state
    # but does not insist on one specific winner
    boundary = opposite and offset in (0.0, COALESCE_S)
    p.record(f"both-{'opp' if opposite else 'same'}", "both", trigger, c0, offset=offset, boundary=boundary)
    p.stream(0.5, hand_open="keep")


# --- evaluation ---------------------------------------------------------------------------------------------------
def gripper_at_log(transitions, t):
    g = 0
    for tr in transitions:
        if tr["t"] is not None and tr["t"] <= t:
            g = 1 if tr["dir"] == "close" else 0
    return g


def gripper_at_mon(samples, t):
    """Observed gripper at time t from the polled Query series (used when the app log is not readable, e.g. the
    live Isaac stack driven with --no-spawn)."""
    g = None
    for st in samples:
        if st["_t"] <= t and st.get("gripper") is not None:
            g = 1 if st["gripper"] >= 0.5 else 0
    return g


def clashes_at_mon(samples, t):
    c = 0
    for st in samples:
        if st["_t"] <= t:
            c = int(st.get("grip_clashes") or 0)
    return c


def evaluate(commands, parsed, monitor, backend):
    """Per command: did the app gripper settle to the expected state within SETTLE_S, and (Isaac) did the SIM
    FINGERS reach it within FINGER_WINDOW. Gripper state comes from the app log transitions when available (exact),
    otherwise from the polled Query series (the only source on a --no-spawn live run). Returns per-family
    aggregates."""
    trans = [t for t in parsed["transitions"] if t["t"] is not None]
    have_log = bool(trans) or bool(parsed["toggles"])
    fam: dict[str, dict] = {}
    for c in commands:
        f = fam.setdefault(c["family"], {"commands": 0, "matched": 0, "missed": 0, "wrong": 0, "clashes": 0,
                                         "app_lat": [], "finger_lat": []})
        f["commands"] += 1
        t0 = c["trigger_t"]
        settle_t = t0 + SETTLE_S
        if have_log:
            observed = gripper_at_log(trans, settle_t)
            changes = [tr for tr in trans if t0 - 0.05 <= tr["t"] <= settle_t]
            n_clash = sum(1 for cl in parsed["clashes"] if cl["t"] is not None and t0 - 0.05 <= cl["t"] <= settle_t)
        else:
            observed = gripper_at_mon(monitor.samples, settle_t)
            changes = []
            n_clash = max(0, clashes_at_mon(monitor.samples, settle_t) - clashes_at_mon(monitor.samples, t0 - 0.05))
        if observed is None:
            f["missed"] += 1
        elif c.get("boundary"):
            f["matched"] += 1   # order (winner) is undefined here; a valid settled state is enough
        elif observed == c["expected"]:
            f["matched"] += 1
            good = [tr for tr in changes if (1 if tr["dir"] == "close" else 0) == c["expected"]]
            if good:
                f["app_lat"].append((good[-1]["t"] - t0) * 1000.0)
            elif not have_log:   # latency from the polled series (coarse): first sample at the expected state
                for st in monitor.samples:
                    if st["_t"] >= t0 and st.get("gripper") is not None and (st["gripper"] >= 0.5) == bool(c["expected"]):
                        f["app_lat"].append((st["_t"] - t0) * 1000.0)
                        break
        elif observed == 1 - c["expected"]:
            f["wrong"] += 1
        else:
            f["missed"] += 1
        f["clashes"] += n_clash
        if backend == "isaac":
            fl = _finger_latency(monitor, t0, c["expected"])
            if fl is not None:
                f["finger_lat"].append(fl * 1000.0)
    return fam


def _finger_latency(monitor, t0, expected):
    for st in monitor.samples:
        gm = st.get("gripper_meas")
        if gm is None or st["_t"] < t0:
            continue
        if (gm >= 0.5) == bool(expected) and st["_t"] - t0 <= FINGER_WINDOW_S:
            return st["_t"] - t0
    return None


def _stat(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    if q == 0.5:
        return statistics.median(xs)
    k = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
    return xs[k]


def print_table(agg, backend, followed, tcp_path, errors):
    print()
    print(f"backend={backend}  clutch_followed={followed}  tcp_path_following={tcp_path:.3f} m  app_errors={errors}")
    hdr = ["family", "seeds", "cmds", "matched", "missed", "wrong", "clashes", "app_ms(med/p90)", "finger_ms(med/p90)"]
    print("  ".join(h.ljust(w) for h, w in zip(hdr, [11, 5, 5, 7, 6, 5, 7, 16, 18])))
    for family in sorted(agg):
        a = agg[family]
        app_ms = f"{_fmt(_stat(a['app_lat'], 0.5))}/{_fmt(_stat(a['app_lat'], 0.9))}"
        fing = "n/a" if backend != "isaac" else f"{_fmt(_stat(a['finger_lat'], 0.5))}/{_fmt(_stat(a['finger_lat'], 0.9))}"
        row = [family, str(a["seeds"]), str(a["commands"]), str(a["matched"]), str(a["missed"]), str(a["wrong"]),
               str(a["clashes"]), app_ms, fing]
        print("  ".join(c.ljust(w) for c, w in zip(row, [11, 5, 5, 7, 6, 5, 7, 16, 18])))


def _fmt(v):
    return "-" if v is None else f"{v:.0f}"


def build_commands(seed, families):
    """Return the family runners to execute for a seed (the Player is driven live, in real time, per family)."""
    runners = {"muse": run_muse, "hand": run_hand, "both": run_both}
    return [(name, runners[name]) for name in families]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="mock", choices=["mock", "isaac"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--families", default="muse,hand,both")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None, help="app listen port (default: spare port when spawning)")
    ap.add_argument("--no-spawn", action="store_true", help="drive an already-running app (e.g. the live Isaac stack)")
    ap.add_argument("--config", default=str(G.ROOT / "configs/default.yaml"))
    ap.add_argument("--poll-hz", type=float, default=4.0)
    a = ap.parse_args()
    families = [f.strip() for f in a.families.split(",") if f.strip()]

    app = None
    if a.no_spawn:
        port = a.port or 47100
    else:
        port = a.port or G.SPARE_PORTS["listen_port"]
        tmp = G.ROOT / "logs" / "grip_scenarios_cfg.yaml"
        ports = dict(G.SPARE_PORTS, listen_port=port)
        G.write_spare_config(a.config, tmp, ports)
        app = G.AppProcess(str(tmp), a.backend, log_path=G.ROOT / "logs" / "grip_scenarios_app.log").start()
        if not app.wait_ready(port, a.host):
            app.stop()
            raise SystemExit("mock app did not come up on the spare port")

    snd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(m):
        snd.sendto(__import__("json").dumps(m).encode(), (a.host, port))

    mon = G.AppMonitor(a.host, port, hz=a.poll_hz)
    mon.start()
    all_cmds: list[dict] = []
    log_start = time.time()
    try:
        for seed in range(a.seeds):
            for name, runner in build_commands(seed, families):
                p = Player(send, np.random.default_rng(1000 * seed + hash(name) % 997))
                p.engage()
                runner(p)
                for c in p.commands:
                    c["seed"] = seed
                all_cmds.extend(p.commands)
                # reset between runs: opens the gripper, clears the source, releases the clutch, so the next family
                # starts from a known open state and a fresh calibration
                send({"kind": "utterance", "t": time.time(), "text": "reset", "offline": True})
                time.sleep(0.6)
        time.sleep(0.5)
    finally:
        mon.stop()
        mon.join(timeout=2)
        if app is not None:
            rc = app.stop()
        else:
            rc = None

    log_text = ""
    if app is not None:
        log_text = Path(app.log_path).read_text()
    parsed = G.parse_app_log(log_text) if log_text else {"transitions": [], "toggles": [], "clashes": [],
                                                          "errors": ["(no app log: --no-spawn, cannot read errors)"]}
    agg = evaluate(all_cmds, parsed, mon, a.backend)
    # seeds per family
    for family in agg:
        agg[family]["seeds"] = len({c["seed"] for c in all_cmds if c["family"] == family})
    print_table(agg, a.backend, mon.followed(), mon.tcp_path_following(), len(parsed["errors"]))
    if rc is not None:
        print(f"(mock app exit code {rc}; -15 = SIGTERM from this harness, not a crash)")


if __name__ == "__main__":
    main()
