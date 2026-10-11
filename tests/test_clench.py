"""Muse 2 double-jaw-clench toggle: detector on synthetic Muse-like signals, arbiter toggle, UDP message."""
import json

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from rover_intent.eeg.clench import ClenchDetector, calibrate, main as clench_main

FS = 256.0
TP, FRONT = [0, 3], [1, 2]  # TP9, AF7, AF8, TP10


def simulate(duration, clenches=(), blinks=(), talk=(), seed=0, line_uv=15.0, bad_channel=None, emg_uv=80.0):
    """Muse-like 4-channel signal in uV. clenches: (start_s, dur_s[, amp_uv]); blinks: times; talk: (start, dur).
    ASSUMED levels (check against a real calibration): EEG is mostly below 30 Hz, a few uV of floor above it; a firm
    jaw clench is tens to hundreds of uV of broadband EMG, strongest at TP9/TP10."""
    rng = np.random.default_rng(seed)
    n = int(duration * FS)
    t = np.arange(n) / FS
    lp = butter(4, 30, fs=FS, output="sos")
    x = sosfilt(lp, rng.normal(0, 25, (4, n)), axis=1)  # slow background EEG
    x += rng.normal(0, 1.5, (4, n))  # amplifier / dry-contact noise floor
    x += 4 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6, (4, 1)))  # alpha
    x += line_uv * np.sin(2 * np.pi * 60 * t + rng.uniform(0, 6, (4, 1)))  # mains hum
    emg_sos = butter(4, [20, 120], btype="band", fs=FS, output="sos")

    def burst(start, dur, amp):
        i0, i1 = int(start * FS), min(n, int((start + dur) * FS))
        if i1 <= i0:
            return
        k = i1 - i0
        ramp = np.minimum(1, np.minimum(np.arange(k), np.arange(k)[::-1]) / (0.02 * FS))
        e = sosfilt(emg_sos, rng.normal(0, 1, (4, k)), axis=1)
        e /= e.std(axis=1, keepdims=True)
        gain = np.array([1.0, 0.6, 0.6, 1.0])[:, None] * amp
        x[:, i0:i1] += e * gain * ramp

    for c in clenches:
        burst(c[0], c[1], c[2] if len(c) > 2 else emg_uv)
    for b in blinks:
        g = np.exp(-0.5 * ((t - b) / 0.08) ** 2)
        x[FRONT] += 150 * g
        x[TP] += 15 * g
    for s, d in talk:  # talking: short irregular weak EMG bursts
        u = s
        while u < s + d:
            dur = rng.uniform(0.08, 0.25)
            burst(u, dur, rng.uniform(8, 15))
            u += dur + rng.uniform(0.05, 0.3)
    if bad_channel is not None:  # loose electrode: big broadband noise on one channel all the time
        x[bad_channel] += rng.normal(0, 60, n)
    return x


def double(at, gap=0.35, dur=0.3):
    return [(at, dur), (at + dur + gap, dur)]


@pytest.fixture(scope="module")
def profile():
    rest = simulate(20, blinks=np.arange(1.5, 20, 3.1), seed=1)
    clench = simulate(24, clenches=[(i * 3 + 0.8, 0.5) for i in range(8)], blinks=[2.2, 11.4], seed=2)
    chew = [(10 + 0.7 * i, 0.3, 50) for i in range(12)]
    other = simulate(20, talk=[(0.5, 9)], clenches=chew, seed=3)
    return calibrate(rest, clench, other, FS)


def run(profile, x, chunk=32):
    d = ClenchDetector.from_profile(profile)
    return [t for i in range(0, x.shape[1], chunk) for t in d.push(x[:, i:i + chunk])], d


def test_calibration_is_good_and_threshold_sits_between_rest_and_clench(profile):
    s = profile["stats"]
    assert s["quality"] == "good", s
    assert s["rest_p99"] < s["on_ratio"] < s["clench_p95"]
    assert s["clench_detected"] == 8 and s["rest_toggles"] == 0 and s["other_toggles"] == 0
    json.dumps(profile)  # saveable


def test_double_clench_toggles_once_with_bounded_latency(profile):
    x = simulate(5, clenches=double(1.0), seed=10)
    toggles, d = run(profile, x)
    assert len(toggles) == 1
    first_onset = 1.0
    assert toggles[0] - first_onset < 1.8  # 0.3 + 0.35 + 0.3 s of clenching + release + confirm


def test_single_and_far_apart_clenches_do_nothing(profile):
    assert run(profile, simulate(4, clenches=[(1.0, 0.3)], seed=11))[0] == []
    assert run(profile, simulate(6, clenches=[(1.0, 0.3), (2.6, 0.3)], seed=12))[0] == []  # 1.3 s gap


def test_two_doubles_give_two_toggles(profile):
    x = simulate(9, clenches=double(1.0) + double(5.0), seed=13)
    assert len(run(profile, x)[0]) == 2


def test_triple_clench_and_chewing_are_ignored(profile):
    triple = [(1.0, 0.25), (1.6, 0.25), (2.2, 0.25)]
    assert run(profile, simulate(5, clenches=triple, seed=14))[0] == []
    chewing = [(1.0 + 0.7 * i, 0.3, 60) for i in range(14)]
    toggles, d = run(profile, simulate(12, clenches=chewing, seed=15))
    assert toggles == [] and any(k == "chew" for k, _ in d.log)


def test_double_right_after_chewing_needs_quiet_first(profile):
    chewing = [(1.0 + 0.7 * i, 0.3, 60) for i in range(6)]
    late = double(chewing[-1][0] + 0.3 + 1.6)  # after a quiet second: counts
    assert len(run(profile, simulate(10, clenches=chewing + late, seed=16))[0]) == 1


def test_sustained_clench_is_not_a_toggle(profile):
    assert run(profile, simulate(6, clenches=[(1.0, 3.0)], seed=17))[0] == []
    x = simulate(9, clenches=[(1.0, 3.0)] + double(5.5), seed=18)
    assert len(run(profile, x)[0]) == 1


def test_blinks_talking_and_mains_never_toggle(profile):
    x = simulate(30, blinks=np.arange(0.5, 30, 0.9), talk=[(2, 25)], line_uv=40.0, seed=19)
    assert run(profile, x)[0] == []


def test_one_loose_electrode_cannot_trigger_but_clench_still_works(profile):
    assert run(profile, simulate(10, bad_channel=0, seed=20))[0] == []
    assert len(run(profile, simulate(5, clenches=double(1.0), bad_channel=0, seed=21))[0]) == 1


@pytest.mark.parametrize("chunk", [1, 7, 256])
def test_chunk_size_does_not_matter(profile, chunk):
    x = simulate(9, clenches=double(1.0) + double(5.0), seed=22)
    ref, _ = run(profile, x, 32)
    got, _ = run(profile, x, chunk)
    assert len(got) == len(ref) == 2 and np.allclose(got, ref)


def test_poor_contact_calibration_is_flagged():
    rest = simulate(12, seed=30)
    weak = simulate(16, clenches=[(i * 2 + 0.5, 0.4, 3.0) for i in range(8)], seed=31)  # barely-there clenches
    assert calibrate(rest, weak, None, FS)["stats"]["quality"] == "poor"


def test_replay_cli(profile, tmp_path, capsys):
    prof = tmp_path / "p.json"
    prof.write_text(json.dumps(profile))
    rec = tmp_path / "r.npz"
    np.savez(rec, rest=simulate(6, seed=40), clench=simulate(6, clenches=double(1.0), seed=41), fs=FS)
    assert clench_main(["replay", "--profile", str(prof), "--file", str(rec)]) == 0
    rows = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
    assert rows[0]["toggles_at_s"] == [] and len(rows[1]["toggles_at_s"]) == 1


# --- arbiter + message ------------------------------------------------------------------------------------------------
def _arbiter():
    from test_core import SCENE  # same scene as the rest of the arbiter tests
    from rover_intent.control.arbiter import Arbiter
    from rover_intent.planner.scene import Scene
    return Arbiter(Scene(SCENE)), np.array(SCENE["cup"]["pos"])


def test_toggle_only_in_follow_and_flips_the_gripper():
    from rover_intent.types import Action, Intent
    a, cup = _arbiter()
    far = np.array([0.40, 0.0, 0.30])
    a.on_grip_toggle(far)
    assert a.gripper == 0.0  # HOLD: ignored
    a.on_intent(Intent(action=Action.follow))
    a.on_grip_toggle(far)
    assert a.gripper == 1.0 and "nothing in reach" in a.last_say
    a.on_grip_toggle(far)
    assert a.gripper == 0.0


def test_toggle_near_object_starts_grab_assist_and_does_not_fight_voice():
    from rover_intent.control.arbiter import Mode
    from rover_intent.types import Action, Intent
    a, cup = _arbiter()
    a.on_intent(Intent(action=Action.follow))
    a.on_grip_toggle(cup + [0.03, 0.0, 0.06])
    assert a.mode == Mode.AUTO and a.assist and "cup" in a.last_say
    b, _ = _arbiter()
    b.on_intent(Intent(action=Action.follow))
    b.on_intent(Intent(action=Action.grasp))  # voice closed it
    b.on_intent(Intent(action=Action.release))  # voice opened it again
    b.on_grip_toggle(np.array([0.40, 0.0, 0.30]))
    assert b.gripper == 1.0  # the toggle acts on the CURRENT state, no stale level to fight


def test_grip_toggle_message_parses():
    from rover_intent.transport.udp import _MSG
    from rover_intent.types import GripToggle
    m = _MSG.validate_python(json.loads(GripToggle(t=1.0).model_dump_json()))
    assert isinstance(m, GripToggle) and m.source == "clench"


def test_randomized_doubles_and_everyday_jaw_activity(profile):
    """Quick doubles (0.15 s relax) must still count; minutes of talking/blinking/chewing must not."""
    rng = np.random.default_rng(7)
    hits = 0
    for k in range(25):
        d1, d2, gap, amp = *rng.uniform(0.15, 0.5, 2), rng.uniform(0.15, 0.7), rng.uniform(40, 120)
        x = simulate(5, clenches=[(1.0, d1, amp), (1.0 + d1 + gap, d2, amp)], blinks=[rng.uniform(0, 5)], seed=500 + k)
        hits += len(run(profile, x)[0]) == 1
    assert hits >= 24
    for k in range(6):  # 3 minutes
        s0 = rng.uniform(0, 20)
        chew = [(s0 + 0.65 * i + rng.uniform(-0.05, 0.05), 0.3, rng.uniform(40, 90)) for i in range(int(rng.integers(4, 12)))]
        x = simulate(30, talk=[(rng.uniform(0, 5), 15)], blinks=np.sort(rng.uniform(0, 30, 10)), clenches=chew, seed=600 + k)
        assert run(profile, x)[0] == []


def test_grip_toggle_survives_a_real_udp_hop():
    from rover_intent.transport.udp import Receiver, Sender
    from rover_intent.types import GripToggle
    import time
    rx = Receiver(0, "127.0.0.1")  # ephemeral port: never touches the running app on 47100
    port = rx.sock.getsockname()[1]
    Sender("127.0.0.1", port).send(GripToggle(t=1.0))
    msgs = []
    for _ in range(50):
        msgs += rx.poll()
        if msgs:
            break
        time.sleep(0.01)
    assert len(msgs) == 1 and isinstance(msgs[0][1], GripToggle)


def test_standalone_packet_is_a_valid_grip_toggle():
    from rover_intent.eeg.clench import toggle_packet
    from rover_intent.transport.udp import _MSG
    from rover_intent.types import GripToggle
    assert isinstance(_MSG.validate_python(json.loads(toggle_packet(12.5))), GripToggle)
