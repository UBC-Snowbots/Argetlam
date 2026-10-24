"""arm.on_drop: when your arm drops out of view while following, the arm glides to the named preset (dev_arm_real: base)
instead of freezing; a lost link (nobody watching) still freezes in place."""
import json
import socket
import threading
import time

import numpy as np
import pytest

from rover_intent import app
from rover_intent.config import load_config
from test_core import _synthetic_arm


def _run(port, lose):
    cfg = load_config(arm="dev_arm_real")
    cfg["body"]["mode"] = "arm_mirror"   # plane is the dev_arm_real default now; this on_drop test drives arm_mirror
    cfg["net"].update(listen_port=port, hud_port=port + 50, bubble_port=port + 51)
    cfg["body"]["clutch"] = {"calib_s": 0.5, "still_m": 0.03, "palm_s": 1.0, "raise_m": 0.05}
    cfg["nemotron"]["api_key_env"] = "NO_SUCH_KEY_FOR_TESTS"
    base = np.asarray(cfg["arm"]["presets"]["base"])
    th = threading.Thread(target=app.run, args=(cfg, "mock", 14.0), kwargs={"record": False}, daemon=True)
    th.start(); time.sleep(0.5)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(3)
    addr = ("127.0.0.1", port)

    def send(m):
        s.sendto(json.dumps(m).encode(), addr)

    def query():
        send({"kind": "query", "t": time.time()})
        while True:
            m = json.loads(s.recvfrom(65536)[0])
            if m.get("kind") == "state":
                return m

    def drain():
        s.settimeout(0.001)
        try:
            while True:
                s.recvfrom(65536)
        except (socket.timeout, BlockingIOError):
            pass
        s.settimeout(3)

    send({"kind": "utterance", "t": time.time(), "text": "follow me"})
    for i in range(110):  # hold still to calibrate, then swing the hand left so the arm leaves base
        k = max(0, i - 24)
        p = _synthetic_arm(lateral=min(0.15, k * 0.005), t=time.time()).model_dump()
        p.update(kind="body", visible=True)
        send(p); time.sleep(1 / 30)
        drain()
    moved = query()
    assert moved["clutch"] == "following"
    off_base = float(np.max(np.abs(np.asarray(moved["q_meas"]) - base)))
    assert off_base > 0.02, "the arm should have left base while following"
    t_end = time.time() + 7.0
    while time.time() < t_end:
        if lose == "tracking":   # packets keep coming, but the arm isn't visible: your arm dropped
            p = _synthetic_arm(lateral=0.15, t=time.time()).model_dump()
            p.update(kind="body", visible=False)
            send(p)
        time.sleep(1 / 30)       # lose == "link": no packets at all
        drain()
    end = query()
    th.join(timeout=10)
    return off_base, end, base


@pytest.mark.parametrize("lose", ["tracking", "link"])
def test_drop_goes_to_base_only_on_tracking_lost(lose):
    off_base, end, base = _run(47186 + (lose == "link"), lose)
    err = float(np.max(np.abs(np.asarray(end["q_meas"]) - base)))
    assert end["clutch"] == "idle"
    if lose == "tracking":
        assert err < 0.01, f"arm should be back at base, still {err:.3f} rad away"
    else:
        assert err > off_base * 0.5, "link lost must freeze the arm in place, not move it to base"


def test_on_drop_must_name_a_preset():
    cfg = load_config(arm="dev_arm_real")
    cfg["arm"]["on_drop"] = "nowhere"
    cfg["net"].update(listen_port=47188, hud_port=47238, bubble_port=47239)
    with pytest.raises(ValueError, match="on_drop"):
        app.run(cfg, "mock", 0.1, record=False)
