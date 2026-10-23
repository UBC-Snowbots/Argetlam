"""Offline checks for the laptop<->VM tunnel path (issue 6): the TCP relay forwards both ways and logs timestamped
connect/disconnect, and the body client's Link reconnects (counting drops) when the tunnel bounces."""
import importlib.util
import socket
import sys
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _stub_laptop_deps():
    """body_client.py imports cv2 / mediapipe (laptop-only). Stub them so we can load just its Link class here."""
    for name in ["cv2", "mediapipe", "mediapipe.tasks", "mediapipe.tasks.python"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    mp = sys.modules["mediapipe"]
    mp.tasks = sys.modules["mediapipe.tasks"]
    mp.tasks.python = sys.modules["mediapipe.tasks.python"]
    mp.Image = object
    mp.ImageFormat = types.SimpleNamespace(SRGB=0)
    tp = sys.modules["mediapipe.tasks.python"]
    tp.BaseOptions = object
    tp.vision = types.ModuleType("vision")


def _load(path, name):
    if "body_client" in path:
        _stub_laptop_deps()
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_relay_forwards_both_ways_and_timestamps(capsys):
    relay = _load("scripts/tcp_relay.py", "tcp_relay")
    assert len(relay._ts()) == len("2026-10-23 00:00:00")        # timestamp helper present and shaped
    tcp_port, app_port = _free_port(), _free_port()
    app = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    app.bind(("127.0.0.1", app_port))
    app.settimeout(3)
    argv = ["tcp_relay.py", "--port", str(tcp_port), "--app", f"127.0.0.1:{app_port}"]
    th = threading.Thread(target=lambda: _run(relay, argv), daemon=True)
    th.start()
    time.sleep(0.4)
    body = _load("laptop/body_client.py", "body_client")
    link = body.Link("127.0.0.1", 0, tunnel=f"127.0.0.1:{tcp_port}")
    link.send({"kind": "body", "t": 1.0})
    data, src = app.recvfrom(9999)
    assert b'"kind": "body"' in data
    app.sendto(b'{"kind":"status","mode":"hold"}', src)          # reply back through the relay
    time.sleep(0.2)
    msgs = link.poll()
    assert any(m.get("kind") == "status" for m in msgs)
    assert link.reconnects == 0
    link.sock.close()
    time.sleep(0.3)
    out = capsys.readouterr().out
    assert "client connected" in out and "[relay]" in out


def _run(relay, argv):
    old = sys.argv
    sys.argv = argv
    try:
        relay.main()
    except Exception:
        pass
    finally:
        sys.argv = old


def test_link_reconnect_counts_a_bounce():
    body = _load("laptop/body_client.py", "body_client2")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.listen()
    accepted = []

    def accept_loop():
        while True:
            try:
                c, _ = srv.accept()
                accepted.append(c)
            except OSError:
                break

    threading.Thread(target=accept_loop, daemon=True).start()
    link = body.Link("127.0.0.1", 0, tunnel=f"127.0.0.1:{port}")
    time.sleep(0.2)
    assert link.reconnects == 0 and len(accepted) == 1
    accepted[0].close()            # server drops the connection
    time.sleep(0.1)
    link.poll()                    # the per-frame poll sees EOF and reconnects
    time.sleep(0.2)
    assert link.reconnects >= 1 and len(accepted) >= 2   # reconnected: the server accepted a second time
    srv.close()
