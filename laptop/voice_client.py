"""LAPTOP voice client: push-to-talk mic -> local Whisper -> text -> robot (UDP over tailnet) -> prints the reply.

Standalone: needs only this file (no repo checkout). Windows / macOS / Linux.

    python -m venv rover-voice
    rover-voice\\Scripts\\activate          (Windows)   |   source rover-voice/bin/activate   (mac/Linux)
    pip install faster-whisper sounddevice numpy
    python voice_client.py --list-devices          # find your mic's index
    python voice_client.py --text                  # 1) check the link: type commands, no mic
    python voice_client.py [--mic N]               # 2) talk: Enter to start, Enter to stop

Speech-to-text runs locally (faster-whisper, CPU int8); only the transcript goes to the robot, where
NVIDIA Nemotron on Nebius Token Factory turns it into a robot command.
"""
import argparse
import json
import os
import socket
import threading
import time

DEFAULT_HOST = os.environ.get("ROVER_HOST", "127.0.0.1")  # the VM's address (tailnet IP or hostname); or pass --host
PORT = 47100
SR = 16000


class Link:
    """UDP straight to the VM, or (--tunnel host:port) newline-JSON over TCP through an SSH tunnel to
    scripts/tcp_relay.py on the VM. Same messages either way."""

    def __init__(self, host, port, tunnel=None):
        self.tunnel = tunnel
        if tunnel:
            th, tp = tunnel.rsplit(":", 1)
            self.sock = socket.create_connection((th, int(tp)), timeout=5)
            self.sock.setblocking(False)
            self._buf = b""
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setblocking(False)
            self.addr = (host, port)

    def send(self, msg: dict):
        data = json.dumps(msg).encode()
        if self.tunnel:
            self.sock.setblocking(True)
            self.sock.sendall(data + b"\n")
            self.sock.setblocking(False)
        else:
            self.sock.sendto(data, self.addr)

    def poll(self):
        """All messages received so far (non-blocking)."""
        out = []
        while True:
            try:
                data = self.sock.recv(65536) if self.tunnel else self.sock.recvfrom(65536)[0]
            except (BlockingIOError, ConnectionResetError, OSError):
                break
            if self.tunnel:
                if not data:
                    break
                self._buf += data
                *lines, self._buf = self._buf.split(b"\n")
                out += [json.loads(x) for x in lines if x.strip()]
            else:
                try:
                    out.append(json.loads(data))
                except ValueError:
                    pass
        return out

    def wait(self, timeout, want=lambda m: True):
        """Block up to `timeout` s for a message matching `want`."""
        import time as _t
        end = _t.time() + timeout
        while _t.time() < end:
            for m in self.poll():
                if want(m):
                    return m
            _t.sleep(0.02)
        return None


def send_and_wait(link, text, timeout=12.0):
    link.poll()  # drop anything stale (status messages)
    link.send({"kind": "utterance", "t": time.time(), "text": text})
    r = link.wait(timeout, lambda m: m.get("kind") == "reply" and m.get("text") == text)
    if r is None:
        print("  !! no reply in %.0fs: is the app running on the VM? (UDP blocked? try --tunnel)" % timeout)
        return
    it = r.get("intent", {})
    what = " ".join(f"{k}={v}" for k, v in it.items() if k != "say")
    print(f"  robot: {r.get('say', '')!r}   [{what}]  mode={r.get('mode')}  nemotron {r.get('latency_s', '?')}s")


def record_until_enter(sd, np, mic):
    chunks, stop = [], threading.Event()

    def cb(indata, frames, t, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=cb, device=mic):
        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        stop.wait()
    return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--tunnel", help="host:port of an SSH tunnel to scripts/tcp_relay.py (when UDP can't reach the VM)")
    ap.add_argument("--text", action="store_true", help="type commands instead of speaking")
    ap.add_argument("--mic", type=int, default=None, help="input device index (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--model", default="base.en", help="whisper size: tiny.en/base.en/small.en")
    args = ap.parse_args()
    link = Link(args.host, args.port, args.tunnel)

    if args.text:
        print(f"text mode -> {args.host}:{args.port}. Try: move the cup beside the bottle | stop | follow me")
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if line:
                send_and_wait(link, line)

    import numpy as np
    import sounddevice as sd
    if args.list_devices:
        print(sd.query_devices())
        return
    from faster_whisper import WhisperModel
    print(f"loading whisper {args.model} (first run downloads it)...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    mic_name = sd.query_devices(args.mic if args.mic is not None else sd.default.device[0])["name"]
    print(f"mic: {mic_name}   robot: {args.host}:{args.port}")
    while True:
        try:
            input("\n[Enter] to talk ")
        except (EOFError, KeyboardInterrupt):
            return
        print("  listening... [Enter] to stop")
        audio = record_until_enter(sd, np, args.mic)
        if len(audio) < SR * 0.3:
            print("  (too short)")
            continue
        level = float(np.sqrt(np.mean(audio ** 2)))
        t0 = time.time()
        segs, _ = model.transcribe(audio, language="en", vad_filter=True, beam_size=1)
        text = " ".join(s.text.strip() for s in segs).strip()
        print(f"  you said: {text!r}   (whisper {time.time() - t0:.1f}s, {len(audio) / SR:.1f}s audio, level {level:.3f})")
        if not text:
            print("  (nothing recognised - mic muted or wrong --mic?)")
            continue
        send_and_wait(link, text)


if __name__ == "__main__":
    main()
