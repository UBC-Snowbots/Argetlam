"""Push-to-talk on top of the VM's voice client (vm_voice_client.py, synced by rover.py; never edited here).

Adds: global Shift+2 (laptop windows) + Shift+2 inside Moonlight (via hotkey_relay.py on the VM) + Enter in this
window, and a speech bubble on the VM screen (bubble.py). Uses the VM client's Link, so --tunnel works.

    python voice_plus.py --tunnel 127.0.0.1:47120 [--mic N] [--model base.en]
"""
import argparse
import json
import os
import queue
import socket
import sys
import threading
import time

try:
    import vm_voice_client as vc   # the laptop: rover.py syncs laptop/voice_client.py under this name
except ImportError:
    import voice_client as vc      # run straight from rover-intent/laptop/

SR = vc.SR
BUBBLE_PORT, RELAY_PORT = 47101, 47102


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=vc.DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=vc.PORT)
    ap.add_argument("--tunnel")
    ap.add_argument("--mic", type=int, default=None)
    ap.add_argument("--model", default="base.en")
    ap.add_argument("--threads", type=int, default=8,
                    help="whisper CPU threads (8 measured ~0.40-0.46 s/command vs ~0.48-0.57 s at the default 4)")
    args = ap.parse_args()

    link = vc.Link(args.host, args.port, args.tunnel)
    usock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def bubble(state, text=""):
        try:
            usock.sendto(json.dumps({"state": state, "text": text}).encode(), (args.host, BUBBLE_PORT))
        except OSError:
            pass

    def converse(text):
        bubble("heard", text)
        link.poll()
        link.send({"kind": "utterance", "t": time.time(), "text": text})
        r = link.wait(12.0, lambda m: m.get("kind") == "reply" and m.get("text") == text)
        if r is None:
            print("  !! no reply in 12s: is the app running on the VM?", flush=True)
            bubble("error", "no reply from the robot app")
            return
        it = r.get("intent", {})
        what = " ".join(f"{k}={v}" for k, v in it.items() if k != "say")
        print(f"  robot: {r.get('say', '')!r}   [{what}]  mode={r.get('mode')}  nemotron {r.get('latency_s', '?')}s",
              flush=True)
        bubble("reply", r.get("say", ""))

    import numpy as np
    import sounddevice as sd
    from faster_whisper import WhisperModel
    print(f"loading whisper {args.model} ({args.threads} threads)...", flush=True)
    model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=args.threads)
    mic_name = sd.query_devices(args.mic if args.mic is not None else sd.default.device[0])["name"]
    print(f"mic: {mic_name}   link: {'tunnel ' + args.tunnel if args.tunnel else 'udp ' + args.host}", flush=True)

    chunks, recording, done, last = [], threading.Event(), queue.Queue(), [0.0]
    stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", device=args.mic,
                            callback=lambda indata, f, t, st: chunks.append(indata.copy()) if recording.is_set() else None)
    stream.start()

    def toggle(src):
        if time.time() - last[0] < 0.4:  # same press seen by two triggers
            return
        last[0] = time.time()
        if not recording.is_set():
            chunks.clear()
            recording.set()
            print(f"\n  [{src}] listening... (same key again to send)", flush=True)
            bubble("listening")
        else:
            recording.clear()
            done.put(np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, np.float32))

    # 1) Shift+2 in any laptop window (physical "2" + Shift held; ignore key-repeat)
    try:
        import keyboard
        held = threading.Event()

        def on_down(e):
            if not held.is_set():
                held.set()
                if keyboard.is_pressed("shift"):
                    toggle("Shift+2")

        keyboard.on_press_key("2", on_down)
        keyboard.on_release_key("2", lambda e: held.clear())
    except Exception as e:  # noqa: BLE001
        print(f"  (laptop hotkey unavailable: {e})", flush=True)

    # 2) Shift+2 inside Moonlight -> VM hotkey_relay.py -> here
    rsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rsock.settimeout(1.0)

    def relay():
        nxt = 0.0
        while True:
            if time.time() >= nxt:
                try:
                    rsock.sendto(b'{"kind": "register"}', (args.host, RELAY_PORT))
                except OSError:
                    pass
                nxt = time.time() + 3.0
            try:
                data, _ = rsock.recvfrom(4096)
                if json.loads(data).get("kind") == "toggle":
                    toggle("Shift+2 in Moonlight")
            except (socket.timeout, OSError, ValueError):
                continue

    threading.Thread(target=relay, daemon=True).start()

    # 3) Enter in this window (always works)
    def enter():
        # This thread used to return on EOFError / KeyboardInterrupt, which silently killed Enter for the rest of
        # the session (2026-10-28: py-spy showed every thread alive except this one). Keep listening instead:
        # ignore a stray Ctrl+C, and on EOF reopen the console input.
        warned = False
        while True:
            try:
                input()
            except KeyboardInterrupt:
                continue
            except EOFError:
                if not warned:
                    print("  (console input closed; reopening it - Shift+2 also works)", flush=True)
                    warned = True
                try:
                    sys.stdin = open("CONIN$" if os.name == "nt" else "/dev/tty")
                except OSError:
                    pass
                time.sleep(0.5)
                continue
            warned = False
            toggle("Enter")

    threading.Thread(target=enter, daemon=True).start()

    print("\nready: Shift+2 (anywhere, incl. Moonlight) or Enter here = start; same again = send. Ctrl+C quits.",
          flush=True)
    try:
        while True:
            try:
                audio = done.get(timeout=0.5)
            except queue.Empty:
                continue
            if len(audio) < SR * 0.3:
                print("  (too short)", flush=True)
                bubble("error", "too short")
                continue
            t0 = time.time()
            segs, _ = model.transcribe(audio, language="en", vad_filter=True, beam_size=1)
            text = " ".join(s.text.strip() for s in segs).strip()
            print(f"  you said: {text!r}   (whisper {time.time() - t0:.1f}s, {len(audio) / SR:.1f}s audio)", flush=True)
            if not text:
                print("  (nothing recognised - mic muted?)", flush=True)
                bubble("error", "didn't catch that")
                continue
            converse(text)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()


if __name__ == "__main__":
    main()
