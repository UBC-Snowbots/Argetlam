"""LAPTOP: push-to-talk mic -> local Whisper (faster-whisper) -> Utterance over UDP to the VM.

    python laptop/stream_voice.py --host <vm-tailnet-ip> [--model small.en]

Press Enter to start recording, Enter again to stop. (Wake word / VAD later.)
Speech-to-text stays local; only the transcript goes on to Nemotron.
"""
import argparse
import threading
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from rover_intent.transport.udp import Sender
from rover_intent.types import Utterance

SR = 16000


def record_until_enter() -> np.ndarray:
    chunks, stop = [], threading.Event()

    def cb(indata, frames, t, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=cb):
        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        stop.wait()
    return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=47100)
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    tx = Sender(args.host, args.port)
    model = WhisperModel(args.model, device=args.device)
    while True:
        input("[Enter] to talk ")
        print("recording... [Enter] to stop")
        audio = record_until_enter()
        segs, _ = model.transcribe(audio, language="en", vad_filter=True)
        text = " ".join(s.text.strip() for s in segs).strip()
        print(f"> {text!r}")
        if text:
            tx.send(Utterance(t=time.time(), text=text))


if __name__ == "__main__":
    main()
