"""Tiny JSON-over-UDP link (laptop -> VM over the tailnet). Latest-message-wins; no deps.

Messages are the pydantic types in rover_intent.types, discriminated by their `kind` field.
"""
from __future__ import annotations

import json
import socket
import time

from pydantic import TypeAdapter

from ..types import BodyPose, EEGState, Utterance

_MSG = TypeAdapter(BodyPose | Utterance | EEGState)


class Sender:
    def __init__(self, host: str, port: int):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, msg) -> None:
        self.sock.sendto(msg.model_dump_json().encode(), self.addr)


class Receiver:
    """Non-blocking. `poll()` drains the socket and returns every message received since the last poll."""

    def __init__(self, port: int, host: str = "0.0.0.0"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)

    def poll(self) -> list:
        out = []
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
            except BlockingIOError:
                return out
            try:
                out.append((time.time(), _MSG.validate_python(json.loads(data))))
            except Exception:
                pass  # drop malformed packets
