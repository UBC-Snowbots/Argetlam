"""Session recording: one JSON line per incoming message (receive time + the message), so a live run can be
replayed offline through the app (scripts/replay_session.py). On by default; the app's --no-record turns it off.

Query messages are skipped (the follower and tools poll the app many times a second; they are not inputs worth
replaying). CameraFrame messages are skipped too (a 10 Hz base64 JPEG stream is huge and is not a control input).
A size cap stops a long session from filling the disk; once hit, recording stops and says so once.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .types import CameraFrame, Query

log = logging.getLogger("rover_intent")


class SessionRecorder:
    def __init__(self, directory, enabled: bool = True, size_cap_mb: float = 50.0, clock=time.time):
        self.enabled = enabled
        self.clock = clock
        self.size_cap = int(size_cap_mb * 1_000_000)
        self.bytes = 0
        self._capped = False
        self._f = None
        self.path: Path | None = None
        if enabled:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / (time.strftime("%Y%m%d_%H%M%S") + ".jsonl")
            self._f = open(self.path, "a", buffering=1)   # line-buffered: a crash still leaves a readable file
            log.info("recording session to %s (--no-record to disable)", self.path)

    def record(self, recv_t: float, msg) -> None:
        """Append one message. recv_t is when the app received it (poll time); msg is a pydantic message model."""
        if not self.enabled or self._capped or self._f is None or isinstance(msg, (Query, CameraFrame)):
            return
        line = json.dumps({"t": round(float(recv_t), 4), "msg": msg.model_dump(mode="json")}) + "\n"
        enc = line.encode()
        if self.bytes + len(enc) > self.size_cap:
            self._capped = True
            log.warning("session recording hit the %d MB cap (%s): stopping", self.size_cap // 1_000_000, self.path)
            self.close()
            return
        self._f.write(line)
        self.bytes += len(enc)

    def close(self) -> None:
        if self._f is not None:
            try:
                self._f.close()
            finally:
                self._f = None
