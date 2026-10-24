# Live webcam panel under the HUD

The laptop body client sends its annotated webcam frame to the VM, and the app forwards it to a video panel drawn
directly below the existing Isaac HUD (`scripts/hud.py`). This lets the operator watch one VM screen (skeleton PiP,
sync, latency, and now the camera) instead of looking back at the laptop's OpenCV window.

## Path

```
laptop/body_client.py  --(CameraFrame, JSON/UDP)-->  app (net.listen_port)
       VideoStreamer                                    forwards verbatim to net.hud_video_port
                                                 -->  scripts/hud.py (binds hud_video_port, decodes with Pillow)
```

- `CameraFrame` is a separate message type (`src/rover_intent/types.py`): a base64 JPEG plus `width/height/seq/t`.
  It rides the same JSON transport as every other message, so it also works over the SSH tunnel (`--tunnel`).
- It is **never** queued behind pose packets. `VideoStreamer` is rate limited and drops frames in between, so the
  pose/control latency is untouched. The app forwards the frame straight to the HUD video port and keeps it off the
  pose path; the HUD keeps only the newest frame (older ones are dropped).
- Camera frames are **not** recorded to the session log (`session.py` skips them, like `Query`), so a replay stays
  small and replayable.

## Enable / disable and flags (laptop/body_client.py)

| flag | default | meaning |
|---|---|---|
| `--stream-video` | on | send the annotated webcam frame to the HUD video panel |
| `--no-video` | - | do not send the video stream |
| `--video-width` / `--video-height` | 480 / 360 | downscale size sent over the wire (tracking still runs full-res) |
| `--video-quality` | 60 | JPEG quality; lower means smaller frames |
| `--video-hz` | 10 | stream frame rate, separate from the pose rate |
| `--window-width` / `--window-height` | 960 / 540 | laptop display window size (smaller keeps it off the screen edge) |
| `--panel-margin` | 40 | gap from the right edge to the button panel |

HUD side: `python3 scripts/hud.py` binds `--video-port` (default 47131, matches `net.hud_video_port`). The HUD needs
**Pillow** for the JPEG decode (`pip install pillow`); without it the panel shows "video needs Pillow" and the rest
of the HUD still works. The panel shows the live frame rate, or "stale N.Ns" in red if frames stop arriving (> 1 s).

## UDP size note (read before a real-network run)

At the defaults (480x360, quality 60) a frame is roughly a 22 KB JPEG, about a 30 KB datagram after base64 + the JSON
envelope. That is fine on localhost and on a good LAN, but a datagram that large is split into ~20 IP fragments, and
over a lossy link losing any one fragment drops the whole frame. `VideoStreamer` warns once if a frame looks too big
for plain UDP and tells you to lower `--video-quality` / `--video-width` or use `--tunnel`. The TCP tunnel has no
datagram-size limit, so for a flaky link prefer `--tunnel`, or drop the quality/size.

## Measured (spare app, mock backend, localhost, 2026-10-24)

A synthetic 480x360 JPEG stream at 10 Hz through the real `CameraFrame` path (spare ports, live stack untouched):

- achieved video rate at the HUD port: **9.9 Hz** (99/99 frames, seq contiguous, no drops)
- JPEG payload: mean **~22 KB** (min 19 KB, max 25 KB); forwarded datagram mean **~30 KB**
- app to HUD forward latency: median **~11 ms**, max ~20 ms (dominated by the app's 50 Hz poll tick)
- pose/status path while video streams: **30 pkt/s, 0 loss, ~0 s echo age** (unaffected by the video)
- a forwarded frame decodes with Pillow to 480x360 (the HUD decode path)

These are localhost numbers. Real laptop camera content, the tailnet/tunnel, and the actual tkinter HUD rendering on
the Moonlight screen are only confirmed on the real setup.

## Button-panel feedback that ships with this (same branch)

- active-preset banner + progress bar on the laptop window (from the app's `preset` / `preset_progress` status)
- ack / refusal toast from the app's replies (green ack, red refusal)
- a red "REAL ARM NOT FOLLOWING" banner when the follower has stopped polling (status `real_following`: `None` means
  no follower has ever been seen, i.e. pure sim; `false` means it was seen and has gone quiet)
- preset **BUSY** rule: a second preset or voice motion while one move is still running is refused ("busy ... press
  STOP first") instead of chain-replacing the move. STOP (an utterance) always gets through and clears the move. The
  rule lives in the shared `do_preset` / jog path, so it is the same for the button panel and for voice.
