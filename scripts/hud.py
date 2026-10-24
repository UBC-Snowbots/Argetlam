"""VM-screen HUD overlay (shows up in Moonlight next to the Isaac view): skeleton PiP, clutch, sync state,
warm-up prompts and latency. Fed by the app over UDP (net.hud_port, default 47130), 10 Hz.

    DISPLAY=:0 python3 scripts/hud.py [--port 47130] [--x 1350 --y 60]

The skeleton is drawn as a MIRROR view (your right hand on the right), the same way you see the arm from behind in
Isaac. The tracked (right) arm is thick orange. Directly BELOW the skeleton panel is the live webcam stream: the
laptop body client sends a downscaled JPEG (CameraFrame) which the app forwards to net.hud_video_port; this window
decodes the latest frame (PIL) and shows it, so the operator does not need the separate OpenCV window on the laptop.

Needs tkinter (stdlib) + Pillow (PIL) for the JPEG decode; the video panel degrades to "video needs Pillow" if PIL
is missing, and the rest of the HUD still works.
"""
import argparse
import base64
import io
import json
import queue
import socket
import threading
import time
import tkinter as tk

try:
    from PIL import Image, ImageTk
except Exception:  # Pillow missing: the HUD still runs, the video panel shows a hint
    Image = ImageTk = None

W, HUD_H = 560, 600                     # HUD_H = the skeleton/sync/latency panel height (unchanged)
VBOX = (20, HUD_H + 14, W - 20, HUD_H + 14 + 300)   # live-video panel box, directly below the HUD panel
H = VBOX[3] + 24
C = {"following": "#1f9d4c", "calibrating": "#2f6fd6", "idle": "#555555",
     "ok": "#22c55e", "clamped": "#f59e0b", "lost": "#ef4444", "calib": "#3b82f6", None: "#777777"}
ARM_R, ARM_L = [(1, 3), (3, 5)], [(0, 2), (2, 4)]           # indices into skel2d: 11,12,13,14,15,16,23,24,0
TORSO = [(0, 1), (0, 6), (1, 7), (6, 7)]
VIDEO_STALE_S = 1.0


def video_status(age, fps):
    """Status line for the video panel: (text, is_stale). age = seconds since the last frame (None = none yet)."""
    if age is None:
        return "no camera", False
    stale = age > VIDEO_STALE_S
    return (f"stale {age:.1f}s" if stale else f"{fps} fps"), stale


def _strip0(s):
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def reach_status_line(m):
    """Compact plane-tuning line, e.g. 'REACH >> | spd4.0 lat.20 vrt.15 db.015'. '' when not in plane mode."""
    pt = m.get("plane_tune")
    if not pt:
        return ""
    arrow = {">>": ">>", "<<": "<<", "hold": "--"}.get(pt.get("reach_state"), "--")
    lat = _strip0("{:.2f}".format(pt["lateral"]))
    vrt = _strip0("{:.2f}".format(pt["vertical"]))
    db = _strip0("{:.3f}".format(pt["deadband"]))
    return "REACH {} | spd{:.1f} lat{} vrt{} db{}".format(arrow, pt["max_speed"] * 100, lat, vrt, db)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=47130)
    ap.add_argument("--video-port", type=int, default=47131, help="UDP port for the forwarded webcam stream")
    ap.add_argument("--x", type=int, default=1350)
    ap.add_argument("--y", type=int, default=60)
    args = ap.parse_args()
    q = queue.Queue()

    def rx():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", args.port))
        while True:
            try:
                q.put(json.loads(s.recvfrom(65536)[0]))
            except ValueError:
                pass
    threading.Thread(target=rx, daemon=True).start()

    # Live video: its own socket + latest-frame holder (keep ONLY the newest frame; older ones are dropped).
    vlock = threading.Lock()
    vstate = {"jpeg": None, "w": 0, "h": 0, "t": 0.0, "seq": -1}
    vfps = []

    def vrx():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", args.video_port))
        while True:
            try:
                data = s.recvfrom(200000)[0]
                mm = json.loads(data)
            except (ValueError, OSError):
                continue
            if mm.get("kind") != "camera_frame":
                continue
            try:
                raw = base64.b64decode(mm["jpeg_b64"])
            except Exception:
                continue
            now = time.time()
            with vlock:
                vstate.update(jpeg=raw, w=mm.get("width", 0), h=mm.get("height", 0), t=now, seq=mm.get("seq", -1))
                vfps[:] = [x for x in vfps if now - x <= 1.0] + [now]
    threading.Thread(target=vrx, daemon=True).start()

    root = tk.Tk()
    root.title("rover teleop HUD")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.geometry(f"{W}x{H}+{args.x}+{args.y}")
    cv = tk.Canvas(root, width=W, height=H, bg="#111418", highlightthickness=0)
    cv.pack()
    state = {"msg": None, "t": 0.0, "vphoto": None, "vseq": -2}

    def draw_video():
        """Live webcam panel directly below the HUD. Decodes the latest frame only; shows age when stale (> 1 s)."""
        x0, y0, x1, y1 = VBOX
        cv.create_rectangle(x0, y0, x1, y1, outline="#333a44")
        cv.create_text(x0 + 6, y0 + 10, text="camera (laptop)", anchor="w", fill="#8892a0", font=("Helvetica", 10))
        with vlock:
            jpeg, vt, vseq, fps = vstate["jpeg"], vstate["t"], vstate["seq"], len(vfps)
        age = time.time() - vt if vt else None
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + 8
        if Image is None:
            cv.create_text(cx, cy, text="video needs Pillow\n(pip install pillow)", fill="#8892a0",
                           font=("Helvetica", 13), justify="center")
            return
        if jpeg is None:
            cv.create_text(cx, cy, text="no camera\n(start the body client with --stream-video)", fill="#8892a0",
                           font=("Helvetica", 13), justify="center")
            return
        if state["vseq"] != vseq:  # decode only when a new frame arrived
            try:
                im = Image.open(io.BytesIO(jpeg)); im.load()
                bw, bh = (x1 - x0 - 12), (y1 - y0 - 34)
                scale = min(bw / im.width, bh / im.height)
                im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
                state["vphoto"], state["vseq"] = ImageTk.PhotoImage(im), vseq
            except Exception:
                state["vphoto"] = None
        if state["vphoto"] is not None:
            cv.create_image(cx, cy, image=state["vphoto"])
        info, stale = video_status(age, fps)
        cv.create_text(x1 - 6, y1 - 10, text=info, anchor="e",
                       fill="#ef4444" if stale else "#22c55e", font=("Helvetica", 11, "bold"))

    def draw():
        while not q.empty():
            state["msg"], state["t"] = q.get(), time.time()
        m, age = state["msg"], time.time() - state["t"]
        cv.delete("all")
        if m is None or age > 2.0:
            cv.create_text(W / 2, HUD_H / 2, text="no data from the app\n(is rover_intent.app running?)",
                           fill="#aaaaaa", font=("Helvetica", 16), justify="center")
            draw_video()
            root.after(100, draw)
            return
        cl = m.get("clutch") or "idle"
        why = f" ({m['release_reason']})" if m.get("release_reason") else ""
        banner = {"idle": f"FROZEN{why}  -  F / open palm up 1 s / raise LEFT hand",
                  "calibrating": f"RE-CENTRE: hold still  {int(100 * (m.get('clutch_progress') or 0))}%",
                  "following": f"FOLLOWING  ({m.get('trigger')})"}.get(cl, cl)
        cv.create_rectangle(0, 0, W, 44, fill=C.get(cl, "#555555"), width=0)
        cv.create_text(12, 22, text=banner, anchor="w", fill="white", font=("Helvetica", 15, "bold"))
        if cl == "calibrating":
            cv.create_rectangle(0, 40, W * (m.get("clutch_progress") or 0), 44, fill="white", width=0)
        # skeleton PiP (mirror view)
        box = (20, 60, W - 20, 330)
        cv.create_rectangle(*box, outline="#333a44")
        cv.create_text(box[0] + 6, box[1] + 10, text="you (mirror view)", anchor="w", fill="#8892a0",
                       font=("Helvetica", 10))
        sk = m.get("skel2d")
        if sk:
            bw, bh = box[2] - box[0], box[3] - box[1]
            pt = [(box[0] + (1 - x) * bw, box[1] + y * bh) for x, y in sk]  # mirror: flip x
            for a, b in TORSO:
                cv.create_line(*pt[a], *pt[b], fill="#9aa4b2", width=3)
            for a, b in ARM_L:
                cv.create_line(*pt[a], *pt[b], fill="#9aa4b2", width=3)
            for a, b in ARM_R:
                cv.create_line(*pt[a], *pt[b], fill="#ff9f1a", width=7)
            hx, hy = pt[5]
            ho = m.get("hand_open")
            cv.create_oval(hx - 12, hy - 12, hx + 12, hy + 12, outline="#ff9f1a", width=3,
                           fill="#ff9f1a" if ho is not None and ho < 0.35 else "")
            nx, ny = pt[8]
            cv.create_oval(nx - 14, ny - 14, nx + 14, ny + 14, outline="#9aa4b2", width=3)
        else:
            cv.create_text((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, text="no skeleton (step into view)",
                           fill="#8892a0", font=("Helvetica", 13))
        # sync state
        g = m.get("ghost")
        gtxt = {"ok": "tracking", "clamped": "clamped: out of reach", "lost": "tracking LOST - arm holding",
                "calib": "calibrating"}.get(g, "no target")
        cv.create_oval(24, 346, 44, 366, fill=C.get(g, "#777777"), width=0)
        cv.create_text(54, 356, text=f"ghost: {gtxt}", anchor="w", fill="white", font=("Helvetica", 13))
        grip = m.get("grip") or 0
        cv.create_text(W - 20, 356, text="gripper CLOSED" if grip >= 0.5 else "gripper open", anchor="e",
                       fill="#ffd166" if grip >= 0.5 else "#8892a0", font=("Helvetica", 13))
        # plane live-tuning: compact knob line (reach state + gains/thresholds)
        rl = reach_status_line(m)
        if rl:
            rs = (m.get("plane_tune") or {}).get("reach_state")
            cv.create_text(W / 2, 382, text=rl, fill="#22c55e" if rs in (">>", "<<") else "#9aa4b2",
                           font=("Helvetica", 12, "bold"))
        tune = m.get("tune")   # guided tune-check prompt
        if tune and tune.get("active"):
            cv.create_text(W / 2, 410, text=f"TUNE CHECK\n{tune.get('prompt', '')}", fill="#00e5ff",
                           font=("Helvetica", 15, "bold"), width=W - 30, justify="center")
            if tune.get("progress") is not None:
                cv.create_rectangle(40, 445, 40 + (W - 80) * tune["progress"], 455, fill="#00e5ff", width=0)
        else:
            # body teleop mapping (plane or arm_mirror)
            cv.create_text(W / 2, 420, text=f"mapping: {m.get('mapping', '?')}",
                           fill="#5b6472", font=("Helvetica", 12))
        if m.get("assist"):
            cv.create_text(W / 2, 480, text="ASSIST: grabbing / releasing...", fill="#ff4fd8", font=("Helvetica", 14, "bold"))
        elif m.get("grab_ready"):
            cv.create_text(W / 2, 480, text=f"make a FIST to grab the {m['grab_ready']}", fill="#ff4fd8",
                           font=("Helvetica", 14, "bold"))
        cv.create_text(W / 2, 505, text="white cross = neutral (box centre)   magenta = grab / touch target",
                       fill="#5b6472", font=("Helvetica", 10))
        # latency / link
        rtt = m.get("rtt_ms")
        rtt_txt = f"{rtt:.0f} ms" if rtt is not None else "?"
        loss = 100 * (m.get("loss") or 0)
        col = "#ef4444" if (rtt or 0) > 250 or loss > 20 else "#f59e0b" if (rtt or 0) > 120 or loss > 5 else "#22c55e"
        cv.create_text(20, 540, anchor="w", fill=col, font=("Helvetica", 13, "bold"),
                       text=f"round trip {rtt_txt}   loss {loss:.0f}%   {m.get('pkt_s', 0)} pkt/s")
        if m.get("vis_pct") is not None:
            v = m["vis_pct"]
            cv.create_text(W - 20, 540, anchor="e", font=("Helvetica", 12, "bold"),
                           fill="#22c55e" if v >= 90 else "#f59e0b" if v >= 60 else "#ef4444",
                           text=f"arm visible {v}%  guide {m.get('inpos_pct')}%  hand {m.get('hand_pct')}%")
        ba = m.get("body_age")
        cv.create_text(20, 566, anchor="w", fill="#8892a0", font=("Helvetica", 11),
                       text=f"last pose {1000 * ba:.0f} ms ago" if ba is not None else "no pose yet")
        if m.get("say"):
            cv.create_text(W - 20, 566, anchor="e", fill="#8892a0", font=("Helvetica", 11),
                           text=f"robot: {m['say']}"[:60])
        draw_video()
        root.after(100, draw)

    draw()
    root.mainloop()


if __name__ == "__main__":
    main()
