"""VM-screen HUD overlay (shows up in Moonlight next to the Isaac view): skeleton PiP, clutch, sync state,
warm-up prompts and latency. Fed by the app over UDP (net.hud_port, default 47130), 10 Hz.

    DISPLAY=:0 python3 scripts/hud.py [--port 47130] [--x 1350 --y 60]

Uses only the standard library (tkinter). The skeleton is drawn as a MIRROR view (your right hand on the right),
the same way you see the arm from behind in Isaac. The tracked (right) arm is thick orange.
"""
import argparse
import json
import queue
import socket
import threading
import time
import tkinter as tk

W, H = 560, 600
C = {"following": "#1f9d4c", "calibrating": "#2f6fd6", "idle": "#555555",
     "ok": "#22c55e", "clamped": "#f59e0b", "lost": "#ef4444", "calib": "#3b82f6", None: "#777777"}
ARM_R, ARM_L = [(1, 3), (3, 5)], [(0, 2), (2, 4)]           # indices into skel2d: 11,12,13,14,15,16,23,24,0
TORSO = [(0, 1), (0, 6), (1, 7), (6, 7)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=47130)
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

    root = tk.Tk()
    root.title("rover teleop HUD")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.geometry(f"{W}x{H}+{args.x}+{args.y}")
    cv = tk.Canvas(root, width=W, height=H, bg="#111418", highlightthickness=0)
    cv.pack()
    state = {"msg": None, "t": 0.0}

    def draw():
        while not q.empty():
            state["msg"], state["t"] = q.get(), time.time()
        m, age = state["msg"], time.time() - state["t"]
        cv.delete("all")
        if m is None or age > 2.0:
            cv.create_text(W / 2, H / 2, text="no data from the app\n(is rover_intent.app running?)",
                           fill="#aaaaaa", font=("Helvetica", 16), justify="center")
            root.after(100, draw)
            return
        cl = m.get("clutch") or "idle"
        banner = {"idle": "FROZEN  -  F / open palm up 1 s / raise LEFT hand",
                  "calibrating": f"HOLD STILL  {int(100 * (m.get('calib') or 0))}%",
                  "following": f"FOLLOWING  ({m.get('trigger')})"}.get(cl, cl)
        cv.create_rectangle(0, 0, W, 44, fill=C.get(cl, "#555555"), width=0)
        cv.create_text(12, 22, text=banner, anchor="w", fill="white", font=("Helvetica", 15, "bold"))
        if cl == "calibrating":
            cv.create_rectangle(0, 40, W * (m.get("calib") or 0), 44, fill="white", width=0)
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
        # warm-up
        co = m.get("coach")
        if co:
            cv.create_text(W / 2, 410, text=co["arrow"], fill="#00e5ff", font=("Helvetica", 44, "bold"))
            cv.create_text(W / 2, 462, text=co["text"], fill="#00e5ff", font=("Helvetica", 16, "bold"))
            done = "  ".join(("✔" if v else "✘") + k for k, v in co["results"].items())
            cv.create_text(W / 2, 492, text=f"warm-up {co['n']}/{co['of']}  ·  {co['left_s']:.0f}s left   {done}",
                           fill="#8892a0", font=("Helvetica", 11))
        elif m.get("coach_msg"):
            cv.create_text(W / 2, 440, text=m["coach_msg"], fill="#00e5ff", font=("Helvetica", 14, "bold"))
        else:
            cv.create_text(W / 2, 440, text="press W in the webcam window for the 60 s warm-up",
                           fill="#5b6472", font=("Helvetica", 12))
        # latency / link
        rtt = m.get("rtt_ms")
        rtt_txt = f"{rtt:.0f} ms" if rtt is not None else "?"
        loss = 100 * (m.get("loss") or 0)
        col = "#ef4444" if (rtt or 0) > 250 or loss > 20 else "#f59e0b" if (rtt or 0) > 120 or loss > 5 else "#22c55e"
        cv.create_text(20, 540, anchor="w", fill=col, font=("Helvetica", 13, "bold"),
                       text=f"round trip {rtt_txt}   loss {loss:.0f}%   {m.get('pkt_s', 0)} pkt/s")
        ba = m.get("body_age")
        cv.create_text(20, 566, anchor="w", fill="#8892a0", font=("Helvetica", 11),
                       text=f"last pose {1000 * ba:.0f} ms ago" if ba is not None else "no pose yet")
        if m.get("say"):
            cv.create_text(W - 20, 566, anchor="e", fill="#8892a0", font=("Helvetica", 11),
                           text=f"robot: {m['say']}"[:60])
        root.after(100, draw)

    draw()
    root.mainloop()


if __name__ == "__main__":
    main()
