"""VM: TCP <-> UDP relay so laptop clients can reach the app through an SSH tunnel (SSH can't forward UDP).

    python scripts/tcp_relay.py            # listens on 127.0.0.1:47101... see --port
Laptop:
    ssh -N -L 47120:127.0.0.1:47120 your-server        (leave running)
    python body_client.py --tunnel 127.0.0.1:47120

Protocol: newline-delimited JSON both ways. Each TCP client gets its own UDP socket, so the app's replies and 10 Hz
status messages (sent to the UDP source address) come back to the right client.
"""
import argparse
import selectors
import socket
import time


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=47120, help="TCP port (bind 127.0.0.1: reachable only via SSH)")
    ap.add_argument("--app", default="127.0.0.1:47100")
    args = ap.parse_args()
    app_host, app_port = args.app.split(":")
    app = (app_host, int(app_port))
    sel = selectors.DefaultSelector()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen()
    srv.setblocking(False)
    sel.register(srv, selectors.EVENT_READ, ("accept", None))
    print(f"[relay] tcp 127.0.0.1:{args.port} <-> udp {app}", flush=True)
    peers = {}  # tcp socket -> (udp socket, buffer)
    while True:
        for key, _ in sel.select():
            kind, other = key.data
            if kind == "accept":
                c, addr = srv.accept()
                c.setblocking(False)
                u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                u.setblocking(False)
                peers[c] = [u, b""]
                sel.register(c, selectors.EVENT_READ, ("tcp", u))
                sel.register(u, selectors.EVENT_READ, ("udp", c))
                print(f"{_ts()} [relay] client connected {addr}", flush=True)
            elif kind == "tcp":
                c = key.fileobj
                try:
                    data = c.recv(65536)
                except BlockingIOError:
                    continue
                except ConnectionResetError:
                    data = b""
                if not data:
                    u = peers.pop(c)[0]
                    sel.unregister(c); sel.unregister(u); c.close(); u.close()
                    print(f"{_ts()} [relay] client disconnected", flush=True)
                    continue
                buf = peers[c][1] + data
                *lines, peers[c][1] = buf.split(b"\n")
                for line in lines:
                    if line.strip():
                        peers[c][0].sendto(line, app)
            elif kind == "udp":
                u, c = key.fileobj, other
                try:
                    while True:
                        c.sendall(u.recv(65536) + b"\n")
                except BlockingIOError:
                    pass
                except (BrokenPipeError, ConnectionResetError):
                    pass


if __name__ == "__main__":
    main()
