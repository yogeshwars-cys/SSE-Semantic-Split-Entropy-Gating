"""web.py: the small HTTP layer the three demo apps share (stdlib only).

Hub fans events out to dashboards over EventSource; Handler serves the app's static files plus the shared ones;
serve() runs a ThreadingHTTPServer in a daemon thread."""
import json
import mimetypes
import os
import queue
import socket
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

SHARED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class Hub:
    """Broadcast JSON events to every open dashboard; a new dashboard gets the recent backlog first."""

    def __init__(self, keep=300):
        self.subs, self.lock, self.recent = [], threading.Lock(), deque(maxlen=keep)

    def publish(self, ev):
        data = json.dumps(ev, separators=(",", ":"))
        with self.lock:
            if ev.get("kind") != "state":                       # state is re-sent every second, no need to replay it
                self.recent.append(data)
            subs = list(self.subs)
        for q in subs:
            try:
                q.put_nowait(data)
            except queue.Full:
                pass

    def stream(self, h):
        q = queue.Queue(maxsize=2000)
        with self.lock:
            self.subs.append(q)
            backlog = list(self.recent)
        try:
            h.send_response(200)
            h.send_header("Content-Type", "text/event-stream")
            h.send_header("Cache-Control", "no-cache")
            h.send_header("Access-Control-Allow-Origin", "*")
            h.end_headers()
            for d in backlog:
                h.wfile.write(f"data: {d}\n\n".encode())
            h.wfile.flush()
            while True:
                try:
                    h.wfile.write(f"data: {q.get(timeout=15)}\n\n".encode())
                except queue.Empty:
                    h.wfile.write(b": ping\n\n")
                h.wfile.flush()
        except OSError:                                         # dashboard closed the tab
            pass
        finally:
            with self.lock:
                self.subs.remove(q)


class Handler(BaseHTTPRequestHandler):
    """Base handler: JSON helpers and static files. `/` -> app index, `/static/x` -> app, `/shared/x` -> common."""
    static_dir = None
    server_version = "sse-demo"

    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def send_static(self):
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            root, rel = self.static_dir, "index.html"
        elif path.startswith("/static/"):
            root, rel = self.static_dir, path[len("/static/"):]
        elif path.startswith("/shared/"):
            root, rel = SHARED, path[len("/shared/"):]
        else:
            return False
        full = os.path.normpath(os.path.join(root, rel))
        if not full.startswith(os.path.normpath(root)) or not os.path.isfile(full):
            self.send_error(404)
            return True
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", (mimetypes.guess_type(full)[0] or "application/octet-stream") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        return True


def serve(handler, host, port):
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def lan_ip():
    """This laptop's address on the LAN (no packet is sent), so the banner shows what the other laptops dial."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def post_json(url, obj, timeout=2.0):
    """Fire one JSON POST (stdlib); raises OSError when the other laptop is unreachable."""
    import http.client
    u = urlsplit(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    try:
        body = json.dumps(obj, separators=(",", ":")).encode()
        conn.request("POST", u.path or "/", body, {"Content-Type": "application/json"})
        conn.getresponse().read()
    finally:
        conn.close()
