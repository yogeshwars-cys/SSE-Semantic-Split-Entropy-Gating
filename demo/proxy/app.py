"""Laptop 2: the caching reverse proxy, with SSE in front of every cache fill.

    python demo/proxy/app.py --origin http://<laptop3>:8080 --viz http://<laptop3>:8090

Clients browse through :8000; the cache dashboard is on :8001. Every decision is also streamed to the SSE
visualiser on laptop 3.

Modes (switch live from the dashboard):
  vulnerable  plain shared cache: whatever the first request on a key brings back is served to everyone
  gates       statistical + semantic gate: a flagged request is answered but never fills the cache
  split       split check: each fill is compared with a clean origin copy (Part B); on a mismatch the clean copy is
              cached instead. Requests for a key under verification are held until it finishes
  sse         gates + split + calibration: a split mismatch blames the reflected header and tightens the gates; a
              sample of gate alarms is audited the same way, and false alarms loosen them
"""
import argparse
import hashlib
import http.client
import os
import queue
import random
import sys
import threading
import time
from collections import Counter
from urllib.parse import parse_qsl, urlencode, urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from demo.common import traffic, web  # noqa: E402
from demo.proxy.engine import Engine  # noqa: E402

MODES = ["vulnerable", "gates", "split", "sse"]
UNKEYED = ("utm_", "fbclid", "gclid")
HOP = {"connection", "keep-alive", "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade",
       "content-length"}
# Audience-only check of what the attacks plant; the cache never uses it to decide anything.
TRUTH = (b"evil.example", b"not supported")


class Entry:
    def __init__(self, status, ctype, body, ttl, state, fired):
        self.status, self.ctype, self.body, self.ttl, self.state, self.fired = status, ctype, body, ttl, state, fired
        self.stored, self.hits = time.time(), 0

    def fresh(self):
        return time.time() - self.stored < self.ttl

    def view(self, key):
        return {"key": key, "status": self.status, "state": self.state, "age": round(time.time() - self.stored, 1),
                "ttl": self.ttl, "hits": self.hits, "size": len(self.body), "fired": self.fired,
                "poisoned": self.status >= 400 or any(m in self.body for m in TRUTH)}


def cache_key(path):
    p, _, q = path.partition("?")
    keyed = sorted((k, v) for k, v in parse_qsl(q, keep_blank_values=True) if not k.startswith(UNKEYED))
    return p + ("?" + urlencode(keyed) if keyed else "")


def max_age(cc):
    for part in (cc or "").split(","):
        k, _, v = part.strip().partition("=")
        if k == "max-age" and v.isdigit():
            return int(v)
    return 0


class Proxy:
    def __init__(self, args):
        self.origin = urlsplit(args.origin)
        self.viz = args.viz.rstrip("/") if args.viz else None
        self.mode, self.ttl, self.hold_ms, self.audit = args.mode, args.ttl, args.hold_ms, args.audit
        print(f"fitting the SSE gates on {args.train} benign requests ...", flush=True)
        self.engine = Engine(traffic.training_set(args.train))
        print(f"  done in {self.engine.fit_s}s (gate threshold {self.engine.gate.thr:.2f}, "
              f"semantic threshold {self.engine.sem.thr:.2f})", flush=True)
        self.cache, self.pending, self.lock = {}, {}, threading.Lock()
        self.stats, self.seq = Counter(), 0
        self.hub, self.out = web.Hub(), queue.Queue(maxsize=4000)
        self.viz_ok, self.rps, self._last = None, 0.0, (time.time(), 0)
        threading.Thread(target=self._forward, daemon=True).start()
        threading.Thread(target=self._tick, daemon=True).start()

    # ---- origin -------------------------------------------------------------------------------------------------
    def fetch(self, path, headers, part=None):
        h = {k: v for k, v in headers.items() if k not in HOP}
        if part:
            h["x-sse-part"] = part                              # lets the origin log show split's Part A / Part B
        conn = http.client.HTTPConnection(self.origin.hostname, self.origin.port or 80, timeout=5)
        try:
            conn.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
            for k, v in h.items():
                conn.putheader(k, v)
            conn.endheaders()
            r = conn.getresponse()
            body = r.read()
            self.stats["origin_fetches"] += 1
            return r.status, r.getheader("Content-Type", "text/html"), r.getheader("Cache-Control", ""), body
        finally:
            conn.close()

    def clean_headers(self, headers):
        return dict(traffic.CLEAN, host=headers.get("host", self.origin.netloc))

    def store(self, key, status, ctype, cc, body, state, fired):
        ttl = self.ttl or max_age(cc)
        if ttl > 0 and "no-store" not in cc and "private" not in cc:
            with self.lock:
                self.cache[key] = Entry(status, ctype, body, ttl, state, fired)
            self.stats["fills"] += 1

    # ---- request path -------------------------------------------------------------------------------------------
    def handle(self, req):
        t0 = time.time()
        path = req.path if req.path.startswith("/") else urlsplit(req.path).path or "/"
        headers = {k.lower(): v for k, v in req.headers.items()}
        key, rt = cache_key(path), path.split("?", 1)[0]
        pnames = traffic.param_names(path)
        qpairs = parse_qsl(path.partition("?")[2], keep_blank_values=True)
        mode = self.mode
        insp = self.engine.inspect(headers, pnames, rt)          # always scored, so the visualiser shows it in every mode
        enforced = mode in ("gates", "sse")
        ev = {"kind": "request", "path": path, "key": key, "route": rt, "mode": mode, "insp": insp["view"],
              "flagged": insp["flagged"], "enforced": enforced,
              "fired": {"gate": insp["og"], "sem": insp["os"]}, "split": None, "learn": []}
        self.stats["requests"] += 1

        status = ctype = body = None
        held = False
        e = self._lookup(key)
        if e is None and mode in ("split", "sse"):
            with self.lock:
                ev_wait = self.pending.get(key)
            if ev_wait is not None:                             # a fill for this key is being verified: hold the view
                held = True
                self.stats["held"] += 1
                ev_wait.wait(5)
                e = self._lookup(key)
        if e is not None:
            e.hits += 1
            self.stats["hits"] += 1
            status, ctype, body = e.status, e.ctype, e.body
            ev["cache"] = "HELD" if held else "HIT"
            ev["verdict"] = "Held, then served the verified copy" if held else "Served from cache"
        else:
            self.stats["misses"] += 1
            status, ctype, body = self._miss(mode, path, key, headers, insp, qpairs, ev)
        self._respond(req, status, ctype, body, ev["cache"], mode)
        ev.update(status=status, ms=round((time.time() - t0) * 1000, 1), t=time.time())
        self.emit(ev)

    def _lookup(self, key):
        with self.lock:
            e = self.cache.get(key)
            if e is not None and not e.fresh():
                del self.cache[key]
                e = None
        return e

    def _miss(self, mode, path, key, headers, insp, qpairs, ev):
        fired = sorted(set(insp["og"]) | set(insp["os"]))
        if mode == "vulnerable" or (mode == "gates" and not insp["flagged"]):
            s, ct, cc, body = self.fetch(path, headers)
            self.store(key, s, ct, cc, body, "unchecked", fired)
            ev["cache"], ev["verdict"] = "MISS", "Filled the cache"
            return s, ct, body
        if mode in ("gates", "sse") and insp["flagged"]:
            s, ct, cc, body = self.fetch(path, headers)
            self.stats["blocked"] += 1
            ev["cache"], ev["verdict"] = "BYPASS", "Blocked at the gate: answered, not cached"
            if mode == "sse" and random.random() < self.audit:  # audit a sample of alarms against the clean copy
                self.stats["audits"] += 1
                bs, bct, bcc, bbody = self.fetch(key, self.clean_headers(headers), part="B")
                if (s, hashlib.sha256(body).digest()) == (bs, hashlib.sha256(bbody).digest()):
                    self.stats["false_alarms"] += 1
                    ev["learn"] = self.engine.learn_false_alarm(insp)
                    self.store(key, s, ct, cc, body, "verified", fired)
                    ev["split"] = {"match": True, "action": "false-alarm", "blamed": []}
                    ev["verdict"] = "False alarm: response matched the clean copy, gates loosened"
                else:
                    ev["split"] = {"match": False, "action": "confirmed", "blamed": []}
            return s, ct, body
        # split check: Part A is this request, Part B the clean reference; views of this key wait meanwhile
        with self.lock:
            gate = self.pending.setdefault(key, threading.Event())
        try:
            s, ct, cc, body = self.fetch(path, headers, part="A")
            if self.hold_ms:
                time.sleep(self.hold_ms / 1000)                 # late-check window, makes holds visible on stage
            bs, bct, bcc, bbody = self.fetch(key, self.clean_headers(headers), part="B")
            self.stats["split_checks"] += 1
            if (s, hashlib.sha256(body).digest()) == (bs, hashlib.sha256(bbody).digest()):
                self.store(key, s, ct, cc, body, "verified", fired)
                ev["split"] = {"match": True, "action": "verified", "blamed": []}
                ev["cache"], ev["verdict"] = "MISS", "Filled the cache with a verified copy"
            else:
                self.stats["split_caught"] += 1
                self.store(key, bs, bct, bcc, bbody, "healed", fired)
                blamed, changes = [], []
                if mode == "sse":
                    blamed, changes = self.engine.learn_miss(insp, body.decode("utf-8", "replace"),
                                                             bbody.decode("utf-8", "replace"), qpairs)
                    self.stats["tightened"] += sum(c["type"] == "tighten" for c in changes)
                ev["split"] = {"match": False, "action": "healed", "blamed": blamed}
                ev["learn"] = changes
                ev["cache"] = "MISS"
                ev["verdict"] = "Poison caught by split: clean copy cached" + (", gates tightened" if changes else "")
            return s, ct, body
        finally:
            with self.lock:
                self.pending.pop(key, None)
            gate.set()

    def _respond(self, req, status, ctype, body, xcache, mode):
        req.send_response(status)
        req.send_header("Content-Type", ctype)
        req.send_header("Content-Length", str(len(body)))
        req.send_header("X-Cache", xcache)
        req.send_header("X-SSE-Mode", mode)
        req.end_headers()
        req.wfile.write(body)

    # ---- events -------------------------------------------------------------------------------------------------
    def emit(self, ev):
        with self.lock:
            self.seq += 1
            ev["id"] = self.seq
        self.hub.publish(ev)
        if self.viz:
            try:
                self.out.put_nowait(ev)
            except queue.Full:
                pass

    def state(self):
        with self.lock:
            entries = [e.view(k) for k, e in self.cache.items() if e.fresh()]
        s = self.stats
        served = s["hits"] + s["misses"]
        return {"kind": "state", "t": time.time(), "mode": self.mode, "modes": MODES, "stats": dict(s),
                "hit_ratio": round(s["hits"] / served, 3) if served else None, "rps": self.rps,
                "cache": sorted(entries, key=lambda x: x["key"]), "engine": self.engine.state(),
                "origin": f"{self.origin.scheme}://{self.origin.netloc}", "viz": self.viz, "viz_ok": self.viz_ok,
                "settings": {"ttl": self.ttl, "hold_ms": self.hold_ms, "audit": self.audit}}

    def _tick(self):
        while True:
            time.sleep(1)
            now, n = time.time(), self.stats["requests"]
            t0, n0 = self._last
            self.rps, self._last = round((n - n0) / (now - t0), 1), (now, n)
            st = self.state()
            self.hub.publish(st)
            if self.viz:
                try:
                    self.out.put_nowait(st)
                except queue.Full:
                    pass

    def _forward(self):
        while True:
            batch = [self.out.get()]
            time.sleep(0.15)
            while len(batch) < 500:
                try:
                    batch.append(self.out.get_nowait())
                except queue.Empty:
                    break
            try:
                web.post_json(self.viz + "/ingest", batch)
                self.viz_ok = True
            except OSError:
                self.viz_ok = False

    def purge(self):
        with self.lock:
            self.cache.clear()
        self.emit({"kind": "note", "t": time.time(), "text": "Cache purged"})


def proxy_handler(px):
    class H(web.Handler):
        def do_GET(self):
            try:
                px.handle(self)
            except OSError as e:                                # origin down or client went away
                try:
                    self.send_error(502, f"origin unreachable: {e}")
                except OSError:
                    pass
    return H


def dash_handler(px):
    class H(web.Handler):
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

        def do_GET(self):
            p = urlsplit(self.path).path
            if p == "/api/state":
                self.send_json(px.state())
            elif p == "/api/stream":
                px.hub.stream(self)
            elif not self.send_static():
                self.send_error(404)

        def do_POST(self):
            p, body = urlsplit(self.path).path, self.read_json()
            if p == "/api/mode" and body.get("mode") in MODES:
                px.mode = body["mode"]
                px.emit({"kind": "note", "t": time.time(), "text": f"Mode set to {px.mode}"})
            elif p == "/api/purge":
                px.purge()
            elif p == "/api/reset":
                px.engine.reset()
                px.purge()
                px.stats.clear()
                px.emit({"kind": "note", "t": time.time(), "text": "Learning, cache and counters reset"})
            elif p == "/api/settings":
                for k, cast in (("ttl", int), ("hold_ms", int), ("audit", float)):
                    if k in body:
                        setattr(px, k, cast(body[k]))
            else:
                return self.send_error(404)
            self.send_json(px.state())
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origin", default="http://127.0.0.1:8080", help="the backend laptop's origin server")
    ap.add_argument("--viz", default="http://127.0.0.1:8090", help="the backend laptop's SSE visualiser ('' for none)")
    ap.add_argument("--mode", default="vulnerable", choices=MODES)
    ap.add_argument("--port", type=int, default=8000, help="proxy port (dashboard on port + 1)")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--ttl", type=int, default=0, help="force a cache TTL in seconds (0: use the origin's max-age)")
    ap.add_argument("--hold-ms", type=int, default=250, help="late-check window of the split check")
    ap.add_argument("--audit", type=float, default=0.3, help="share of gate alarms re-checked against a clean copy")
    ap.add_argument("--train", type=int, default=4000, help="benign requests the gates are fitted on")
    a = ap.parse_args()
    px = Proxy(a)
    web.serve(proxy_handler(px), a.bind, a.port)
    web.serve(dash_handler(px), a.bind, a.port + 1)
    ip = web.lan_ip()
    print(f"proxy      http://{ip}:{a.port}   (point the client laptop here)")
    print(f"dashboard  http://{ip}:{a.port + 1}")
    print(f"origin     {a.origin}   visualiser {a.viz or 'off'}   mode {a.mode}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
