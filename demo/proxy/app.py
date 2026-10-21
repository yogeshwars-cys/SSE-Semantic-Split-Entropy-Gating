"""Laptop B: a plain caching reverse proxy. No gating, no SSE -- that lives in the backend now.

    python demo/proxy/app.py --origin http://<laptopA>:8080

Clients browse through :8000; the cache dashboard is on :8001. The proxy just caches what the origin returns
(obeying Cache-Control and an optional forced TTL) and shows telemetry: the cache grid, hit ratio, and an
audience-only "poisoned" flag so you can watch entries go red (SSE off at the origin) or stay clean (SSE on).
It has no authority over whether a response is poisoned -- that is decided upstream in the backend."""
import argparse
import http.client
import os
import sys
import threading
import time
from collections import Counter
from urllib.parse import parse_qsl, urlencode, urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from demo.common import web  # noqa: E402

UNKEYED = ("utm_", "fbclid", "gclid")
HOP = {"connection", "keep-alive", "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade",
       "content-length"}
# Audience-only sniff of what the attacks plant; the cache never uses it to decide anything.
TRUTH = (b"evil.example", b"not supported")


class Entry:
    def __init__(self, status, ctype, body, ttl):
        self.status, self.ctype, self.body, self.ttl = status, ctype, body, ttl
        self.stored, self.hits = time.time(), 0

    def fresh(self):
        return time.time() - self.stored < self.ttl

    def view(self, key):
        return {"key": key, "status": self.status, "age": round(time.time() - self.stored, 1), "ttl": self.ttl,
                "hits": self.hits, "size": len(self.body),
                "poisoned": self.status >= 400 or any(m in self.body for m in TRUTH)}


def keyed_path(path):
    p, _, q = path.partition("?")
    keyed = sorted((k, v) for k, v in parse_qsl(q, keep_blank_values=True) if not k.startswith(UNKEYED))
    return p + ("?" + urlencode(keyed) if keyed else "")


def cache_key(host, path):
    return host + "|" + keyed_path(path)


def max_age(cc):
    for part in (cc or "").split(","):
        k, _, v = part.strip().partition("=")
        if k == "max-age" and v.isdigit():
            return int(v)
    return 0


class Proxy:
    def __init__(self, args):
        self.origin = urlsplit(args.origin)
        self.ttl = args.ttl
        self.cache, self.lock = {}, threading.Lock()
        self.stats, self.seq = Counter(), 0
        self.hub = web.Hub()
        self.rps, self._last = 0.0, (time.time(), 0)
        threading.Thread(target=self._tick, daemon=True).start()

    def fetch(self, path, headers):
        h = {k: v for k, v in headers.items() if k not in HOP}
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

    def handle(self, req):
        t0 = time.time()
        path = req.path if req.path.startswith("/") else urlsplit(req.path).path or "/"
        headers = {k.lower(): v for k, v in req.headers.items()}
        host = (headers.get("host") or self.origin.netloc).split(":")[0]
        key = cache_key(host, path)
        self.stats["requests"] += 1
        e = self._lookup(key)
        if e is not None:
            e.hits += 1
            self.stats["hits"] += 1
            status, ctype, body, tag = e.status, e.ctype, e.body, "HIT"
        else:
            self.stats["misses"] += 1
            s, ct, cc, body = self.fetch(path, headers)
            ttl = self.ttl or max_age(cc)
            if ttl > 0 and "no-store" not in cc and "private" not in cc:
                with self.lock:
                    self.cache[key] = Entry(s, ct, body, ttl)
                self.stats["fills"] += 1
                tag = "MISS"
            else:
                tag = "MISS(no-store)"
            status, ctype = s, ct
        req.send_response(status)
        req.send_header("Content-Type", ctype)
        req.send_header("Content-Length", str(len(body)))
        req.send_header("X-Cache", tag)
        req.end_headers()
        req.wfile.write(body)
        self._emit({"kind": "request", "t": time.time(), "host": host, "path": path, "key": key,
                    "cache": tag, "status": status, "ms": round((time.time() - t0) * 1000, 1),
                    "poisoned": status >= 400 or any(m in body for m in TRUTH)})

    def _lookup(self, key):
        with self.lock:
            e = self.cache.get(key)
            if e is not None and not e.fresh():
                del self.cache[key]
                e = None
        return e

    def state(self):
        with self.lock:
            entries = [e.view(k) for k, e in self.cache.items() if e.fresh()]
        s = self.stats
        served = s["hits"] + s["misses"]
        return {"kind": "state", "t": time.time(), "stats": dict(s), "rps": self.rps,
                "hit_ratio": round(s["hits"] / served, 3) if served else None,
                "cache": sorted(entries, key=lambda x: x["key"]), "poisoned_entries": sum(e["poisoned"] for e in entries),
                "origin": f"{self.origin.scheme}://{self.origin.netloc}", "ttl": self.ttl}

    def purge(self):
        with self.lock:
            self.cache.clear()
        self._emit({"kind": "note", "t": time.time(), "text": "Cache purged"})

    def _emit(self, ev):
        with self.lock:
            self.seq += 1
            ev["id"] = self.seq
        self.hub.publish(ev)

    def _tick(self):
        while True:
            time.sleep(1)
            now, n = time.time(), self.stats["requests"]
            t0, n0 = self._last
            self.rps, self._last = round((n - n0) / (now - t0), 1), (now, n)
            self.hub.publish(self.state())


def proxy_handler(px):
    class H(web.Handler):
        def do_GET(self):
            try:
                px.handle(self)
            except OSError as e:
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
            p = urlsplit(self.path).path
            if p == "/api/purge":
                px.purge()
            elif p == "/api/reset":
                px.purge()
                px.stats.clear()
            else:
                return self.send_error(404)
            self.send_json(px.state())
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origin", default="http://127.0.0.1:8080", help="the backend laptop's origin server")
    ap.add_argument("--port", type=int, default=8000, help="proxy port (dashboard on port + 1)")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--ttl", type=int, default=0, help="force a cache TTL in seconds (0: obey the origin's max-age)")
    a = ap.parse_args()
    px = Proxy(a)
    web.serve(proxy_handler(px), a.bind, a.port)
    web.serve(dash_handler(px), a.bind, a.port + 1)
    ip = web.lan_ip()
    print(f"proxy      http://{ip}:{a.port}   (point the client laptop here)")
    print(f"dashboard  http://{ip}:{a.port + 1}")
    print(f"origin     {a.origin}   (SSE on/off lives on the origin's visualiser, not here)", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
