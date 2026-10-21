"""Laptop A: the origin web server WITH SSE built in, plus the SSE visualiser.

    python demo/backend/app.py

Origin on :8080 (what the proxy fetches), visualiser on :8090 (open it here). SSE lives HERE, in the origin, and
is switched on/off at will from the visualiser (button) or `POST /api/sse {"on": true|false}`.

The origin serves five fake `.demo` sites plus a toy search engine (demo/common/sites.py). Each site reflects
unkeyed inputs into cacheable responses -- the cache-poisoning bugs (BENCHMARKS.md Part I):
  x-forwarded-host     -> host of the page's <script src>
  x-forwarded-prefix   -> the stylesheet's path prefix
  accept-language      -> written into <html lang="..."> unescaped
  utm_content          -> echoed into the share link (the cache leaves utm_* out of the key)
  user-agent           -> a legacy MSIE gets a "browser not supported" page
  cookie               -> over 4 KB is answered 400, and the error is cacheable (CPDoS)

With SSE OFF the origin reflects all of them, so the shared cache gets poisoned.
With SSE ON the origin runs the SSE gate on every request and SANITIZES: any reflected input the gate flags as
anomalous (plus the structural DoS payloads) is ignored, so the page -- and whatever the cache stores -- is clean.
Benign traffic scores below threshold and is reflected unchanged, so the cached page stays byte-identical.
The cache (laptop B) has no say in any of this; it just caches whatever the origin returns."""
import argparse
import html
import os
import sys
import threading
import time
from urllib.parse import parse_qsl, urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sse"))
from demo.common import sites, traffic, web  # noqa: E402
from anomaly_model import Gate  # noqa: E402
from semantic_model import Semantic  # noqa: E402

SKIP = {"host", "content-length", "x-sse-part", "connection", "keep-alive", "proxy-connection"}
# Header inputs the origin reflects (SSE sanitizes an anomalous one by dropping it so the default is used).
REFLECTORS = ("x-forwarded-host", "x-forwarded-prefix", "accept-language")


def _r(x):
    return None if x is None else round(float(x), 2)


class SSE:
    """The SSE gate, fitted on benign traffic, that the origin consults when it is switched on."""

    def __init__(self, train, fpr=0.01):
        t = time.time()
        self.gate, self.sem = Gate(fpr=fpr), Semantic(fpr=fpr)
        self.gate.fit(train)
        self.sem.fit(train)
        self.fit_s = round(time.time() - t, 2)
        self.lock = threading.RLock()

    def inspect(self, headers, pnames):
        h = {k: v for k, v in headers.items() if k not in SKIP}
        with self.lock:
            g, s = self.gate, self.sem
            zg = g.header_z(h)
            req = g.score(h, pnames)
            zs = s.header_z(h)
            flagged = {k for k, z in zg.items() if z > g.thr} | {k for k, z in zs.items() if z > s.thr}
            rows = [{"name": k, "value": v if len(v) <= 160 else v[:157] + "...", "len": len(v),
                     "zg": _r(zg.get(k)), "tg": _r(g.thr), "zs": _r(zs.get(k)), "ts": _r(s.thr) if k in zs else None,
                     "flag": k in flagged} for k, v in h.items()]
        return {"h": h, "pnames": pnames, "zg": zg, "zs": zs, "flagged": flagged,
                "req": {"z": _r(req), "t": _r(g.thr)}, "rows": rows}

    def learn(self, insp, sanitized):
        """Each sanitized poison teaches the gate (sharpen the blamed field), so the ticker shows SSE adapting."""
        h, zg, zs = insp["h"], insp["zg"], insp["zs"]
        changes = []
        with self.lock:
            for k in sanitized:
                if k in h:
                    w0 = self.gate.w.get(k, 1.0)
                    self.gate.learn(h, insp["pnames"], True, blamed=k)
                    changes.append({"type": "weight", "field": k, "old": _r(w0), "new": _r(self.gate.w.get(k, 1.0))})
            if zs:
                k = max(zs, key=zs.get)
                self.sem.learn(h, True, zs=zs)
                changes.append({"type": "sharpen", "field": k})
        return changes

    def state(self, on):
        with self.lock:
            g, s = self.gate, self.sem
            return {"sse_on": on, "gate_thr": _r(g.thr), "sem_thr": _r(s.thr), "baseline_n": g.base.N,
                    "fit_s": self.fit_s, "weights": {k: _r(v) for k, v in g.w.items()},
                    "boosted": {k: int((sp.gain > 1).sum()) for k, sp in s.space.items() if (sp.gain > 1).any()}}


def plan_sanitize(headers, path, flagged, sse_on):
    """What SSE neutralizes for this request. Returns (clean headers, clean query, [(field, reason)])."""
    h = dict(headers)
    q = dict(parse_qsl(path.partition("?")[2], keep_blank_values=True))
    done = []
    if not sse_on:
        return h, q, done
    for name in REFLECTORS:                                     # gate-flagged reflected header -> ignore it
        if name in h and name in flagged:
            h.pop(name)
            done.append((name, "gate"))
    if len(h.get("cookie", "")) > 4096:                         # structural DoS: oversize cookie
        h.pop("cookie", None)
        done.append(("cookie", "oversize"))
    if "MSIE " in h.get("user-agent", ""):                      # structural DoS: legacy-UA lockout page
        h["user-agent"] = traffic.CLEAN["user-agent"]
        done.append(("user-agent", "legacy-ua"))
    if "utm_content" in q:                                      # unkeyed reflected parameter
        q.pop("utm_content")
        done.append(("utm_content", "unkeyed-param"))
    return h, q, done


def render(host, route, q, headers):
    """(status, cache-control, body, reflected). Raw interpolation on purpose: the poisoning bugs. `headers`/`q`
    are already sanitized when SSE is on, so this function stays identical in both modes."""
    reflected = []
    cc = "public, max-age=%d"
    if len(headers.get("cookie", "")) > 4096:
        reflected.append("cookie (oversize, 400)")
        return 400, cc, "<!doctype html><title>400</title><h1>400 Request Header Or Cookie Too Large</h1>", reflected
    ua = headers.get("user-agent", "")
    brand = sites.SITES.get(host, {}).get("brand", host)
    if "MSIE " in ua:
        reflected.append("user-agent (legacy page)")
        return 200, cc, (f"<!doctype html><title>{brand}</title><h1>Your browser is not supported</h1>"
                         f"<p>Detected: {html.escape(ua)}</p><p>Please upgrade to view {brand}.</p>"), reflected

    asset_host = headers.get("x-forwarded-host", "cdn.assets.example")
    if "x-forwarded-host" in headers:
        reflected.append("x-forwarded-host")
    prefix = headers.get("x-forwarded-prefix", "")
    if prefix:
        reflected.append("x-forwarded-prefix")
    lang = (headers.get("accept-language", "en-US").split(",")[0].split(";")[0].strip()) or "en-US"
    if lang != "en-US":
        reflected.append("accept-language")

    if host == sites.SEARCH_HOST:
        return _render_search(q, lang, prefix, asset_host, reflected, cc)
    site = sites.SITES.get(host)
    if site is None:
        return 404, "no-store", f"<!doctype html><h1>Unknown host {html.escape(host)}</h1>", reflected
    if route == "/products" and q.get("id") in site["products"]:
        title, text = site["products"][q["id"]]
    elif route in site["pages"]:
        title, text = site["pages"][route]
    else:
        return 404, "no-store", "<!doctype html><h1>Not found</h1>", reflected

    utm = q.get("utm_content")
    if utm is not None:
        reflected.append("utm_content")
    share = (f'<a class="share" href="{route}?utm_content={utm}">Share</a>' if utm
             else f'<a class="share" href="{route}">Share</a>')
    nav = "".join(f'<a href="{u}">{n}</a>' for u, n in (("/", "Home"), ("/products?id=1", "Browse"),
                                                        ("/products?id=2", "More"), ("/blog", "Notes")))
    body = (f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><title>{title} | {site["brand"]}</title>'
            f'<link rel="stylesheet" href="{prefix}/static/site.css">'
            f'<script src="https://{asset_host}/static/app.js"></script></head>'
            f'<body><header style="background:{site["colour"]};color:#fff;padding:12px"><b>{site["brand"]}</b>'
            f' <nav>{nav}</nav></header><main><p>{site["tagline"]}</p><h1>{title}</h1><p>{text}</p>{share}</main>'
            f'</body></html>')
    return 200, cc, body, reflected


def _render_search(q, lang, prefix, asset_host, reflected, cc):
    term = q.get("q", "")
    head = (f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><title>Find.demo</title>'
            f'<link rel="stylesheet" href="{prefix}/static/site.css">'
            f'<script src="https://{asset_host}/static/app.js"></script></head>')
    link_host = asset_host if "x-forwarded-host" in reflected else None
    if not term:
        cards = "".join(f'<li><a href="http://{h}/">{s["brand"]}</a> - {s["tagline"]}</li>'
                        for h, s in sites.SITES.items())
        return 200, cc, (head + '<body><header><b>Find.demo</b></header><main><h1>Find.demo</h1>'
                         '<form action="/search"><input name="q" placeholder="Search"><button>Search</button></form>'
                         f'<ul>{cards}</ul></main></body></html>'), reflected
    base = f"http://{link_host}" if link_host else ""
    rows = "".join(f'<li><a href="{base}/go?to=http://{h}/products?id=1">{s["brand"]}: {term}</a>'
                   f'<p>{s["tagline"]}</p></li>' for h, s in sites.SITES.items())
    return 200, cc, (head + f'<body><header><b>Find.demo</b></header><main><h1>Results for {html.escape(term)}</h1>'
                     f'<ol>{rows}</ol></main></body></html>'), reflected


class Backend:
    def __init__(self, args):
        self.ttl = args.ttl
        self.sse_on = args.sse
        print(f"fitting the SSE gate on {args.train} benign requests ...", flush=True)
        self.sse = SSE(traffic.training_set(args.train))
        print(f"  done in {self.sse.fit_s}s (gate thr {self.sse.gate.thr:.2f}, sem thr {self.sse.sem.thr:.2f}); "
              f"SSE starts {'ON' if self.sse_on else 'OFF'}", flush=True)
        self.hub, self.latest = web.Hub(keep=400), {}
        self.seq = 0
        self.lock = threading.Lock()
        threading.Thread(target=self._tick, daemon=True).start()

    def serve_request(self, path, headers, client):
        t0 = time.time()
        route = path.partition("?")[0]
        pnames = traffic.param_names(path)
        insp = self.sse.inspect(headers, pnames)
        clean_h, clean_q, sanitized = plan_sanitize(headers, path, insp["flagged"], self.sse_on)
        status, cc, body, reflected = render(headers.get("host", "kestrel.demo").split(":")[0], route, clean_q, clean_h)
        if "%d" in cc:
            cc = cc % self.ttl
        learn = self.sse.learn(insp, [f for f, _ in sanitized]) if (self.sse_on and sanitized) else []
        rows = [dict(r, sanitized=any(s == r["name"] for s, _ in sanitized)) for r in insp["rows"]]
        self._emit({"kind": "request", "t": time.time(), "host": headers.get("host", ""), "path": path,
                    "status": status, "sse_on": self.sse_on, "flagged": bool(insp["flagged"]),
                    "sanitized": [{"field": f, "reason": why} for f, why in sanitized],
                    "reflected": reflected, "insp": {"rows": rows, "req": insp["req"]}, "learn": learn,
                    "ms": round((time.time() - t0) * 1000, 2), "from": client})
        return status, body.encode()

    def set_sse(self, on):
        self.sse_on = bool(on)
        self._emit({"kind": "note", "t": time.time(), "field": f"SSE turned {'ON' if self.sse_on else 'OFF'}",
                    "type": "note"})

    def state(self):
        return {"kind": "state", "t": time.time(), "engine": self.sse.state(self.sse_on), "sse_on": self.sse_on,
                "ttl": self.ttl, "sites": sites.catalogue()}

    def _emit(self, ev):
        with self.lock:
            self.seq += 1
            ev["id"] = self.seq
        self.hub.publish(ev)

    def _tick(self):
        while True:
            time.sleep(1)
            st = self.state()
            self.latest["state"] = st
            self.hub.publish(st)


def origin_handler(be):
    class H(web.Handler):
        def do_GET(self):
            headers = {k.lower(): v for k, v in self.headers.items()}
            status, data = be.serve_request(self.path, headers, self.client_address[0])
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store" if status >= 404 else f"public, max-age={be.ttl}")
            self.end_headers()
            self.wfile.write(data)
    return H


def viz_handler(be):
    class H(web.Handler):
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

        def do_GET(self):
            p = urlsplit(self.path).path
            if p == "/api/stream":
                be.hub.stream(self)
            elif p == "/api/state":
                self.send_json(be.state())
            elif p == "/api/sites":
                self.send_json(sites.catalogue())
            elif not self.send_static():
                self.send_error(404)

        def do_POST(self):
            p = urlsplit(self.path).path
            if p == "/api/sse":
                be.set_sse(self.read_json().get("on"))
                self.send_json(be.state())
            elif p == "/ingest":                                # optional: client streams its poison rate here
                evs = self.read_json()
                for ev in (evs if isinstance(evs, list) else [evs]):
                    be.hub.publish(ev)
                self.send_json({"ok": True})
            else:
                self.send_error(404)
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080, help="origin port")
    ap.add_argument("--viz-port", type=int, default=8090, help="SSE visualiser / control port")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--ttl", type=int, default=30, help="max-age the origin sends (the cache obeys it)")
    ap.add_argument("--train", type=int, default=3000, help="benign requests the SSE gate is fitted on")
    ap.add_argument("--sse", action="store_true", help="start with SSE already on (default: off)")
    a = ap.parse_args()
    be = Backend(a)
    web.serve(origin_handler(be), a.bind, a.port)
    web.serve(viz_handler(be), a.bind, a.viz_port)
    ip = web.lan_ip()
    print(f"origin      http://{ip}:{a.port}   (give this to the proxy laptop: --origin)")
    print(f"            serving {', '.join(sites.hosts())}")
    print(f"visualiser  http://{ip}:{a.viz_port}   (open it here; the SSE on/off switch lives on this page)", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
