"""Laptop 3: the origin web server and the SSE visualiser.

    python demo/backend/app.py

Origin on :8080 (what the proxy fetches), visualiser on :8090 (open it in a browser on this laptop). The proxy
streams every gate score, split verdict and calibration step here; the origin adds its own request log.

The origin is a small shop that is deliberately poisonable through unkeyed inputs, as in BENCHMARKS.md Part I:
  X-Forwarded-Host     becomes the host of the page's script URL
  X-Forwarded-Prefix   becomes the stylesheet's path prefix (the header a Param Miner style scan finds)
  Accept-Language      its first tag is written into <html lang="..."> unescaped
  utm_content          is echoed into the share link (the cache leaves utm_* out of the key)
  User-Agent           an old MSIE gets the "browser not supported" page
  Cookie               over 4 KB is answered 400, and the error is cacheable (CPDoS)
Benign variation in any of these leaves the page byte-identical, so the split check only fires on real poison."""
import argparse
import html
import os
import sys
import time
from urllib.parse import parse_qsl, urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from demo.common import web  # noqa: E402

SHOP = "Kestrel Supply"
PAGES = {
    "/": ("Field gear that lasts", "Waxed canvas packs, steel flasks and wool layers, repaired for free for ten years."),
    "/blog": ("Workshop notes", "How we re-wax a 2014 rucksack, and why we stopped using nylon thread."),
}
PRODUCTS = {"1": ("Ridgeline 28L pack", "Waxed canvas, leather straps, 1.4 kg. 189 EUR."),
            "2": ("Ember steel flask", "Double-walled, 750 ml, keeps tea hot for 14 hours. 42 EUR."),
            "3": ("Moor wool overshirt", "Boiled merino, three sizes, made in Porto. 128 EUR.")}


def render(path, headers, ttl):
    """(status, body, reflected inputs). Raw interpolation on purpose: these are the poisoning bugs."""
    route, _, query = path.partition("?")
    q = dict(parse_qsl(query, keep_blank_values=True))
    reflected = []
    cc = f"public, max-age={ttl}"
    if len(headers.get("cookie", "")) > 4096:
        reflected.append("cookie (oversize, 400)")
        return 400, cc, ("<!doctype html><title>400</title><h1>400 Request Header Or Cookie Too Large</h1>"), reflected
    ua = headers.get("user-agent", "")
    if "MSIE " in ua:
        reflected.append("user-agent (legacy page)")
        return 200, cc, (f"<!doctype html><title>{SHOP}</title><h1>Your browser is not supported</h1>"
                         f"<p>Detected: {ua}</p><p>Please upgrade to view {SHOP}.</p>"), reflected
    if route == "/products" and q.get("id") in PRODUCTS:
        title, text = PRODUCTS[q["id"]]
    elif route in PAGES:
        title, text = PAGES[route]
    else:
        return 404, "no-store", "<!doctype html><h1>Not found</h1>", reflected
    asset_host = headers.get("x-forwarded-host", "cdn.kestrel.example")
    if "x-forwarded-host" in headers:
        reflected.append("x-forwarded-host")
    prefix = headers.get("x-forwarded-prefix", "")
    if prefix:
        reflected.append("x-forwarded-prefix")
    lang = headers.get("accept-language", "en-US").split(",")[0].split(";")[0].strip() or "en-US"
    if lang != "en-US":
        reflected.append("accept-language")
    utm = q.get("utm_content")
    if utm is not None:
        reflected.append("utm_content")
    share = f'<a class="share" href="{route}?utm_content={utm}">Share</a>' if utm else f'<a class="share" href="{route}">Share</a>'
    nav = "".join(f'<a href="{u}">{n}</a>' for u, n in (("/", "Home"), ("/products?id=1", "Packs"),
                                                          ("/products?id=2", "Flasks"), ("/blog", "Workshop")))
    body = (f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><title>{title} | {SHOP}</title>'
            f'<link rel="stylesheet" href="{prefix}/static/site.css">'
            f'<script src="https://{asset_host}/static/app.js"></script></head>'
            f'<body><header><b>{SHOP}</b><nav>{nav}</nav></header><main><h1>{title}</h1><p>{text}</p>{share}</main>'
            f'</body></html>')
    return 200, cc, body, reflected


def origin_handler(hub, ttl):
    class H(web.Handler):
        def do_GET(self):
            t0 = time.time()
            headers = {k.lower(): v for k, v in self.headers.items()}
            status, cc, body, reflected = render(self.path, headers, ttl)
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", cc)
            self.end_headers()
            self.wfile.write(data)
            hub.publish({"kind": "origin", "t": time.time(), "path": self.path, "status": status,
                         "part": headers.get("x-sse-part", ""), "reflected": reflected, "bytes": len(data),
                         "ms": round((time.time() - t0) * 1000, 2), "from": self.client_address[0]})
    return H


def viz_handler(hub, latest):
    class H(web.Handler):
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

        def do_GET(self):
            p = urlsplit(self.path).path
            if p == "/api/stream":
                hub.stream(self)
            elif p == "/api/state":
                self.send_json(latest.get("state") or {})
            elif not self.send_static():
                self.send_error(404)

        def do_POST(self):
            if urlsplit(self.path).path != "/ingest":
                return self.send_error(404)
            evs = self.read_json()
            for ev in evs if isinstance(evs, list) else [evs]:
                if ev.get("kind") == "state":
                    latest["state"] = ev
                    latest["seen"] = time.time()
                hub.publish(ev)
            self.send_json({"ok": True})
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080, help="origin port")
    ap.add_argument("--viz-port", type=int, default=8090, help="SSE visualiser port")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--ttl", type=int, default=10, help="max-age the origin sends; short so the attack can re-race fills")
    a = ap.parse_args()
    hub, latest = web.Hub(keep=400), {}
    web.serve(origin_handler(hub, a.ttl), a.bind, a.port)
    web.serve(viz_handler(hub, latest), a.bind, a.viz_port)
    ip = web.lan_ip()
    print(f"origin      http://{ip}:{a.port}   (give this to the proxy laptop: --origin)")
    print(f"visualiser  http://{ip}:{a.viz_port}   (give this to the proxy laptop: --viz; open it here)", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
