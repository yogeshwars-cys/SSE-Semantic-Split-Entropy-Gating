"""Laptop A: the origin web server and the SSE visualiser.

    python demo/backend/app.py

Origin on :8080 (what the proxy fetches), visualiser on :8090 (open it in a browser on this laptop). The proxy
streams every gate score, split verdict and calibration step here; the origin adds its own request log.

The origin serves five fake `.demo` sites plus a toy search engine (demo/common/sites.py), each deliberately
poisonable through unkeyed inputs, as in BENCHMARKS.md Part I:
  x-forwarded-host     becomes the host of the page's script URL
  x-forwarded-prefix   becomes the stylesheet's path prefix (the header a Param Miner style scan finds)
  accept-language      its first tag is written into <html lang="..."> unescaped
  utm_content          is echoed into the share link (the cache leaves utm_* out of the key)
  user-agent           an old MSIE gets the "browser not supported" page
  cookie               over 4 KB is answered 400, and the error is cacheable (CPDoS)
On find.demo the result-card link is built from x-forwarded-host, so the search page itself is poisonable.
Benign variation in any of these leaves the page byte-identical, so the split check only fires on real poison."""
import argparse
import html
import os
import sys
import time
from urllib.parse import parse_qsl, urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from demo.common import sites, web  # noqa: E402


def _head(title, brand, colour, lang, prefix, asset_host):
    return (f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8">'
            f'<title>{title} | {brand}</title>'
            f'<link rel="stylesheet" href="{prefix}/static/site.css">'
            f'<style>body{{font-family:system-ui,sans-serif;margin:0}}header{{background:{colour};color:#fff;'
            f'padding:14px 20px}}header nav a{{color:#fff;margin-left:14px;text-decoration:none}}'
            f'main{{padding:22px;max-width:720px}}</style>'
            f'<script src="https://{asset_host}/static/app.js"></script></head>')


def _nav(brand):
    links = (("/", "Home"), ("/products?id=1", "Browse"), ("/products?id=2", "More"), ("/blog", "Notes"))
    return f'<header><b>{brand}</b><nav>' + "".join(f'<a href="{u}">{n}</a>' for u, n in links) + '</nav></header>'


def render(host, path, headers, ttl):
    """(status, cache-control, body, reflected inputs). Raw interpolation on purpose: these are the poisoning bugs."""
    route, _, query = path.partition("?")
    q = dict(parse_qsl(query, keep_blank_values=True))
    reflected = []
    cc = f"public, max-age={ttl}"

    if len(headers.get("cookie", "")) > 4096:                   # cacheable 400 (CPDoS)
        reflected.append("cookie (oversize, 400)")
        return 400, cc, "<!doctype html><title>400</title><h1>400 Request Header Or Cookie Too Large</h1>", reflected
    ua = headers.get("user-agent", "")
    brand = sites.SITES.get(host, {}).get("brand", host)
    if "MSIE " in ua:                                           # legacy page, cacheable denial for all viewers
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
    body = (_head(title, site["brand"], site["colour"], lang, prefix, asset_host)
            + f'<body>{_nav(site["brand"])}'
            + f'<main><p class="tag">{site["tagline"]}</p><h1>{title}</h1><p>{text}</p>{share}</main></body></html>')
    return 200, cc, body, reflected


def _render_search(q, lang, prefix, asset_host, reflected, cc):
    """find.demo: a landing page listing the five sites, and /search?q= returning result cards.

    Each card's click-through link is built from x-forwarded-host, so a poisoned fill sends every later searcher to
    the attacker's host. Benign requests carry no x-forwarded-host, so the default link keeps the page byte-identical."""
    term = q.get("q", "")
    head = _head("Find.demo" if not term else f"Results for {term}", "Find.demo", "#0f766e", lang, prefix, asset_host)
    link_host = asset_host if "x-forwarded-host" in reflected or asset_host != "cdn.assets.example" else None
    if not term:
        cards = "".join(f'<li><a href="http://{h}/">{s["brand"]}</a> — {s["tagline"]}</li>'
                        for h, s in sites.SITES.items())
        body = (head + '<body><header><b>Find.demo</b></header><main><h1>Find.demo</h1>'
                '<form action="/search"><input name="q" placeholder="Search the demo web"><button>Search</button></form>'
                f'<ul class="sites">{cards}</ul></main></body></html>')
        return 200, cc, body, reflected
    # A result for each site; the outbound link goes through find.demo's redirector, whose host comes from the header.
    base = f"http://{link_host}" if link_host else ""
    rows = "".join(
        f'<li><a href="{base}/go?to=http://{h}/products?id=1">{s["brand"]}: {term}</a>'
        f'<p>{s["tagline"]}</p></li>' for h, s in sites.SITES.items())
    body = (head + f'<body><header><b>Find.demo</b></header><main><h1>Results for {html.escape(term)}</h1>'
            f'<ol class="results">{rows}</ol></main></body></html>')
    return 200, cc, body, reflected


def origin_handler(hub, ttl):
    class H(web.Handler):
        def do_GET(self):
            t0 = time.time()
            headers = {k.lower(): v for k, v in self.headers.items()}
            host = (headers.get("host") or "").split(":")[0] or "kestrel.demo"
            status, cc, body, reflected = render(host, self.path, headers, ttl)
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", cc)
            self.end_headers()
            self.wfile.write(data)
            hub.publish({"kind": "origin", "t": time.time(), "host": host, "path": self.path, "status": status,
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
            elif p == "/api/sites":
                self.send_json(sites.catalogue())
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
    print(f"            serving {', '.join(sites.hosts())}")
    print(f"visualiser  http://{ip}:{a.viz_port}   (give this to the proxy laptop: --viz; open it here)", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
