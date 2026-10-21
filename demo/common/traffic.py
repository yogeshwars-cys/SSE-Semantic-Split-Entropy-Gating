"""traffic.py: benign browser traffic for the demo.

The proxy fits the SSE gates on training_set() at start-up, and the client's simulated victims draw their requests from
the same generator, so the gates see victims as normal and only the attack payloads stand out. Header names are
lower-case, as the proxy scores them."""
import random

PAGES = [("/", 5), ("/products?id=1", 2), ("/products?id=2", 2), ("/products?id=3", 1), ("/blog", 2)]

LANGS = ["en-US,en;q=0.9", "en-US,en;q=0.9", "en-US,en;q=0.5", "en-US", "en-US,en;q=0.9,es;q=0.8",
         "en-US,en;q=0.9,hi;q=0.8", "en-US,en;q=0.9,de;q=0.8,fr;q=0.7", "en-US,en-GB;q=0.9,en;q=0.8"]
ACCEPT_CHROMIUM = ("text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,"
                   "application/signed-exchange;v=b3;q=0.7")
ACCEPT_FIREFOX = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
ACCEPT_SAFARI = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
UTM = {"utm_source": ["newsletter", "twitter", "google"], "utm_medium": ["email", "social", "cpc"],
       "utm_campaign": ["spring_sale", "launch_week"], "utm_content": ["hero_banner", "footer_link", "sidebar"]}

# The neutral request the split check sends to the origin as its clean reference (Part B).
CLEAN = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/129.0.0.0 Safari/537.36",
    "accept": ACCEPT_CHROMIUM,
    "accept-language": "en-US,en;q=0.9",
    "accept-encoding": "gzip, deflate, br",
}


def _browser(rng):
    kind = rng.choices(["chrome", "edge", "chrome_mac", "firefox", "safari", "android", "iphone"],
                       weights=[34, 12, 14, 14, 10, 9, 7])[0]
    v = rng.randint(118, 130)
    if kind == "chrome":
        ua = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{v}.0.0.0 Safari/537.36"
    elif kind == "edge":
        ua = (f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{v}.0.0.0 "
              f"Safari/537.36 Edg/{v}.0.{rng.randint(2000, 2900)}.{rng.randint(40, 99)}")
    elif kind == "chrome_mac":
        ua = f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{v}.0.0.0 Safari/537.36"
    elif kind == "firefox":
        fv = rng.randint(115, 131)
        ua = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:{fv}.0) Gecko/20100101 Firefox/{fv}.0"
    elif kind == "safari":
        sv = rng.choice(["16.6", "17.0", "17.2", "17.4.1", "17.5", "18.0"])
        ua = f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{sv} Safari/605.1.15"
    elif kind == "android":
        ua = f"Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{v}.0.0.0 Mobile Safari/537.36"
    else:
        a, b = rng.choice([(16, 6), (17, 1), (17, 4), (17, 5), (18, 0)])
        ua = (f"Mozilla/5.0 (iPhone; CPU iPhone OS {a}_{b} like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
              f"Version/{a}.{b} Mobile/15E148 Safari/604.1")
    return kind, v, ua


def browser_headers(rng, host):
    """One browser navigation's headers (without the page itself)."""
    kind, v, ua = _browser(rng)
    chromium = kind in ("chrome", "edge", "chrome_mac", "android")
    h = {"host": host, "connection": "keep-alive"}
    if chromium:
        brand = "Microsoft Edge" if kind == "edge" else "Google Chrome"
        plat = {"chrome": "Windows", "edge": "Windows", "chrome_mac": "macOS", "android": "Android"}[kind]
        h["sec-ch-ua"] = f'"{brand}";v="{v}", "Chromium";v="{v}", "Not?A_Brand";v="99"'
        h["sec-ch-ua-mobile"] = "?1" if kind == "android" else "?0"
        h["sec-ch-ua-platform"] = f'"{plat}"'
    if rng.random() < 0.9:
        h["upgrade-insecure-requests"] = "1"
    h["user-agent"] = ua
    h["accept"] = ACCEPT_CHROMIUM if chromium else ACCEPT_FIREFOX if kind == "firefox" else ACCEPT_SAFARI
    if chromium or kind == "firefox":
        h["sec-fetch-site"] = rng.choice(["none", "same-origin", "same-origin", "cross-site"])
        h["sec-fetch-mode"] = "navigate"
        h["sec-fetch-user"] = "?1"
        h["sec-fetch-dest"] = "document"
    if rng.random() < 0.35:
        h["referer"] = rng.choice([f"http://{host}/", f"http://{host}/products?id={rng.randint(1, 3)}",
                                   f"http://{host}/blog", "https://www.google.com/"])
    h["accept-encoding"] = "gzip, deflate, br, zstd" if (chromium and v >= 123) or kind == "firefox" else "gzip, deflate, br"
    h["accept-language"] = rng.choice(LANGS)
    if rng.random() < 0.6:
        c = f"sid={rng.getrandbits(128):032x}; theme={rng.choice(['light', 'dark'])}"
        if rng.random() < 0.4:
            c += f"; _ga=GA1.1.{rng.randint(10**9, 10**10 - 1)}.{rng.randint(1690000000, 1730000000)}"
        if rng.random() < 0.3:
            c += f"; cart={rng.randint(1, 9)}"
        if rng.random() < 0.5:
            c += "; consent=granted"
        h["cookie"] = c
    if rng.random() < 0.15:
        h["cache-control"] = "max-age=0"
    if rng.random() < 0.1:
        h["dnt"] = "1"
    return h


def page(rng, utm_rate=0.1):
    """A page to visit; by default one visit in ten arrives from a marketing link carrying utm_* parameters.
    The victims use utm_rate=0: the origin echoes utm_content, so a benign marketing visit would also make the split
    check fire, which muddies the demo's counters."""
    p = rng.choices([u for u, _ in PAGES], weights=[w for _, w in PAGES])[0]
    if rng.random() < utm_rate:
        tags = "&".join(f"{k}={rng.choice(v)}" for k, v in UTM.items() if rng.random() < 0.8)
        if tags:
            p += ("&" if "?" in p else "?") + tags
    return p


def param_names(path):
    q = path.split("?", 1)[1] if "?" in path else ""
    return [kv.split("=", 1)[0] for kv in q.split("&") if kv]


def training_set(n=4000, seed=7, host="shop.local"):
    """(headers without host, query-parameter names) for n benign navigations, the gates' notion of normal."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        h = browser_headers(rng, host)
        h.pop("host")
        out.append((h, param_names(page(rng))))
    return out
