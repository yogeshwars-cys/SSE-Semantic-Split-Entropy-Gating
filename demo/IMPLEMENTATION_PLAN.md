# CPDOS × SSE live demo — implementation plan

A closed-lab, three-laptop demo for a cybersecurity project. It shows a **Cache-Poisoned
Denial-of-Service (CPDOS)** attack landing against a caching reverse proxy, then shows the repo's
**SSE** gate (Semantic Split Entropy gating) turned on in the loop and stopping it. Everything is
make-believe on your own LAN: the domains are `.demo` names that resolve nowhere, the "search
engine" is a toy page, and the only thing under attack is our own cache. This mirrors the web cache
poisoning *defence* benchmarks already in `BENCHMARKS.md` Part I and `results/cache/`.

## Topology (3 laptops, one LAN)

```
  Laptop C  —— client / attacker ———►  Laptop B  —— reverse proxy cache ——►  Laptop A  —— server
  victims + attacker console            SSE gate sits in the fill path         5 fake domains +
  + its own dashboard                   + cache dashboard                      demo search engine
                                                     └──────── all events stream to ────────┘
                                                                        SSE visualiser (on A)
```

- **Laptop A — the server.** Origin for the 5 fake domains + the demo search engine (`:8080`), and
  the **SSE visualiser** (`:8090`). "SSE installed into the loop" = the proxy on B runs the gate on
  every cache fill and streams each gate score / split verdict / calibration step here, so A is where
  the SSE brain is *shown*.
- **Laptop B — the reverse proxy cache.** Clients browse through `:8000`; the **cache dashboard** is
  on `:8001`. The live mode switch (`vulnerable → gates → split → sse`) lives here.
- **Laptop C — client / attacker.** Generates benign victim traffic (randomised 1–3 s think-time)
  and hosts the **attacker console** + its own dashboard.

## What already exists (keep)

- `sse/` — the gates + calibration loop (unchanged, reused).
- `demo/common/web.py` — stdlib HTTP layer: `Hub` (EventSource fan-out), `Handler` (static + JSON),
  `serve`, `lan_ip`, `post_json`.
- `demo/common/traffic.py` — benign browser headers + `training_set()` the gates fit on.
- `demo/backend/app.py` — origin (one shop, all 6 poison vectors) + visualiser ingest. **Single
  site today.**
- `demo/proxy/app.py` + `demo/proxy/engine.py` — caching proxy with the 4 modes and SSE wired into
  the fill path (gate → split check → calibration). **Cache key is path-only today.**

## What to build (checklist)

### 1. Five fake domains + demo search engine  —  `demo/common/sites.py` (new), edit `demo/backend/app.py`
- Move site content into `sites.py`: a `SITES` dict keyed by host, each with brand, theme colour,
  pages/products. Five `.demo` domains, e.g. `kestrel.demo` (shop), `aurora.demo` (news),
  `vaultline.demo` (bank), `medina.demo` (clinic), `civic.demo` (city services).
- Add `SEARCH_HOST = "find.demo"`: a landing page listing all five + a search box; `/search?q=` returns
  ranked result cards whose click-through link is built from `x-forwarded-host` (so the search page is
  itself poisonable — the headline "poison the search engine" moment).
- `render(host, path, headers, ttl)` selects the site by host and **keeps every poison vector** (see
  below). Benign variation stays byte-identical so the split check only fires on real poison.
- Add `catalogue()` → list of `{host, brand, colour}` for the dashboards, and a `/api/sites` route.
- Origin event gains `host`.

### 2. Host-aware cache key  —  `demo/proxy/app.py`
- `cache_key(host, path)` → `host + "|" + path` (utm_/fbclid/gclid still dropped). Pass `host` from the
  request's Host header through `handle` / `_miss` / `fetch` / `clean_headers`. One line of plumbing;
  without it two domains collide in the cache.

### 3. The three attack vectors, selectable live  —  attacker console drives these; origin already supports them
All three already exist in the origin's `render()`; the attacker console (laptop C) just chooses which
to send against which domain:
- **(a) Search-result link poison** — unkeyed `x-forwarded-host` on a hot `find.demo/search?q=…`;
  every victim searching that term gets a malicious result link from cache.
- **(b) Homepage script poison** — unkeyed `x-forwarded-host` on a domain's `/`; cached `<script src>`
  points at `evil.example`.
- **(c) Cacheable 400 DoS** — oversize `cookie` (>4 KB); origin returns a 400 the cache stores, taking
  that key down for all victims. (Keep MSIE `user-agent` "not supported" page as a bonus 4th button.)
- The cache's `Entry.view()` already flags `poisoned`; extend the `TRUTH` sniff so the dashboards light
  up for all three (`b"evil.example"`, `b"not supported"`, and status ≥ 400 already covered).

### 4. Client / attacker laptop  —  `demo/client/app.py` + `demo/client/static/` (new)
- **Victims:** N worker threads pull `(host, headers, path)` from `traffic.training_set`-style
  generation, GET through the proxy, sleep a random **1–3 s** between requests. Count hits/misses and
  whether the body they got is poisoned (same `TRUTH` sniff) → "victims served poison" is the number
  that drops to zero once SSE is on.
- **Attacker console:** buttons for vectors (a)/(b)/(c) × target domain; fires the crafted request
  straight at the proxy, then shows whether the cache now holds the poison.
- **Dashboard** (`:7000`/`:7001`): victim throughput, % poisoned, attacker log. Reuses `web.Hub`.
- Talks to the proxy by IP (`--proxy http://<laptopB>:8000`); optional `--viz` to also stream its
  events to A.

### 5. The three dashboards (real-time visual modules)  —  `demo/*/static/` (new)
Shared look in `demo/common/static/` (one `app.css`, one `sse.js` EventSource helper). Each page is
plain HTML + vanilla JS subscribing to its app's `/api/stream`; no build step.
- **A — SSE visualiser** (`demo/backend/static/index.html`): per-request gate/semantic z-scores vs
  thresholds (the `insp.view.rows`), split Part A/Part B verdicts, and the calibration ticker
  (tighten/loosen/weight events). The "is SSE thinking" screen.
- **B — cache dashboard** (`demo/proxy/static/index.html`): mode switch, live cache grid (key, state
  `unchecked/verified/healed`, age, **poisoned flag in red**), hit ratio, counters (fills, blocked,
  split_caught, held, tightened). The "is the cache poisoned" screen.
- **C — client dashboard** (`demo/client/static/index.html`): victim rps, % poisoned over time,
  attacker action log.

### 6. Docs
- `demo/README.md` — exact per-laptop run commands with IP placeholders, firewall note (allow
  :8000/:8001/:8080/:8090/:7000/:7001 on the LAN), and the on-stage script (below).
- One line in the repo context noting this is an authorised closed-lab defensive-security demo.

## Run (fill in the real LAN IPs)

```bash
# Laptop A — server
python demo/backend/app.py                     # origin :8080, visualiser :8090  → open :8090 here

# Laptop B — reverse proxy cache
python demo/proxy/app.py --origin http://<A>:8080 --viz http://<A>:8090 --mode vulnerable
                                                # proxy :8000, dashboard :8001  → open :8001 here

# Laptop C — client / attacker
python demo/client/app.py --proxy http://<B>:8000 --viz http://<A>:8090
                                                # dashboard :7001  → open :7001 here
```

## On-stage script

1. **Benign steady state.** Start victims on C. B's dashboard fills with `verified`/`unchecked`
   entries across the 5 domains; A's visualiser shows every header scoring *below* threshold. "% poison
   = 0."
2. **Poison lands (SSE off).** Mode = `vulnerable`. Fire a vector from C's attacker console at one
   domain. B's grid shows that key flip to **poisoned (red)**; victims hitting that key start serving
   the poison — the "% poison" line on C jumps. Show the stored entry.
3. **Turn SSE on.** Purge the cache, set mode = `sse` on B. Re-run the same attack. A's visualiser
   shows the crafted header spike over threshold → gate blocks the fill (or the split check heals it
   with the clean copy and the calibrator tightens that header's route). B's grid stays clean;
   C's "% poison" holds at 0.
4. (Optional) cycle the other two vectors; show the calibration ticker learning each one.

## Notes / caveats
- Cross-host navigation in a real browser won't switch the Host header on its own, so laptop C drives
  traffic with explicit Host headers; the search engine page is the visual, C is the engine.
- Short origin TTL (`--ttl 10`) so a poisoned key expires and the attack can re-race fills on stage.
- `--hold-ms` on the proxy widens the split-check window so "held, then served the verified copy" is
  visible rather than instant.
- All attack payloads target only our own cache; nothing leaves the LAN.
```
