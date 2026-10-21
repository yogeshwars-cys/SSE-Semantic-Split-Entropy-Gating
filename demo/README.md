# CPDOS × SSE — three-laptop live demo

A closed-lab demo that shows a **Cache-Poisoned Denial-of-Service (CPDOS)** attack landing against a caching
reverse proxy, then shows the repo's **SSE** gate (Semantic Split Entropy gating, from `sse/`) switched into the
fill path and stopping it. It mirrors the web cache-poisoning *defence* benchmarks in
[`../BENCHMARKS.md`](../BENCHMARKS.md) Part I.

> **Authorised closed-lab defensive-security demo.** Everything is make-believe on your own LAN: the five domains
> are `.demo` names that resolve nowhere, the "bank"/"clinic"/"city" sites hold no real data, the "search engine"
> is a toy page, and the only thing any attack can reach is our own cache. No packet leaves the LAN. Do not point
> it at anything you do not own.

## What you see

Three screens, updating live:

| Laptop | Role | Screen | Port |
|---|---|---|---|
| **A** | origin server + **SSE visualiser** | "what the gate is thinking": per-field gate/semantic z-scores vs thresholds, split Part-A/Part-B verdicts, the calibration ticker | origin `:8080`, visualiser `:8090` |
| **B** | **reverse-proxy cache** | "is the cache poisoned": live cache grid (state + poisoned flag in red), hit ratio, counters, the live mode switch | proxy `:8000`, dashboard `:8001` |
| **C** | **client / attacker** | "how many users got poison": victim throughput, % served poison over time, the attacker console | dashboard/console `:7001` |

```
  Laptop C ── victims + attacker ──►  Laptop B ── SSE gate on every fill ──►  Laptop A ── 5 sites + search
       (dashboard :7001)                (dashboard :8001)                      (origin :8080, visualiser :8090)
                                               └──────── events stream to ─────────┘
```

The five sites are `kestrel.demo` (shop), `aurora.demo` (news), `vaultline.demo` (bank), `medina.demo` (clinic),
`civic.demo` (city), plus the search engine `find.demo` (defined in [`common/sites.py`](common/sites.py)).

## The four attack vectors

Each is an unkeyed input the origin reflects into a **cacheable** response, so one crafted request poisons the
shared entry for every later visitor (the bugs live on purpose in `backend/app.py`'s `render()`):

| Vector | What it does |
|---|---|
| **Search-result link poison** | unkeyed `x-forwarded-host` on `find.demo/search?q=…` → every later searcher gets a result link pointing at `evil.example` |
| **Homepage script poison** | unkeyed `x-forwarded-host` on a site's `/` → the cached `<script src>` points at `evil.example` |
| **Cacheable 400 DoS** | oversize `cookie` (>4 KB) → origin returns a 400 the cache stores, taking that key down for everyone |
| **Legacy user-agent DoS** (bonus) | an MSIE `user-agent` → cached "browser not supported" page served to all |

The dashboards' "poisoned" flag is an audience-only sniff for `evil.example`, the "not supported" page, or a
status ≥ 400; the cache itself never uses it to decide anything.

## Modes (switch live from Laptop B's dashboard)

| Mode | Behaviour |
|---|---|
| `vulnerable` | plain shared cache, no defence — whatever the first request on a key brings back is served to everyone |
| `gates` | the SSE statistical + semantic gate scores each request; a flagged one is answered but never fills the cache |
| `split` | each fill is compared against a clean origin re-fetch (Part B); on a mismatch the clean copy is cached and waiting views are held, then served it |
| `sse` | gates + split + calibration: a split mismatch blames the reflected header and tightens that route's gate; a sample of gate alarms is audited, and false alarms loosen the gate |

## Run (fill in the real LAN IPs)

All three only need Python 3.10+ and `numpy` (same as the repo core). From the repo root on each laptop:

```bash
# Laptop A — origin + SSE visualiser.  Open http://<A>:8090 here.
python demo/backend/app.py

# Laptop B — reverse-proxy cache.  Open http://<B>:8001 here.
python demo/proxy/app.py --origin http://<A>:8080 --viz http://<A>:8090 --mode vulnerable

# Laptop C — client / attacker.  Open http://<C>:7001 here.
python demo/client/app.py --proxy http://<B>:8000 --viz http://<A>:8090
```

Each process prints its own LAN address on start-up. Useful flags:

- proxy `--ttl 10` forces a short cache TTL so a poisoned key expires and the attack can re-race fills on stage;
  `--hold-ms 250` widens the split-check window so "held, then served the verified copy" is visible rather than
  instant; `--audit 0.3` is the share of gate alarms re-checked against a clean copy; `--train 4000` is how many
  benign requests the gates are fitted on at start-up.
- client `--workers 6` simulated victims, `--search-rate 0.25` share of visits that are searches, `--start` to
  begin victim traffic immediately (otherwise press **Start victims** on the dashboard).

**Firewall:** on the LAN, allow inbound TCP `:8000 :8001 :8080 :8090` (A and B) and `:7001` (C). All three bind
`0.0.0.0` by default; pass `--bind` to narrow it.

### One-laptop dry run

Point everything at loopback to rehearse before wiring up the LAN:

```bash
python demo/backend/app.py &
python demo/proxy/app.py --origin http://127.0.0.1:8080 --viz http://127.0.0.1:8090 &
python demo/client/app.py --proxy http://127.0.0.1:8000 --viz http://127.0.0.1:8090 --start
```

## On-stage script

1. **Benign steady state.** Start victims on C. B's grid fills with `verified`/`unchecked` entries across the five
   sites; A's visualiser shows every header scoring *below* threshold; C's "% poison" sits at 0.
2. **Poison lands (SSE off).** Mode = `vulnerable`. Fire a vector from C's attacker console at one site. B's grid
   shows that key flip to **poisoned (red)**; victims hitting it start serving the poison, so C's "% poison" line
   climbs. Show the stored entry.
3. **Turn SSE on.** Purge the cache (B), set mode = `sse`. Re-run the same attack. A's visualiser shows the crafted
   header spike over threshold → the gate blocks the fill (or the split check heals it with the clean copy and the
   calibrator tightens that header's route). B's grid stays clean; C's "% poison" holds at 0.
4. (Optional) cycle the other vectors and watch the calibration ticker learn each one.

A measured loopback run of exactly this: `vulnerable` served poison to victims while the attacker hammered two
vectors; switching to `sse` drove victim poison to **0%**, with the gate blocking the crafted fills and **zero
poisoned entries** left in the cache — the same shape as `BENCHMARKS.md` Part I (unprotected 70–100% harmed → Full
SSE 0%).

## Notes / caveats

- A real browser won't change its `Host` header when you navigate between the `.demo` sites, so Laptop C drives
  traffic with explicit `Host` headers; the `find.demo` page is the visual, C is the engine behind it.
- The teacher for the calibration loop here is the split check's own clean re-fetch (as in Part I) — not a dataset
  label. A gate alarm is only confirmed a false alarm when the audited response matches the clean copy.
- This is a demonstration harness, not a product: it is single-process per laptop, keeps the cache in memory, and
  is tuned for legibility on stage (short TTLs, a visible hold window) rather than throughput.
