# CPDOS × SSE — three-laptop live demo

A closed-lab demo of the web cache-poisoning defence from [`../BENCHMARKS.md`](../BENCHMARKS.md) Part I.
**SSE lives in the backend (origin) and is switched on/off at will.** With it off, crafted requests poison the
shared cache; with it on, the origin sanitizes anomalous inputs so whatever the cache stores is clean. The cache
is a plain reverse proxy that only reports telemetry — it has no say in gating.

> **Authorised closed-lab defensive-security demo.** The five domains are `.demo` names that resolve nowhere, the
> "bank"/"clinic"/"city" sites hold no real data, and the only thing any attack can reach is our own cache. No
> packet leaves the LAN.

## Roles

| Laptop | Role | Screen | Ports |
|---|---|---|---|
| **A** | **origin + SSE** — serves the sites and runs the SSE gate; **the on/off switch is here** | "what SSE is thinking" + the SSE toggle | origin `:8080`, visualiser/control `:8090` |
| **B** | **cache** — plain caching reverse proxy, no gating | "is the cache poisoned" (telemetry only) | proxy `:8000`, dashboard `:8001` |
| **C** | **client / attacker** — victim browsers + attacker console | victim traffic + attack buttons | console `:7001` |

```
  Laptop C ── victims + attacks ──►  Laptop B ── caches what A returns ──►  Laptop A ── sites + SSE gate + ON/OFF
   (console :7001)                    (dashboard :8001, passive)            (origin :8080, visualiser :8090)
```

How it fits together: C sends traffic (and crafted attacks) through B; B caches whatever A returns; A decides —
with **SSE off** it reflects the crafted input into a cacheable page (poison), with **SSE on** it ignores the
flagged input and returns a clean page. B just stores and shows the result.

## The attack vectors (fired from C's console)

Each is an unkeyed input the origin reflects into a cacheable response:

| Vector | What it plants |
|---|---|
| **Homepage script poison** | `x-forwarded-host` → cached `<script src>` points at `evil.example` |
| **Search-result link poison** | `x-forwarded-host` on `find.demo/search` → poisoned result links |
| **Cacheable 400 DoS** | oversize `cookie` → origin returns a cacheable 400 |
| **Legacy user-agent DoS** | MSIE `user-agent` → cached "browser not supported" page |

With **SSE on**, the origin's gate flags the anomalous value and drops it (or, for the structural DoS payloads,
neutralizes them), so the page — and the cached copy — is clean. Benign traffic scores below threshold and is
served unchanged, so normal pages stay byte-identical and cache normally.

## Run (fill in the real LAN IPs)

Python **3.10+** everywhere. **Only Laptop A needs `numpy`** (the SSE gate); B and C are standard-library only.
Run each command from the folder where you extracted that laptop's zip (it contains a `demo/` folder, so the
`demo\...` paths resolve).

```bash
# Laptop A — origin + SSE.  Open http://<A>:8090 here (the SSE on/off switch is on this page).
pip install numpy
python demo/backend/app.py                         # origin :8080, visualiser/control :8090

# Laptop B — cache.  Open http://<B>:8001 here.
python demo/proxy/app.py --origin http://<A>:8080  # proxy :8000, dashboard :8001

# Laptop C — client / attacker.  Open http://<C>:7001 here.
python demo/client/app.py --proxy http://<B>:8000 --viz http://<A>:8090
```

(On Windows use `python` or `py`, and `tar -xf <zip>` to extract.)

Useful flags: backend `--ttl 30` (cache lifetime it advertises), `--train 3000` (benign fit size), `--sse` (start
with SSE already on). Proxy `--ttl N` forces a cache TTL instead of obeying the origin.

**Firewall:** allow inbound TCP `:8000 :8001 :8080 :8090` on A/B and `:7001` on C across the LAN.

## On-stage script

1. **Benign steady state.** Start victims on C. B's grid fills with clean entries; A's visualiser shows every
   field scoring below threshold.
2. **Poison lands — SSE off.** With A's switch **OFF**, fire a vector from C. B's grid shows that key go **red
   (poisoned)**; A's visualiser shows the crafted field, but SSE takes no action.
3. **Flip SSE on.** Click the switch **ON** on A, purge B, fire the same vector. A's visualiser shows the field
   flagged and **sanitized**; B's grid stays clean. Flip it back off and watch poison return — that's the whole point.

## Test it without clicking

`run_test.py` (stdlib only) drives the three apps and prints the result:

```bash
python run_test.py                                             # one machine
python run_test.py --backend http://<A>:8090 --proxy http://<B>:8001   # across the LAN, from laptop C
```

It fires each vector with SSE off then on (expect `POISONED` → `clean`), then runs a live window and reports the
cache's poisoned-entry count with SSE off vs on.

## Notes

- A real browser won't change its `Host` header between the `.demo` sites, so C drives traffic with explicit Host
  headers; the `find.demo` page is the visual, C is the engine.
- Whether a victim is *served* poison depends on the cache race (who fills a key first within its TTL); the
  reliable signal is **B's poisoned-entry count**, which tracks the cache's actual state. Lower the TTL for a
  punchier re-race on stage.
- This is a demonstration harness, not a product: in-memory cache, single process per laptop, tuned for legibility.
