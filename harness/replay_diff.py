"""Differential residual memory on the SSE gates: store what a miss had beyond normal, check new requests against it.

On every missed attack, for the blamed field (the semantic layer's oddest field), the residual
    d = top-T of max(x - mu, 0),  normalised
is stored, keyed by field only (route-agnostic): x is the field value's 3-gram TF-IDF unit vector (the field's
HeaderSpace IDF), mu the mean benign vector of that field over the fit sample, so grams normal traffic carries
cancel and d keeps only the few grams the miss carried beyond normal. Every request is then also checked
directionally: it is flagged when cos(x, d_k) > tau for a live d_k of that field, even if its distance score is
in range. A new d within cos `merge` of a stored one merges into it (strength +1); a false alarm raised by the check
costs the matched d_k one strength (dropped at 0); at most `cap` per field (weakest evicted).
Detectors (same split, seed and teacher as replay_learn.py, whose published lap 1 they compare with):
  gates_learn_diff        gates learn + differential memory, no route overrides (does d replace them?)
  gates_learn_route_diff  gates learn + route overrides + differential memory
  control                 gates_learn_route with the check off (tau = inf): must reproduce replay_learn's lap 1
tau is picked per detector on the 17 Jul precalibration day only (fewest wrong verdicts); the picked instance is
carried into the 18-19 Jul replay, lap 1 (traffic never seen). Teacher = dataset label.

usage: python -I replay_diff.py <data_capec_multilabel.csv> <sse dir> <out.json>
"""
import copy
import json
import os
import random
import sys
import time
import urllib.parse
from collections import Counter

import numpy as np

CSV, REAL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [sys.argv[0], CSV, REAL, OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir
import replay_learn as rl  # noqa: E402
import replay_realtime as rr  # noqa: E402
import run_sse_srbh as base  # noqa: E402
from semantic_model import gram_counts  # noqa: E402

NORMAL = rr.NORMAL
TAUS = tuple(float(x) for x in os.environ.get("DIFF_TAUS", "0.5,0.7,0.85").split(","))


class DiffMemory:
    def __init__(self, sem, fit, T=8, tau=0.7, merge=0.8, cap=2048):
        self.idf = {k: sp.idf for k, sp in sem.space.items()}
        self.T, self.tau, self.merge, self.cap = T, tau, merge, cap
        acc, n = {k: np.zeros(len(v)) for k, v in self.idf.items()}, Counter()
        for h, _ in fit:
            for k, v in h.items():
                if k in acc:
                    i, w = self.vec(k, v)
                    acc[k][i] += w
                    n[k] += 1
        self.mu = {k: acc[k] / max(n[k], 1) for k in acc}
        self.d = {k: [] for k in self.idf}                     # per field: [grams, weights, strength]
        self._flat = {}

    def vec(self, k, v):
        i, c = gram_counts(v)
        w = c * self.idf[k][i]
        return i, w / (np.linalg.norm(w) or 1.0)

    def _arrays(self, k):
        f = self._flat.get(k)
        if f is None:
            live = [(j, g, w) for j, (g, w, s) in enumerate(self.d[k]) if s > 0]
            if not live:
                f = (np.zeros(0, np.int64),) * 2 + (np.zeros(0),)
            else:
                f = (np.concatenate([g for _, g, _ in live]), np.concatenate([np.full(len(g), j) for j, g, _ in live]),
                     np.concatenate([w for _, _, w in live]))
            self._flat[k] = f
        return f

    def align(self, k, i, w):
        """Best cos(x, d_j) over the field's live residuals: (score, j)."""
        G, J, W = self._arrays(k)
        if not len(G):
            return 0.0, -1
        p = np.minimum(np.searchsorted(i, G), len(i) - 1)
        m = i[p] == G
        if not m.any():
            return 0.0, -1
        s = np.bincount(J[m], W[m] * w[p[m]], minlength=len(self.d[k]))
        j = int(s.argmax())
        return float(s[j]), j

    def check(self, h):
        """(field, j) of the first live residual the request aligns with above tau, else None."""
        for k, v in h.items():
            if k in self.idf and self.d[k]:
                sc, j = self.align(k, *self.vec(k, v))
                if sc > self.tau:
                    return k, j
        return None

    def add(self, k, v):
        i, w = self.vec(k, v)
        r = w - self.mu[k][i]
        keep = r > 0
        i, r = i[keep], r[keep]
        if not len(i):
            return
        top = np.argsort(-r)[: self.T]
        i, r = i[top], r[top] / np.linalg.norm(r[top])
        o = np.argsort(i)
        i, r = i[o], r[o]
        sc, j = self.align(k, i, r)
        if sc > self.merge:
            self.d[k][j][2] += 1
            return
        if len(self.d[k]) >= self.cap:
            weakest = min(range(len(self.d[k])), key=lambda x: self.d[k][x][2])
            self.d[k][weakest] = [i, r, 1]
        else:
            self.d[k].append([i, r, 1])
        self._flat.pop(k, None)

    def weaken(self, k, j):
        self.d[k][j][2] -= 1
        if self.d[k][j][2] <= 0:
            self._flat.pop(k, None)

    def stats(self):
        live = {k: sum(s > 0 for _, _, s in v) for k, v in self.d.items() if v}
        return {"live_residuals": live, "stored": sum(len(v) for v in self.d.values()),
                "bytes": int(sum(g.nbytes + w.nbytes for v in self.d.values() for g, w, _ in v))}


class DiffDetector(rl.Detector):
    def __init__(self, gate, sem, route, mem):
        super().__init__(gate, sem, route)
        self.mem, self.counts = mem, Counter()

    def step(self, now, attack, rt, row, tl, learn=True, check=False):
        h, pnames, zg, zs = self.features(row)
        og, os_ = self.verdict(rt, zg, zs, now)
        hit = self.mem.check(h)
        flagged = bool(og or os_ or hit)
        self.counts["diff_only_flags"] += hit is not None and not (og or os_)
        if flagged == attack or not learn:
            return flagged
        tl["corrected"] += 1
        if attack:
            self.correct(now, True, rt, h, pnames, zg, zs, og, os_)
            if zs:
                self.mem.add(max(zs, key=zs.get), h[max(zs, key=zs.get)])
        else:
            if og or os_:
                self.correct(now, False, rt, h, pnames, zg, zs, og, os_)
            if hit:
                self.mem.weaken(*hit)
                self.counts["diff_false_alarms"] += 1
        return flagged


def main():
    t0 = time.perf_counter()
    rows, _ = base.load()
    rnd = random.Random(base.SEED)
    day1 = [r for r in rows if r[0] < rr.DAY2]
    replay = [r for r in rows if rr.DAY2 <= r[0] < base.ATTACK_END]
    b1 = [r for r in day1 if r[1] == NORMAL]
    rnd.shuffle(b1)
    fit_rows = b1[:base.N_FIT]                                   # same draw as replay_realtime.py / replay_learn.py
    used = {id(r) for r in fit_rows}
    precal_stream = [r for r in day1 if id(r) not in used]
    fit = [base.to_request(r[2], True) for r in fit_rows]
    gate, sem = base.Gate(fpr=0.01), base.Semantic(fpr=0.01)
    gate.fit(fit)
    sem.fit(fit)
    out = {"protocol": __doc__.split("usage:")[0].strip(), "taus": TAUS,
           "reference_lap1": {"gates_learn": {"tpr": 0.680, "fpr": 0.057, "wrong": 91950, "route_overrides": 0},
                              "gates_learn_route": {"tpr": 0.7673, "fpr": 0.0507, "wrong": 67953,
                                                    "route_overrides": 171034}}}
    runs = (("control_gates_learn_route", True, (float("inf"),)),   # diff check off: must reproduce lap 1
            ("gates_learn_diff", False, TAUS), ("gates_learn_route_diff", True, TAUS))
    for name, route, taus in runs:
        if os.environ.get("DIFF_ONLY") and name != os.environ["DIFF_ONLY"]:
            continue
        pre = {}
        for tau in taus:
            det = DiffDetector(gate, sem, route, DiffMemory(sem, fit, tau=tau))
            t = time.perf_counter()
            r = rl.run(det, precal_stream, label=f"{name} precal tau{tau}")
            pre[tau] = (r, det)
            print(f"{name} precal tau={tau}: tpr={r['tpr']:.3f} fpr={r['fpr']:.3f} wrong={r['wrong']} "
                  f"({time.perf_counter() - t:.0f}s) {det.mem.stats()['live_residuals']}", flush=True)
        tau = min(pre, key=lambda x: pre[x][0]["wrong"])
        det = pre[tau][1]
        del pre
        t = time.perf_counter()
        det.counts = Counter()
        p1 = rl.run(det, replay, label=f"{name} lap1")
        lat = time.perf_counter() - t
        out[name] = {"tau_picked_on_precal": tau,
                     "lap1": {k: p1[k] for k in ("tpr", "fpr", "precision", "wrong", "per_class_tpr")},
                     "seconds": round(lat, 1), "ms_per_request": round(lat / len(replay) * 1e3, 2),
                     "route_overrides": len(det.cal.o), "memory": det.mem.stats(), "diff_counts": dict(det.counts)}
        print(f"== {name} (tau {tau}) lap 1: tpr={p1['tpr']:.4f} fpr={p1['fpr']:.4f} precision={p1['precision']:.4f} "
              f"wrong={p1['wrong']}  overrides={len(det.cal.o)}  memory={det.mem.stats()}  "
              f"{lat / len(replay) * 1e3:.2f} ms/req", flush=True)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
