"""calibration.py: the calibration loop of SSE (Semantic Split Entropy Gating).

A verifier with ground truth (the split check, or any labelled feedback) tells SSE when a verdict was wrong, and
that verdict is fed back to the front gates:
  1. blame (when a flagged response and a clean reference exist): find which request header / query value shows up in
     the flagged body but not in the clean one; otherwise blame the field the statistical gate scored highest;
  2. calibrate: per (route, layer, field) thresholds move to just past the score of the value that was misjudged.
Scoped to the route (path without query). With a half-life the override decays back to the global threshold; with
halflife=None (real-time mode) it stays until the next wrong verdict moves it. The gates themselves also learn from
every wrong verdict (anomaly_model.Gate.learn, semantic_model.Semantic.learn)."""
import re


def route(path):
    return path.split("?", 1)[0]


def blame(flagged, clean, headers, qpairs):
    """Header / query-param names whose value (or a >= 4-char token of it) is in the flagged body only."""
    hit = []
    for kind, items in (("header", headers.items()), ("param", qpairs)):
        for k, v in items:
            toks = [t for t in re.split(r"[\s;,()\"'<>=]+", v) if len(t) >= 4] + ([v] if len(v) >= 4 else [])
            if any(t in flagged and t not in clean for t in toks):
                hit.append((kind, k))
    return hit


class Calibrator:
    """halflife=None: real-time mode. Overrides never decay; every wrong verdict moves them instead
    (tighten() on a miss, loosen() on a false alarm), so the threshold follows the feedback stream."""

    def __init__(self, halflife=120.0, margin=0.25):
        self.hl, self.margin, self.o = halflife, margin, {}     # (route, layer, name) -> (threshold, set_at)
        self.by = {}                                            # (route, layer) -> names with an override there

    def _set(self, rt, layer, name, new, now):
        self.o[(rt, layer, name)] = (new, now)
        self.by.setdefault((rt, layer), set()).add(name)

    def tighten(self, rt, layer, name, z, base, now):
        new = z - self.margin
        if new < self.thr(rt, layer, name, base, now):
            self._set(rt, layer, name, new, now)
            return True
        return False

    def loosen(self, rt, layer, name, z, base, now):
        """False alarm: lift this (route, field) threshold just above the benign value's score."""
        new = z + self.margin
        if new > self.thr(rt, layer, name, base, now):
            self._set(rt, layer, name, new, now)
            return True
        return False

    def thr(self, rt, layer, name, base, now):
        t = self.o.get((rt, layer, name))
        if t is None:
            return base
        if self.hl is None:
            return t[0]
        return base - (base - t[0]) * 0.5 ** ((now - t[1]) / self.hl)

    def over(self, rt, layer, zs, base, now):
        """Every field whose score passes its own threshold (base where nothing was learned)."""
        return [name for name, z in zs.items() if z > self.thr(rt, layer, name, base, now)]

    def odd(self, rt, layer, zs, base, now):
        return any(zs.get(name, -1e9) > self.thr(rt, layer, name, base, now)
                   for name in list(self.by.get((rt, layer), ())))   # only this route's entries, not the whole table

    def snapshot(self, now, bases):
        return {f"{r}|{l}|{n}": round(self.thr(r, l, n, bases.get(l, 0.0), now), 2) for (r, l, n) in list(self.o)}
