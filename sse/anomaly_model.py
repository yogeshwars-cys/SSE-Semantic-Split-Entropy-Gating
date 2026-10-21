"""anomaly_model.py: the statistical front layer. Information-theory and vector-similarity score of a request
against a baseline of normal traffic; learns from its own wrong verdicts (Gate.learn)."""
import math
import statistics
from collections import Counter, defaultdict
from functools import lru_cache


@lru_cache(maxsize=1 << 18)                                      # field values repeat a lot; entropy is pure
def shannon(s):
    if not s:
        return 0.0
    c = Counter(s)
    return -sum(v / len(s) * math.log2(v / len(s)) for v in c.values())


class Baseline:
    def __init__(self, reqs):
        self.N = len(reqs)
        self.names, self.params = Counter(), Counter()
        lens, ents = defaultdict(list), defaultdict(list)
        for h, p in reqs:
            for k, v in h.items():
                self.names[k] += 1
                lens[k].append(len(v))
                ents[k].append(shannon(v))
            for k in p:
                self.params[k] += 1
        self.acc = {k: [len(lens[k]), sum(lens[k]), sum(x * x for x in lens[k]), sum(ents[k]), sum(x * x for x in ents[k])]
                    for k in lens}                              # running sums, so update() can fold in more traffic
        allL = [x for v in lens.values() for x in v] or [0]
        allE = [x for v in ents.values() for x in v] or [0]
        self.glen = (statistics.mean(allL), statistics.pstdev(allL) or 1)
        self.gent = (statistics.mean(allE), statistics.pstdev(allE) or 1)
        self.len = {k: (statistics.mean(v), max(statistics.pstdev(v), 2.0)) for k, v in lens.items()}
        self.ent = {k: (statistics.mean(v), max(statistics.pstdev(v), 0.3)) for k, v in ents.items()}
        self.cent = {k: self.names[k] / self.N for k in self.names}
        self.cnorm = math.sqrt(sum(x * x for x in self.cent.values())) or 1

    def update(self, hdrs, params):
        """Fold one more normal request into the baseline (a false alarm teaches what normal looks like)."""
        self.N += 1
        for k in params:
            self.params[k] += 1
        for k, v in hdrs.items():
            self.names[k] += 1
            a = self.acc.setdefault(k, [0, 0.0, 0.0, 0.0, 0.0])
            ln, e = len(v), shannon(v)
            a[0] += 1; a[1] += ln; a[2] += ln * ln; a[3] += e; a[4] += e * e
            ml, me = a[1] / a[0], a[3] / a[0]
            self.len[k] = (ml, max(math.sqrt(max(a[2] / a[0] - ml * ml, 0.0)), 2.0))
            self.ent[k] = (me, max(math.sqrt(max(a[4] / a[0] - me * me, 0.0)), 0.3))
        self.cent = {k: c / self.N for k, c in self.names.items()}
        self.cnorm = math.sqrt(sum(x * x for x in self.cent.values())) or 1

    def surprisal(self, count):
        return -math.log2((count + 0.1) / (self.N + 0.2))

    def raw(self, hdrs, params, w=None):
        w = w or {}
        s1 = max([self.surprisal(self.names.get(k, 0)) for k in hdrs] +
                 [self.surprisal(self.params.get(k, 0)) for k in params] + [0.0])
        z_len = z_ent = 0.0
        for k, v in hdrs.items():
            ml, sl = self.len.get(k, self.glen)
            me, se = self.ent.get(k, self.gent)
            z_len = max(z_len, abs(len(v) - ml) / sl * w.get(k, 1.0))
            z_ent = max(z_ent, abs(shannon(v) - me) / se * w.get(k, 1.0))
        dot = sum(self.cent.get(k, 0.0) for k in hdrs)
        cos = dot / ((math.sqrt(len(hdrs)) * self.cnorm) or 1)
        return (s1, z_len, z_ent, 1 - cos)


class Gate:
    """Fit on half the supplied normal requests, set the threshold at the (1 - fpr) quantile of the other half."""

    def __init__(self, fpr=0.01, lr=0.05, w_min=0.5, w_max=3.0):
        self.fpr, self.ready = fpr, False
        self.lr, self.w_min, self.w_max, self.w = lr, w_min, w_max, {}   # per-field sensitivity, learned from failures

    def fit(self, reqs):
        half = len(reqs) // 2
        self.base = Baseline(reqs[:half])
        raw = [self.base.raw(*r) for r in reqs[:half]]
        self.mu = [statistics.mean(c) for c in zip(*raw)]
        self.sd = [statistics.pstdev(c) or 1 for c in zip(*raw)]
        ns = sorted(self.score(*r) for r in reqs[half:])
        self.thr = ns[max(int(len(ns) * (1 - self.fpr)) - 1, 0)]
        self.ready = True

    def score(self, h, p):
        return max((x - m) / s for x, m, s in zip(self.base.raw(h, p, self.w), self.mu, self.sd))

    def learn(self, h, p, attack, blamed=None, fired=()):
        """Recalibrate the gate itself from a wrong verdict. Miss: the blamed field gets more sensitive.
        False alarm: the request joins the baseline and the fields that fired get less sensitive."""
        if attack:
            if blamed is not None:
                self.w[blamed] = min(self.w_max, self.w.get(blamed, 1.0) * (1 + self.lr))
            return
        self.base.update(h, p)
        for k in fired:
            if k in h:
                self.w[k] = max(self.w_min, self.w.get(k, 1.0) * (1 - self.lr))

    def odd(self, h, p):
        return self.ready and self.score(h, p) > self.thr

    def header_z(self, h):
        """Per-header score on the same scale as score(): the worst of its name surprisal, length and entropy z."""
        b, out = self.base, {}
        for k, v in h.items():
            ml, sl = b.len.get(k, b.glen)
            me, se = b.ent.get(k, b.gent)
            wk = self.w.get(k, 1.0)
            comp = (b.surprisal(b.names.get(k, 0)), abs(len(v) - ml) / sl * wk, abs(shannon(v) - me) / se * wk)
            out[k] = max((x - m) / s for x, m, s in zip(comp, self.mu, self.sd))
        return out
