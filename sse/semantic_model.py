"""semantic_model.py: the last-line layer behind the statistical gate. Vector similarity of header VALUES against
the values seen in normal traffic; learns from its own wrong verdicts (Semantic.learn).

Each value becomes a TF-IDF vector of hashed character 3-grams (lower-cased, digits folded to 0, so version bumps
look alike). Its distance to the nearest baseline value of the same header is 1 - max cosine. 3-grams never seen in
the baseline get the highest IDF, so a few injected characters in a long normal-looking value pull the cosine down.
Per header the distance is turned into a z-score; a request is odd when its largest z passes the (1 - fpr) quantile
of held-out normal requests. Headers the baseline never saw are left to the statistical gate."""
import re
import statistics
import zlib
from functools import lru_cache

import numpy as np

DIM = 4096


def grams(v):
    s = "\x02" + re.sub(r"[0-9]", "0", v.lower()) + "\x03"
    return [zlib.crc32(s[i:i + 3].encode()) % DIM for i in range(len(s) - 2)] or [0]


@lru_cache(maxsize=1 << 18)                                      # values repeat a lot; their grams never change
def gram_counts(v):
    idx, cnt = np.unique(grams(v), return_counts=True)
    idx.flags.writeable = cnt.flags.writeable = False
    return idx, cnt


def _block(rows):
    """Growable part of a HeaderSpace: added values indexed by 3-gram (CSC-style, numpy only)."""
    g = np.concatenate([i for i, _ in rows]).astype(np.int32)
    w = np.concatenate([x for _, x in rows]).astype(np.float32)
    r = np.repeat(np.arange(len(rows), dtype=np.int32), [len(i) for i, _ in rows])
    o = np.argsort(g, kind="stable")
    return g[o], r[o], w[o], np.searchsorted(g[o], np.arange(DIM + 1)), len(rows)


def _merge(blocks):
    """One block holding every row of `blocks` (arrays concatenated, never rebuilt row by row)."""
    off = np.cumsum([0] + [b[4] for b in blocks[:-1]])
    g = np.concatenate([b[0] for b in blocks])
    r = np.concatenate([b[1] + np.int32(o) for b, o in zip(blocks, off)])
    w = np.concatenate([b[2] for b in blocks])
    o = np.argsort(g, kind="stable")
    return g[o], r[o], w[o], np.searchsorted(g[o], np.arange(DIM + 1)), int(sum(b[4] for b in blocks))


def _best(block, idx, w):
    g, r, wb, st, n = block
    s, e = st[idx], st[idx + 1]
    ln = e - s
    tot = int(ln.sum())
    if not tot:
        return 0.0
    pos = np.repeat(s - np.cumsum(ln) + ln, ln) + np.arange(tot)   # every stored entry of the query's grams
    return float(np.bincount(r[pos], wb[pos] * np.repeat(w, ln), minlength=n).max())


class HeaderSpace:
    def __init__(self, values, lr=0.05, gain_max=4.0):
        self.gain = np.ones(DIM)                                # per-3-gram weight, learned from failures
        vals = sorted(set(values))
        df = np.zeros(DIM)
        rows = []
        for v in vals:
            g = grams(v)
            df[list(set(g))] += 1
            rows.append(g)
        self.idf = np.log((len(vals) + 1) / (df + 1)) + 1          # unseen 3-gram -> highest weight
        M = np.zeros((len(vals), DIM), dtype=np.float32)
        for i, g in enumerate(rows):
            idx, w = self.weights(g)
            M[i, idx] = w
        self.MT = np.ascontiguousarray(M.T)                     # row per 3-gram: a query touches only its own rows
        self.rare = np.log((len(vals) + 1) / (0.01 * len(vals) + 1)) + 1   # idf of a gram in <= 1% of values
        self.lr, self.gain_max = lr, gain_max
        self.blocks, self.pending, self._pb = [], [], None      # (older store, converted by _dense)
        self.L, self.nL = None, 0                               # normal values learned after fit
        self.ver, self._dc = 0, {}                              # dist cache, valid while ver is unchanged

    def weights(self, g):
        idx, cnt = np.unique(g, return_counts=True)
        return self._w(idx, cnt)

    def _w(self, idx, cnt):
        w = cnt * self.idf[idx] * self.gain[idx]
        return idx, w / (np.linalg.norm(w) or 1)

    def dist(self, v):
        c = self._dc.get(v)
        if c is not None and c[0] == self.ver:
            return c[1]
        d = self._dist(v)
        if len(self._dc) > 100_000:
            self._dc.clear()
        self._dc[v] = (self.ver, d)
        return d

    def _dist(self, v):
        idx, w = self._w(*gram_counts(v))
        w = w.astype(np.float32)
        best = float((w @ self.MT[idx]).max())
        L = self._dense()
        if self.nL:                                             # learned normal values: same gather + dot as MT
            best = max(best, float((w @ L[idx, :self.nL]).max()))
        return 1 - best

    def _dense(self):
        """Learned values as a dense (3-gram x value) matrix like MT; converts the older block store once."""
        if getattr(self, "L", None) is None:
            blocks, pending = getattr(self, "blocks", []), getattr(self, "pending", [])
            n = sum(b[4] for b in blocks) + len(pending)
            self.L, self.nL = np.zeros((DIM, max(64, 2 * n)), dtype=np.float32), 0
            for g, r, w, _, k in blocks:
                self.L[g, self.nL + r] = w
                self.nL += k
            for i, w in pending:
                self.L[i, self.nL] = w
                self.nL += 1
            self.blocks, self.pending, self._pb = [], [], None
        return self.L

    def add(self, v):
        """A false alarm: remember the value as normal."""
        self.ver += 1
        L = self._dense()
        if self.nL == L.shape[1]:                               # grow by doubling
            self.L = L = np.concatenate([L, np.zeros_like(L)], axis=1)
        idx, w = self._w(*gram_counts(v))
        L[idx, self.nL] = w
        self.nL += 1

    def sharpen(self, v):
        """A miss: the value's grams that are rare in normal traffic weigh more from now on."""
        self.ver += 1
        idx = gram_counts(v)[0]
        idx = idx[self.idf[idx] >= self.rare]
        self.gain[idx] = np.minimum(self.gain_max, self.gain[idx] * (1 + self.lr))

    def soften(self, v):
        """A false alarm: undo boosts on this value's grams."""
        self.ver += 1
        idx = gram_counts(v)[0]
        self.gain[idx] = np.maximum(1.0, self.gain[idx] * (1 - self.lr))


class Semantic:
    """Fit on half the supplied normal requests, set the threshold at the (1 - fpr) quantile of the other half."""

    def __init__(self, fpr=0.01, min_values=20):
        self.fpr, self.min_values, self.ready = fpr, min_values, False

    def fit(self, reqs):
        half = len(reqs) // 2
        byname = {}
        for h, _ in reqs[:half]:
            for k, v in h.items():
                byname.setdefault(k, []).append(v)
        self.space = {k: HeaderSpace(v) for k, v in byname.items() if len(v) >= self.min_values}
        d = {k: [] for k in self.space}
        for h, _ in reqs[half:]:
            for k, v in h.items():
                if k in self.space:
                    d[k].append(self.space[k].dist(v))
        self.stat = {k: (statistics.mean(x), max(statistics.pstdev(x), 0.02)) for k, x in d.items() if x}
        ns = sorted(self.score(h, p) for h, p in reqs[half:])
        self.thr = ns[max(int(len(ns) * (1 - self.fpr)) - 1, 0)]
        self.ready = True

    def header_z(self, h):
        return {k: (self.space[k].dist(v) - self.stat[k][0]) / self.stat[k][1] for k, v in h.items() if k in self.stat}

    def learn(self, h, attack, zs=None, fired=()):
        """Recalibrate the layer itself from a wrong verdict. Miss: sharpen the oddest field's rare grams.
        False alarm: every field that fired learns the value as normal and drops boosts on its grams."""
        if attack:
            zs = self.header_z(h) if zs is None else zs
            if zs:
                k = max(zs, key=zs.get)
                self.space[k].sharpen(h[k])
            return
        for k in fired:
            if k in self.space and k in h:
                self.space[k].soften(h[k])
                self.space[k].add(h[k])

    def score(self, h, p=()):
        return max(list(self.header_z(h).values()) + [0.0])

    def odd(self, h, p=()):
        return self.ready and self.score(h, p) > self.thr
