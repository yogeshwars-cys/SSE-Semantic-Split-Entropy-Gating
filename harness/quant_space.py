"""quant_space.py: compressed semantic store for the prototype hybrid (binary Hadamard codes + int8 rescore).

Drop-in for a fitted semantic_model.HeaderSpace: every stored normal value (fit-time and learned) is kept as
  code   sign bits of a randomised Hadamard transform of its 3-gram TF-IDF vector: random +-1 signs, a fast
         Walsh-Hadamard transform, the first `bits` coordinates of a seeded permutation (SRHT): bits/8 bytes
  int8   the vector itself, int8 (x 127), in a numpy matrix with one row per 3-gram (like HeaderSpace.MT)
A lookup encodes the query the same way, takes the `k` stored values nearest in Hamming distance (XOR + popcount
over packed bits), rescores those exactly against the int8 rows and returns 1 - the best cosine.
binary=True keeps no int8 vectors at all: each value is two bit planes over the 3-grams coding each count as
0 / 1 / 2 / "3 or more", plus an exception list (patched frame-of-reference) holding the exact count of every
3-gram at 3 or more as a sorted (value, gram) key + uint16 count, + the norm of its counts (float16). Counts stay
exact and the planes binary; rescoring rebuilds idf x count / norm on the query's 3-grams only. Values are built from
their strings; per-gram gains become uint8 log2 levels and IDF float16.
Learning (add / sharpen / soften) behaves as in HeaderSpace. compress() swaps a fitted Semantic's spaces and re-sets
its per-field mean / spread and 1% threshold with compressed distances, on held-out normal traffic.
"""
import statistics

import numpy as np

from semantic_model import DIM, gram_counts


def fwht(X):
    """Walsh-Hadamard transform of every row (length a power of 2), unnormalised."""
    m, n = X.shape
    h = 1
    while h < n:
        Y = X.reshape(m, n // (2 * h), 2, h)
        X = np.stack([Y[:, :, 0] + Y[:, :, 1], Y[:, :, 0] - Y[:, :, 1]], axis=2).reshape(m, n)
        h *= 2
    return X


class CompressedSpace:
    GSTEP = 2.0 / 255                                           # uint8 log2-gain levels over [1, 4]

    def __init__(self, sp, bits=2048, k=64, seed=0, binary=False, values=None):
        rng = np.random.default_rng(seed)
        self.sign = rng.choice(np.array([-1.0, 1.0], np.float32), DIM)
        self.keep = rng.permutation(DIM)[:bits]
        self.bits, self.k, self.binary = bits, k, binary
        self.rare, self.lr, self.gain_max = sp.rare, sp.lr, sp.gain_max
        if binary:
            self.idf = sp.idf.astype(np.float16)
            self.gq = np.round(np.log2(sp.gain) / self.GSTEP).astype(np.uint8)
        else:
            self.idf, self._gain = sp.idf, sp.gain
        X = np.ascontiguousarray(sp.MT.T)                       # fit-time values, (n, DIM) float32
        if getattr(sp, "nL", 0):
            X = np.vstack([X, np.ascontiguousarray(sp.L[:, :sp.nL].T)])
        n, cap = len(X), max(64, 2 * len(X))
        self.C = np.zeros((cap, bits // 8), np.uint8)
        self.n = 0
        if binary:                                               # two bit planes: count 0 / 1 / 2 / 3+ (exceptions)
            self.B = np.zeros((cap, 2, DIM // 8), np.uint8)
            self.norm = np.zeros(cap, np.float16)
            self.ex_key, self.ex_cnt, self.ex_n = np.zeros(1024, np.int64), np.zeros(1024, np.uint16), 0
            vals = sorted(set(values))                          # the values HeaderSpace was fitted on, same order
            for s in range(0, len(vals), 2048):
                self._append_values(vals[s:s + 2048], gain=False)
            self.ver, self._dc = 0, {}
            return
        self.Q = np.zeros((DIM, cap), np.int8)                  # one row per 3-gram, int8 x 127
        for s in range(0, n, 2048):
            self._append(X[s:s + 2048])
        self.ver, self._dc = 0, {}

    def _code(self, X):
        return np.packbits(fwht(X * self.sign)[:, self.keep] > 0, axis=1)

    @property
    def gain(self):
        return np.exp2(self.gq * self.GSTEP) if self.binary else self._gain

    def _grow(self, m):
        while self.n + m > self.C.shape[0]:
            self.C = np.concatenate([self.C, np.zeros_like(self.C)], axis=0)
            if self.binary:
                self.B = np.concatenate([self.B, np.zeros_like(self.B)], axis=0)
                self.norm = np.concatenate([self.norm, np.zeros_like(self.norm)])
            else:
                self.Q = np.concatenate([self.Q, np.zeros_like(self.Q)], axis=1)

    def _append(self, X):
        m = len(X)
        self._grow(m)
        self.Q[:, self.n:self.n + m] = np.round(X.T * 127).astype(np.int8)
        self.C[self.n:self.n + m] = self._code(X)
        self.n += m

    def _append_values(self, vals, gain=True):
        """Binary store: count planes, norm and Hadamard code of each value string."""
        m = len(vals)
        self._grow(m)
        cnt = np.zeros((m, DIM), np.float32)
        code = np.zeros((m, DIM), np.uint8)
        X = np.zeros((m, DIM), np.float32)
        for r, v in enumerate(vals):
            idx, c = gram_counts(v)
            cnt[r, idx] = c
            code[r, idx] = np.minimum(c, 3)
            hi = c >= 3                                          # exact counts of 3+ go to the exception list
            if hi.any():
                e = int(hi.sum())
                while self.ex_n + e > len(self.ex_key):
                    self.ex_key = np.concatenate([self.ex_key, np.zeros_like(self.ex_key)])
                    self.ex_cnt = np.concatenate([self.ex_cnt, np.zeros_like(self.ex_cnt)])
                self.ex_key[self.ex_n:self.ex_n + e] = (self.n + r) * DIM + idx[hi]   # rows append in order: stays sorted
                self.ex_cnt[self.ex_n:self.ex_n + e] = np.minimum(c[hi], 65535)
                self.ex_n += e
            i, w = self._w(idx, c) if gain else (idx, c * self.idf[idx].astype(np.float64) / (np.linalg.norm(c * self.idf[idx].astype(np.float64)) or 1))
            X[r, i] = w
        for p in range(2):
            self.B[self.n:self.n + m, p] = np.packbits((code >> p) & 1, axis=1)
        self.norm[self.n:self.n + m] = np.sqrt(((cnt * self.idf.astype(np.float32)) ** 2).sum(axis=1))
        self.C[self.n:self.n + m] = self._code(X)
        self.n += m

    def _w(self, idx, cnt):
        w = cnt * self.idf[idx].astype(np.float64) * (np.exp2(self.gq[idx] * self.GSTEP) if self.binary else self._gain[idx])
        return idx, w / (np.linalg.norm(w) or 1)

    def _dense(self, idx, w):
        x = np.zeros((1, DIM), np.float32)
        x[0, idx] = w
        return x

    def dist(self, v):
        c = self._dc.get(v)
        if c is not None and c[0] == self.ver:
            return c[1]
        idx, w = self._w(*gram_counts(v))
        ham = np.bitwise_count(self.C[:self.n] ^ self._code(self._dense(idx, w))).sum(axis=1)
        k = min(self.k, self.n)
        cand = np.argpartition(ham, k - 1)[:k]
        if self.binary:                                         # rebuild idf x count / norm on the query's grams only
            sh = (7 - (idx & 7)).astype(np.uint8)
            b = self.B[cand][:, :, idx >> 3]
            c = (((b[:, 0] >> sh) & 1) + 2 * ((b[:, 1] >> sh) & 1)).astype(np.float32)
            m = c == 3
            if m.any():                                          # patch exact counts from the exception list
                keys = (cand[:, None] * DIM + idx[None, :])[m]
                pos = np.minimum(np.searchsorted(self.ex_key[:self.ex_n], keys), self.ex_n - 1)
                c[m] = self.ex_cnt[pos]
            x = c * self.idf[idx].astype(np.float32) / np.maximum(self.norm[cand].astype(np.float32), 1e-6)[:, None]
            best = float((x @ w.astype(np.float32)).max())
        else:
            best = float((w.astype(np.float32) @ self.Q[idx][:, cand].astype(np.float32)).max()) / 127
        d = 1 - best
        if len(self._dc) > 100_000:
            self._dc.clear()
        self._dc[v] = (self.ver, d)
        return d

    def add(self, v):
        self.ver += 1
        if self.binary:
            self._append_values([v])
            return
        idx, w = self._w(*gram_counts(v))
        self._append(self._dense(idx, w))

    def sharpen(self, v):
        self.ver += 1
        idx = gram_counts(v)[0]
        idx = idx[self.idf[idx] >= self.rare]
        if self.binary:
            up = int(round(np.log2(1 + self.lr) / self.GSTEP))
            self.gq[idx] = np.minimum(round(np.log2(self.gain_max) / self.GSTEP), self.gq[idx].astype(np.int32) + up)
        else:
            self._gain[idx] = np.minimum(self.gain_max, self._gain[idx] * (1 + self.lr))

    def soften(self, v):
        self.ver += 1
        idx = gram_counts(v)[0]
        if self.binary:
            down = int(round(-np.log2(1 - self.lr) / self.GSTEP))
            self.gq[idx] = np.maximum(0, self.gq[idx].astype(np.int32) - down)
        else:
            self._gain[idx] = np.maximum(1.0, self._gain[idx] * (1 - self.lr))

    def nbytes(self):
        vals = (self.B[:self.n].nbytes + self.norm[:self.n].nbytes + self.ex_n * 10 if self.binary
                else self.Q[:, :self.n].nbytes)
        side = self.idf.nbytes + (self.gq.nbytes if self.binary else self._gain.nbytes)
        return int(vals + self.C[:self.n].nbytes + side)


def compress(sem, fit, **kw):
    """Swap every HeaderSpace of a fitted Semantic for a CompressedSpace and re-set its thresholds.
    Returns (float32 bytes before, compressed bytes after)."""
    before = sum(sp.MT.nbytes for sp in sem.space.values())
    half = len(fit) // 2
    vals = {k: [] for k in sem.space}
    for h, _ in fit[:half]:                                      # Semantic.fit's values per field
        for k, v in h.items():
            if k in vals:
                vals[k].append(v)
    sem.space = {k: CompressedSpace(sp, seed=i, values=vals[k], **kw) for i, (k, sp) in enumerate(sorted(sem.space.items()))}
    d = {k: [] for k in sem.space}
    for h, _ in fit[half:]:
        for k, v in h.items():
            if k in sem.space:
                d[k].append(sem.space[k].dist(v))
    sem.stat = {k: (statistics.mean(x), max(statistics.pstdev(x), 0.02)) for k, x in d.items() if x}
    ns = sorted(sem.score(h, p) for h, p in fit[half:])
    sem.thr = ns[max(int(len(ns) * (1 - sem.fpr)) - 1, 0)]
    return before, sum(sp.nbytes() for sp in sem.space.values())
