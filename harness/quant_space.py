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
int8=True (with binary=True) keeps the same binary store and rescores in integers: the query's idf x weight is
folded into one int8 vector (x 127 / its max), each candidate's 3-gram counts come out of the bit planes and the
exception list as uint8 (saturating at 255), and the cosine is their int32 dot / scale / the stored norm (the u8 x s8
dot that VNNI / pmaddubsw compute). The `refine` best of those are then scored exactly (exact counts, float64, summed
in order) and the best exact cosine is the answer, so int8 only has to rank. It also makes every step portable so the C++ port (cpp/) reproduces it exactly:
SRHT signs and permutation from splitmix64, norms summed in order, gains from a pow() table, ties in the Hamming
top-k going to the lowest index.
Learning (add / sharpen / soften) behaves as in HeaderSpace. compress() swaps a fitted Semantic's spaces and re-sets
its per-field mean / spread and 1% threshold with compressed distances, on held-out normal traffic.
"""
import math
import statistics

import numpy as np

from semantic_model import DIM, gram_counts


M64 = (1 << 64) - 1


def splitmix64(seed):
    """Portable generator, the same stream as cpp/sse.hpp's SplitMix64."""
    s = seed & M64
    while True:
        s = (s + 0x9E3779B97F4A7C15) & M64
        z = ((s ^ (s >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        yield z ^ (z >> 31)


def srht_params(seed, bits):
    """Random +-1 signs and the first `bits` entries of a Fisher-Yates permutation of the 3-gram space."""
    g = splitmix64(seed)
    sign = np.array([1.0 if next(g) >> 63 else -1.0 for _ in range(DIM)], np.float32)
    perm = list(range(DIM))
    for i in range(DIM - 1, 0, -1):
        j = next(g) % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return sign, np.array(perm[:bits])


def seqnorm(w):
    """L2 norm summed left to right (np.linalg.norm's order depends on the BLAS build)."""
    return math.sqrt(float(np.cumsum(w * w)[-1])) if len(w) else 0.0


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

    def __init__(self, sp, bits=2048, k=64, seed=0, binary=False, values=None, int8=False, refine=4):
        if int8:
            binary = True
            self.sign, self.keep = srht_params(seed, bits)
            self.glut = np.array([math.pow(2.0, q * self.GSTEP) for q in range(256)])
        else:
            rng = np.random.default_rng(seed)
            self.sign = rng.choice(np.array([-1.0, 1.0], np.float32), DIM)
            self.keep = rng.permutation(DIM)[:bits]
        self.bits, self.k, self.binary, self.int8, self.refine = bits, k, binary, int8, refine
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
            if self.int8:
                i, w = self._w(idx, c, gain)
                self.norm[self.n + r] = seqnorm(c * self.idf[idx].astype(np.float64))
            else:
                i, w = self._w(idx, c) if gain else (idx, c * self.idf[idx].astype(np.float64) / (np.linalg.norm(c * self.idf[idx].astype(np.float64)) or 1))
            X[r, i] = w
        for p in range(2):
            self.B[self.n:self.n + m, p] = np.packbits((code >> p) & 1, axis=1)
        if not self.int8:
            self.norm[self.n:self.n + m] = np.sqrt(((cnt * self.idf.astype(np.float32)) ** 2).sum(axis=1))
        self.C[self.n:self.n + m] = self._code(X)
        self.n += m

    def _w(self, idx, cnt, gain=True):
        if self.int8:
            w = cnt * self.idf[idx].astype(np.float64) * (self.glut[self.gq[idx]] if gain else 1.0)
            return idx, w / (seqnorm(w) or 1)
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
        if self.int8:                                           # ties at the k-th distance: lowest index first
            kth = np.partition(ham, k - 1)[k - 1]
            less = np.flatnonzero(ham < kth)
            cand = np.concatenate([less, np.flatnonzero(ham == kth)[:k - len(less)]])
            sh = (7 - (idx & 7)).astype(np.uint8)               # u8 counts x s8 query (idf folded in), int32 dot
            b = self.B[cand][:, :, idx >> 3]
            c = (((b[:, 0] >> sh) & 1) + 2 * ((b[:, 1] >> sh) & 1)).astype(np.int32)
            m = c == 3
            if m.any():
                keys = (cand[:, None] * DIM + idx[None, :])[m]
                pos = np.minimum(np.searchsorted(self.ex_key[:self.ex_n], keys), self.ex_n - 1)
                c[m] = self.ex_cnt[pos]
            t = w * self.idf[idx].astype(np.float64)
            scale = 127.0 / t.max()
            tq = np.round(t * scale).astype(np.int32)
            nrm = np.maximum(self.norm[cand].astype(np.float64), 1e-6)
            cos8 = np.minimum(c, 255) @ tq / scale / nrm
            r = min(self.refine, len(cand))                     # exact refine of the int8 top r (ties: lowest row)
            top = np.lexsort((cand, -cos8))[:r]
            best = float((np.cumsum(c[top] * t, axis=1)[:, -1] / nrm[top]).max())
        elif self.binary:                                       # rebuild idf x count / norm on the query's grams only
            cand = np.argpartition(ham, k - 1)[:k]
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
            cand = np.argpartition(ham, k - 1)[:k]
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
