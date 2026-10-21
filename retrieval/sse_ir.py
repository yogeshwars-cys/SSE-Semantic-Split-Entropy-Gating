"""sse_ir.py: SSE for retrieval, generalised. numpy only.

Three layers, as in sse/, all recalibrated on every training miss:
  info gate   surprisal-weighted BM25 over word postings of the full document, Part A and Part B (split_doc)
  semantic    hashed character 3-gram + word + bigram TF-IDF, cosine to each document
  split       reranks the front gates' top `depth` candidates with a linear verdict over 7 features: gate on the
              full document / Part A / Part B, semantic, and the share of high-information query terms found in
              A, in B, in both. It starts as the front-gate blend (0.65 gate + 0.35 semantic), so untrained it
              changes nothing.
A miss is a training query whose best relevant document is not ahead of the strongest wrong one by `margin`
(or is not among the candidates at all). Each miss recalibrates all three layers at once from the differentiated
information between the right and the wrong document: every split weight, word gain and 3-gram gain moves by how
much its own feature or term actually separated the two in the verdict (the pairwise hinge gradient), symmetric
up and down. Once per lap everything shrinks back toward its start, so only what the training queries keep
pushing survives. Calibration sees training data only: a validation slice of the training queries picks the lap
and step sizes; test queries are scored once, frozen.

Compression:
  learned gains  int8 log-gains with stochastic rounding (updates smaller than one step still add up in
                 expectation): 1 byte per word and per 3-gram bucket, fixed however many misses are learned
  indexes        numpy CSR, uint8 values (BM25 tf linear, 3-gram weights sqrt-companded) + int32 document ids,
                 5 bytes per entry against 8 for float32 CSR; the front gates only shortlist and the split stage
                 rescores the shortlist.
"""
import copy
import math

import numpy as np

from sse_retrieval import char_grams, split_doc, tokenize

DIM = 1 << 18
FEATURES = ("gate", "semantic", "gate_a", "gate_b", "info_a", "info_b", "info_ab", "residual")
W0 = np.array([0.65, 0.35, 0, 0, 0, 0, 0, 0.15], np.float64)   # "residual" is 0 until a residual memory holds anything


def top_residual(i, w, ref, T):
    """Positive part of (w - ref[i]), its top T entries, unit length, sorted by index (empty if nothing is left)."""
    r = w - ref
    keep = r > 0
    i, r = i[keep], r[keep]
    if not len(i):
        return i, r
    o = np.argsort(-r)[:T]
    i, r = i[o], r[o] / np.linalg.norm(r[o])
    o = np.argsort(i)
    return i[o], r[o]


class Residuals:
    """Differential residual memory. On a training miss it stores
         key    what the query carried beyond the average training query  (top Tk 3-gram residual)
         value  what the right document carried beyond the wrong one       (top Tv 3-gram residual)
    A new query whose residual aligns with a key (cos > tau) gets its candidates rescored along that key's value:
    the split's "residual" feature. A key whose value pushed a wrong document up on a later miss loses strength
    (dead at 0); a key within cos `merge` of a stored one merges into it. Keys sit in one flat growable index
    (gram, key id, weight), so matching is one searchsorted over Tk entries per key."""

    def __init__(self, mu_q, tau=0.6, Tk=16, Tv=32, merge=0.85):
        self.mu_q, self.tau, self.Tk, self.Tv, self.merge = mu_q, tau, Tk, Tv, merge
        self.G, self.J, self.W, self.n = np.zeros(1024, np.int64), np.zeros(1024, np.int32), np.zeros(1024), 0
        self.strength, self.values = [], []

    def match(self, ki, kw):
        """Live keys aligned with residual (ki, kw) above tau: (ids, cos)."""
        if not self.n or not len(ki):
            return np.zeros(0, np.int64), np.zeros(0)
        G, J, W = self.G[:self.n], self.J[:self.n], self.W[:self.n]
        p = np.minimum(np.searchsorted(ki, G), len(ki) - 1)
        m = ki[p] == G
        a = np.bincount(J[m], W[m] * kw[p[m]], minlength=len(self.strength))
        a[np.asarray(self.strength) <= 0] = 0
        ids = np.flatnonzero(a > self.tau)
        return ids, a[ids]

    def key(self, qi, qw):
        return top_residual(qi, qw, self.mu_q[qi], self.Tk)

    def add(self, ki, kw, pi, pw, ni, nw):
        ref = np.zeros(len(pi))
        p = np.searchsorted(ni, pi)
        ok = (p < len(ni)) & (ni[np.minimum(p, len(ni) - 1)] == pi)
        ref[ok] = nw[p[ok]]
        vi, vw = top_residual(pi, pw, ref, self.Tv)
        if not len(ki) or not len(vi):
            return
        ids, a = self.match(ki, kw)
        if len(ids) and a.max() > self.merge:                   # same miss again: strengthen, fold the value in
            j = int(ids[a.argmax()])
            oi, ow = self.values[j]
            u, inv = np.unique(np.concatenate([oi, vi]), return_inverse=True)
            uw = np.bincount(inv, np.concatenate([ow * self.strength[j], vw]))
            self.values[j] = top_residual(u, uw / np.linalg.norm(uw), np.zeros(len(u)), self.Tv)
            self.strength[j] += 1
            return
        while self.n + len(ki) > len(self.G):
            self.G, self.J, self.W = (np.concatenate([x, np.zeros_like(x)]) for x in (self.G, self.J, self.W))
        self.G[self.n:self.n + len(ki)], self.J[self.n:self.n + len(ki)] = ki, len(self.strength)
        self.W[self.n:self.n + len(ki)] = kw
        self.n += len(ki)
        self.strength.append(1)
        self.values.append((vi, vw))

    def weaken(self, ids):
        for j in ids:
            self.strength[j] -= 1

    def nbytes(self):
        return int(self.n * (8 + 4 + 8) + sum(i.nbytes + w.nbytes for i, w in self.values))


class LogGains:
    """Multiplicative gains stored as int8 log-gains, stochastic rounding, clipped to [lo, hi]."""

    def __init__(self, n, lo=0.30, hi=2.40, seed=0):
        self.step = max(-math.log(lo), math.log(hi)) / 127
        self.lo, self.hi = round(math.log(lo) / self.step), round(math.log(hi) / self.step)
        self.q = np.zeros(n, np.int8)
        self.rng = np.random.default_rng(seed)

    def get(self, idx):
        return np.exp(self.q[idx].astype(np.float64) * self.step)

    def add(self, idx, delta):
        """delta: log-gain change per (unique) index."""
        x = self.q[idx] + delta / self.step
        f = np.floor(x)
        self.q[idx] = np.clip(f + (self.rng.random(len(x)) < x - f), self.lo, self.hi).astype(np.int8)

    def shrink(self, lam):
        nz = np.flatnonzero(self.q)
        if len(nz):
            self.add(nz, -lam * self.q[nz] * self.step)


class CSR:
    """Row-major sparse matrix (rows = words or 3-gram buckets, columns = documents), uint8 values."""

    def __init__(self, rows, cols, vals, n_rows, n_docs, scale):
        o = np.argsort(rows, kind="stable")                        # documents stay ascending inside each row
        self.doc, self.q = cols[o].astype(np.int32), vals[o]
        self.ptr = np.searchsorted(rows[o], np.arange(n_rows + 1))
        self.N, self.scale = n_docs, scale

    def gather(self, rows):
        """Positions of every entry in `rows`, and which of `rows` each one belongs to."""
        s, e = self.ptr[rows], self.ptr[rows + 1]
        ln = e - s
        return np.repeat(s - np.cumsum(ln) + ln, ln) + np.arange(ln.sum()), np.repeat(np.arange(len(rows)), ln)

    def value(self, pos):
        return self.q[pos] * self.scale

    def score(self, rows, w):
        pos, k = self.gather(rows)
        return np.bincount(self.doc[pos], w[k] * self.value(pos), minlength=self.N), (pos, k)

    def at(self, g, n, d):
        """Values of the n gathered rows in document d (0 where absent)."""
        pos, k = g
        m = self.doc[pos] == d
        out = np.zeros(n)
        out[k[m]] = self.value(pos[m])
        return out

    def nbytes(self):
        return int(self.doc.nbytes + self.q.nbytes + self.ptr.nbytes)


class SqrtCSR(CSR):
    def value(self, pos):
        return (self.q[pos] * self.scale) ** 2


def postings(docs_tokens, vocab, k1=1.2, b=0.75):
    """BM25-saturated term frequencies, word-major."""
    ids, cnt = zip(*[np.unique(np.fromiter((vocab[t] for t in toks), np.int32, len(toks)), return_counts=True)
                     if toks else (np.zeros(0, np.int32), np.zeros(0, np.int64)) for toks in docs_tokens])
    dl = np.maximum(np.array([len(t) for t in docs_tokens], np.float64), 1)
    doc = np.repeat(np.arange(len(ids), dtype=np.int32), [len(i) for i in ids])
    tf = np.concatenate(cnt).astype(np.float64)
    ntf = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[doc] / dl.mean()))
    scale = (k1 + 1) / 255
    return CSR(np.concatenate(ids), doc, np.round(ntf / scale).astype(np.uint8), len(vocab), len(ids), scale)


class SSEIR:
    def __init__(self, corpus, depth=100, split_f=0.28, dim=DIM):
        self.doc_ids = [str(r["_id"]) for r in corpus]
        self.depth, self.dim = depth, dim
        full, toks_a, toks_b = [], [], []
        for r in corpus:
            title, text = r.get("title", "") or "", r.get("text", "") or ""
            a, b = split_doc(title, text, strategy="regions", f=split_f)
            full.append(f"{title}. {title}. {text}".strip())
            toks_a.append(tokenize(a))
            toks_b.append(tokenize(b))
        self.texts = full
        toks = [tokenize(t) for t in full]
        self.vocab = {}
        for tt in (toks, toks_a, toks_b):
            for d in tt:
                for t in d:
                    self.vocab.setdefault(t, len(self.vocab))
        self.full, self.pa, self.pb = (postings(tt, self.vocab) for tt in (toks, toks_a, toks_b))
        N = len(corpus)
        df = np.diff(self.full.ptr)
        s = -np.log2((df + 0.1) / (N + 0.2))
        mu, sd = s[df > 0].mean(), s[df > 0].std() or 1.0
        self.surprisal, self.mu_s = s, mu
        self.base = s * (1 + 0.15 * np.maximum(0, (s - mu) / sd))  # EntropyGate's base weight
        rows = [np.unique(np.asarray(char_grams(t, dim=dim), np.int32), return_counts=True) for t in full]
        gdf = np.bincount(np.concatenate([i for i, _ in rows]), minlength=dim)
        self.idf = np.log((N + 1) / (gdf + 1)) + 1
        vals = []
        for i, c in rows:
            w = (1 + np.log(c)) * self.idf[i]
            vals.append(np.sqrt(w / (np.linalg.norm(w) or 1)))
        self.sem = SqrtCSR(np.concatenate([i for i, _ in rows]),
                           np.repeat(np.arange(N, dtype=np.int32), [len(i) for i, _ in rows]),
                           np.round(np.concatenate(vals) * 255).astype(np.uint8), dim, N, 1 / 255)
        self.word_gain, self.gram_gain = LogGains(len(self.vocab), seed=1), LogGains(dim, seed=2)
        self.w = W0.copy()
        self._wsum, self._wn = np.zeros_like(W0), 0               # averaged split weights over a lap
        self.res = None

    def enable_residuals(self, train_texts, **kw):
        """Turn on the differential residual memory; mu = the average training query (training data only)."""
        mu = np.zeros(self.dim)
        for q in train_texts:
            i, w = self.unit(q)
            mu[i] += w
        self.res = Residuals(mu / max(len(train_texts), 1), **kw)

    def unit(self, text):
        """3-gram TF-IDF unit vector (base IDF, no learned gains), sorted indices."""
        i, c = np.unique(np.asarray(char_grams(text, dim=self.dim), np.int64), return_counts=True)
        w = (1 + np.log(c)) * self.idf[i]
        return i, w / (np.linalg.norm(w) or 1)

    def nbytes(self):
        return {"postings_full": self.full.nbytes(), "postings_a": self.pa.nbytes(), "postings_b": self.pb.nbytes(),
                "semantic": self.sem.nbytes(), "learned_gains": int(self.word_gain.q.nbytes + self.gram_gain.q.nbytes),
                "residual_memory": self.res.nbytes() if self.res else 0}

    def state(self):
        return self.word_gain.q.copy(), self.gram_gain.q.copy(), self.w.copy(), copy.deepcopy(self.res)

    def load(self, st):
        self.word_gain.q, self.gram_gain.q, self.w = st[0].copy(), st[1].copy(), st[2].copy()
        self.res = copy.deepcopy(st[3]) if len(st) > 3 else None
        self._wsum, self._wn = np.zeros_like(W0), 0

    def _run(self, text):
        """Front gates over every document, split verdict over the shortlist. Keeps what learning needs."""
        c = {}
        for t in tokenize(text):
            if t in self.vocab:
                c[self.vocab[t]] = c.get(self.vocab[t], 0) + 1
        t = np.fromiter(c, np.int64, len(c))
        wq = self.base[t] * self.word_gain.get(t) * (1 + 0.25 * np.log1p(np.fromiter(c.values(), np.float64, len(c))))
        se, gf = self.full.score(t, wq)
        gi, gc = np.unique(np.asarray(char_grams(text, dim=self.dim), np.int64), return_counts=True)
        gw = (1 + np.log(gc)) * self.idf[gi] * self.gram_gain.get(gi)
        gw /= np.linalg.norm(gw) or 1
        ss, gs = self.sem.score(gi, gw)
        me, ms = float(se.max()) or 1.0, float(ss.max()) or 1.0
        s1 = 0.65 * se / me + 0.35 * ss / ms
        k = min(self.depth, len(s1))
        cand = np.argpartition(-s1, k - 1)[:k]
        cand = cand[np.argsort(-s1[cand], kind="stable")]
        sa, ga = self.pa.score(t, wq)
        sb, gb = self.pb.score(t, wq)
        na, nb = float(sa[cand].max()) or 1.0, float(sb[cand].max()) or 1.0
        hi = np.flatnonzero((self.surprisal[t] >= 0.85 * self.mu_s) & np.array([len(x) >= 4 for x in self._words(t)], bool))
        inn = np.zeros((2, k))
        both = np.zeros(k)
        if len(hi):
            where = np.full(self.full.N, -1, np.int64)
            where[cand] = np.arange(k)
            m = []
            for P in (self.pa, self.pb):
                pos, j = P.gather(t[hi])
                col = where[P.doc[pos]]
                M = np.zeros((len(hi), k), bool)
                M[j[col >= 0], col[col >= 0]] = True
                m.append(M)
            inn = np.stack([M.mean(0) for M in m])
            both = (m[0] & m[1]).mean(0)
        resid, ids = np.zeros(k), np.zeros(0, np.int64)
        if self.res is not None and self.res.n:
            ui, uw = self.unit(text)
            ids, a = self.res.match(*self.res.key(ui, uw))
            if len(ids):                                         # rescore the shortlist along the matched values
                vi = np.concatenate([self.res.values[j][0] for j in ids])
                vw = np.concatenate([self.res.values[j][1] * aj for j, aj in zip(ids, a)])
                u, inv = np.unique(vi, return_inverse=True)
                r = self.sem.score(u, np.bincount(inv, vw))[0][cand]
                resid = r / (float(r.max()) or 1.0)
        F = np.column_stack([se[cand] / me, ss[cand] / ms, sa[cand] / na, sb[cand] / nb, inn[0], inn[1], both, resid])
        c.update(t=t, wq=wq, gf=gf, ga=ga, gb=gb, gi=gi, gw=gw, gs=gs, norm=(me, ms, na, nb), s1=s1, cand=cand, F=F,
                 matched=ids)
        return F @ self.w, c

    def _words(self, t):
        if not hasattr(self, "_inv"):
            self._inv = [None] * len(self.vocab)
            for w, i in self.vocab.items():
                self._inv[i] = w
        return [self._inv[i] for i in t]

    def search(self, text, k=10):
        v, c = self._run(text)
        return c["cand"][np.argsort(-v, kind="stable")][:k]

    def learn(self, text, relevant, lr=3.0, lr_split=0.05, margin=0.08):
        """One training query. Returns None (no miss), "split" (reranked wrongly) or "recall" (no relevant
        document shortlisted). A miss recalibrates the split weights and both gates' gains."""
        v, c = self._run(text)
        self._wsum += self.w
        self._wn += 1
        cand, F = c["cand"], c["F"]
        rel = np.isin(cand, relevant)
        if rel.all():
            return None
        neg_i = int(np.flatnonzero(~rel)[np.argmax(v[~rel])])
        neg = int(cand[neg_i])
        if rel.any():
            pos_i = int(np.flatnonzero(rel)[np.argmax(v[rel])])
            if v[pos_i] - v[neg_i] >= margin:
                return None
            pos, w, kind = int(cand[pos_i]), self.w, "split"
            self.w = self.w + lr_split * (F[pos_i] - F[neg_i])   # the split recalibrates on every miss
        else:                                                    # shortlist missed it: teach the front gates
            pos = int(max(relevant, key=lambda d: c["s1"][d]))
            w, kind = W0, "recall"
        if self.res is not None:
            if kind == "split" and len(c["matched"]) and F[neg_i, 7] > F[pos_i, 7]:
                self.res.weaken(c["matched"])                    # the stored direction pushed the wrong doc up
            self.res.add(*self.res.key(*self.unit(text)), *self.unit(self.texts[pos]), *self.unit(self.texts[neg]))
        me, ms, na, nb = c["norm"]
        n = len(c["t"])

        def word_grad(d):                                        # d verdict / d log word gain, per query word
            return c["wq"] * (w[0] * self.full.at(c["gf"], n, d) / me + w[2] * self.pa.at(c["ga"], n, d) / na
                              + w[3] * self.pb.at(c["gb"], n, d) / nb)
        if n:
            self.word_gain.add(c["t"], lr * (word_grad(pos) - word_grad(neg)))
        m = len(c["gi"])
        g = w[1] * c["gw"] * (self.sem.at(c["gs"], m, pos) - self.sem.at(c["gs"], m, neg)) / ms
        self.gram_gain.add(c["gi"], lr * g)
        return kind

    def shrink(self, lam):
        """Once per lap: the split keeps its weights averaged over the lap (an averaged perceptron, far less
        noisy than wherever the last miss left them), then gains go back toward 1 and split weights toward the
        front-gate blend by lam."""
        if self._wn:
            self.w = self._wsum / self._wn
        self._wsum, self._wn = np.zeros_like(W0), 0
        self.word_gain.shrink(lam)
        self.gram_gain.shrink(lam)
        self.w = W0 + (1 - lam) * (self.w - W0)
