"""Prototype hybrid: balanced gate learning, path-token route thresholds, damped threshold moves. SR-BH lap 1.

Three changes to replay_learn.py's gates_learn_route hybrid (sse/ itself is untouched):
  balanced   a miss sharpens every above-normal field (z >= half the threshold), statistical weights and semantic
             rare 3-grams, at lr_miss (0.15, 3x the false-alarm rate) instead of one field at 0.05
  tokens     route thresholds keyed by path tokens instead of the exact path: each segment (digits folded), the
             first one and two segments, the extension, the depth, the exact path and a global token. A field's
             threshold on a path = the global 1% threshold + the mean shift of its tokens, so a correction learned on
             /wp-admin/a.php also moves /wp-admin/b.php. Shifts live in a fixed hashed table (2^20 entries)
  damped     a correction moves the threshold halfway (eta 0.5) to its target, so it takes two corrections in the
             same direction to commit; an entry that flips direction gets eta / (1 + flips); shifts capped at +-4
Variants (HYB_VARIANT): balanced (exact overrides kept), tokens (tokens + damped, original gate learning), full.
HYB_QUANT=1 compresses both stores: the semantic layer's stored values become binary Hadamard codes + int8 rescore
(quant_space.py, thresholds re-set with compressed distances) and the token shifts become int8 (stochastic rounding).
HYB_QUANT=2 goes fully binary: the semantic layer keeps two 3-gram count bit planes + an exact-count exception list +
a float16 norm per value instead of int8 vectors (gains as uint8 log levels), and each token entry packs its
direction and flip count into one byte.
HYB_QUANT=3 is HYB_QUANT=2 with an int8 rescore (quant_space.py, int8=True): Hamming top-k on the binary codes, then a
uint8-count x int8-query int32 dot instead of a float rescore. Every random draw (SRHT signs and permutation, the
token table's stochastic rounding) comes from splitmix64 and token thresholds sum their int8 shifts as integers, so
cpp/ reproduces this run request by request. HYB_TRACE=<file> writes one verdict per request (precal then lap 1).
HYB_EXACT=1 (with HYB_QUANT=3) adds an exact layer to the token thresholds: one int8 correction per (layer, field,
exact path) in a fixed hashed table (2^22 entries, step 8/127), set in full after each token move so the threshold
lands just past the misjudged score (rounded to the safe side). Tokens generalise across paths; the exact entry is
the published hybrid's per-route override at a fixed size (cpp/: SSE_EXACT=1).
Same split, seed, precalibration (17 Jul) and teacher (dataset label) as replay_learn.py, so lap 1 (18-19 Jul,
traffic never seen) compares directly with its gates_learn_route: 76.7% / 5.1%, 67,953 wrong, 171,034 overrides.
Settings are fixed in advance, not tuned on the replay. HYB_LAPS=N then re-runs the same 18-19 Jul stream as
belt laps 2..N, learning every lap (belt.py's protocol: convergence on traffic already seen, not unseen traffic).

usage: python -I replay_hybrid2.py <data_capec_multilabel.csv> <sse dir> <out.json>
"""
import json
import math
import os
import random
import re
import sys
import time
import zlib

import numpy as np

CSV, REAL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [sys.argv[0], CSV, REAL, OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir
import replay_learn as rl  # noqa: E402
import replay_realtime as rr  # noqa: E402
import run_sse_srbh as base  # noqa: E402

NORMAL = rr.NORMAL
VARIANT = os.environ.get("HYB_VARIANT", "full")
LAPS = int(os.environ.get("HYB_LAPS", 1))


def route_tokens(rt):
    segs = [re.sub(r"[0-9]+", "0", x) for x in rt.lower().split("/") if x]
    toks = ["*", "r:" + rt, "n:%d" % min(len(segs), 6)] + ["s:" + x for x in segs[:8]]
    if segs:
        toks.append("p1:" + segs[0])
        if len(segs) > 1:
            toks.append("p2:" + segs[0] + "/" + segs[1])
        if "." in segs[-1]:
            toks.append("e:" + segs[-1].rsplit(".", 1)[1])
    return list(dict.fromkeys(toks))


class TokenThresholds:
    """Route thresholds as a fixed hashed table of per-(layer, field, path token) shifts, damped moves."""

    def __init__(self, bits=20, eta=0.5, margin=0.25, cap=4.0, int8=False, packed=False, seed=0, portable=False,
                 exact=False):
        self.mask, self.eta, self.margin, self.cap = (1 << bits) - 1, eta, margin, cap
        self.int8, self.step, self.packed = int8, cap / 127, packed
        self.d = np.zeros(1 << bits, np.int8 if int8 else np.float32)
        if packed:                                               # bits 0-5 flips (cap 63), 6 moved up, 7 moved down
            self.st = np.zeros(1 << bits, np.uint8)
        else:
            self.last = np.zeros(1 << bits, np.int8)
            self.flips = np.zeros(1 << bits, np.uint8)
        self.portable = portable                                 # splitmix64 draws, integer shift sums (cpp/ parity)
        if portable:
            import quant_space
            g = quant_space.splitmix64(seed)
            self.uniform = lambda n: np.array([(next(g) >> 11) * 2.0 ** -53 for _ in range(n)])
        else:
            self.rng = np.random.default_rng(seed)
            self.uniform = self.rng.random
        self._rt, self._toks = None, None                       # one-path memo: a request asks for one path
        self.exact = exact
        if exact:
            self.E, self.emask, self.estep = np.zeros(1 << 22, np.int8), (1 << 22) - 1, 8.0 / 127

    def ids(self, rt, layer, field):
        if rt != self._rt:
            self._rt, self._toks = rt, route_tokens(rt)
        toks = self._toks
        return np.fromiter((zlib.crc32(f"{layer}|{field}|{t}".encode()) & self.mask for t in toks), np.int64, len(toks))

    def shift(self, i):
        return self.d[i] * self.step if self.int8 else self.d[i]

    def thr(self, rt, layer, field, base):
        if self.portable:
            i = self.ids(rt, layer, field)
            t = base + float(self.d[i].astype(np.int64).sum()) * self.step / len(i)
            if self.exact:
                t = t + float(self.E[self.eid(rt, layer, field)]) * self.estep
            return t
        return base + float(self.shift(self.ids(rt, layer, field)).mean())

    def over(self, rt, layer, zs, base):
        return [k for k, z in zs.items() if z > self.thr(rt, layer, k, base)]

    def move(self, rt, layer, field, delta):
        i = self.ids(rt, layer, field)
        sign = np.int8(1 if delta > 0 else -1)
        if self.packed:
            st = self.st[i].astype(np.int32)
            flips = np.minimum((st & 63) + ((st & (128 if sign > 0 else 64)) > 0), 63)
            self.st[i] = (flips | (64 if sign > 0 else 128)).astype(np.uint8)
        else:
            flipped = self.last[i] == -sign
            self.flips[i] = np.minimum(self.flips[i].astype(np.int32) + flipped, 255).astype(np.uint8)
            self.last[i] = sign
            flips = self.flips[i]
        eta = self.eta / (1.0 + np.minimum(flips, 9))
        x = np.clip(self.shift(i) + eta * delta, -self.cap, self.cap)
        if self.int8:                                            # stochastic rounding: small moves still add up
            x = x / self.step
            f = np.floor(x)
            x = np.clip(f + (self.uniform(len(x)) < x - f), -127, 127)
        self.d[i] = x

    def eid(self, rt, layer, field):
        return zlib.crc32(f"{layer}|{field}|x:{rt}".encode()) & self.emask

    def pin(self, rt, layer, field, target, base, down):
        """Exact layer: after the token move, set this path's entry so the threshold lands on target."""
        t2 = self.thr(rt, layer, field, base)
        if not (target < t2 if down else target > t2):
            return
        e = self.eid(rt, layer, field)
        x = float(self.E[e]) + (target - t2) / self.estep
        self.E[e] = max(-127.0, min(127.0, math.floor(x) if down else math.ceil(x)))

    def tighten(self, rt, layer, field, z, base):
        t = self.thr(rt, layer, field, base)
        if z - self.margin < t:
            self.move(rt, layer, field, z - self.margin - t)
            if self.exact:
                self.pin(rt, layer, field, z - self.margin, base, True)

    def loosen(self, rt, layer, field, z, base):
        t = self.thr(rt, layer, field, base)
        if z + self.margin > t:
            self.move(rt, layer, field, z + self.margin - t)
            if self.exact:
                self.pin(rt, layer, field, z + self.margin, base, False)

    def stats(self):
        flipped = (self.st & 63) > 0 if self.packed else self.flips > 0
        side = self.st.nbytes if self.packed else self.last.nbytes + self.flips.nbytes
        out = {"entries_set": int((self.d != 0).sum()), "entries_flipped": int(flipped.sum()),
               "bytes": int(self.d.nbytes + side + (self.E.nbytes if self.exact else 0))}
        if self.exact:
            out["exact_entries_set"] = int((self.E != 0).sum())
        return out


class Hybrid2(rl.Detector):
    def __init__(self, gate, sem, balanced, tokens, lr_miss=0.15, int8=False, packed=False, portable=False, exact=False):
        super().__init__(gate, sem, route_overrides=not tokens)
        self.balanced, self.lr_miss = balanced, lr_miss
        self.tok = TokenThresholds(int8=int8, packed=packed, portable=portable, exact=exact) if tokens else None
        self.trace = None

    def verdict(self, rt, zg, zs, now):
        if self.tok is None:
            return super().verdict(rt, zg, zs, now)
        return self.tok.over(rt, "gate", zg, self.gate.thr), self.tok.over(rt, "sem", zs, self.sem.thr)

    def step(self, now, attack, rt, row, tl, learn=True, check=False):
        h, pnames, zg, zs = self.features(row)
        og, os_ = self.verdict(rt, zg, zs, now)
        flagged = bool(og or os_)
        if self.trace is not None:
            self.trace.append("1" if flagged else "0")
        if flagged == attack or not learn:
            return flagged
        tl["corrected"] += 1
        g, s = self.gate, self.sem
        fields = {k: v for k, v in zg.items() if k != "_req"}
        if attack:
            blamed = max(fields, key=fields.get) if fields else None
            if self.balanced:                                    # every above-normal field learns, harder
                up = [k for k, v in fields.items() if v >= 0.5 * g.thr] or ([blamed] if blamed else [])
                for k in up:
                    g.w[k] = min(g.w_max, g.w.get(k, 1.0) * (1 + self.lr_miss))
                sup = [k for k, v in zs.items() if v >= 0.5 * s.thr] or ([max(zs, key=zs.get)] if zs else [])
                for k in sup:
                    sp = s.space[k]
                    lr0, sp.lr = sp.lr, self.lr_miss
                    sp.sharpen(h[k])
                    sp.lr = lr0
            else:
                g.learn(h, pnames, True, blamed=blamed)
                s.learn(h, True, zs=zs)
            if blamed is not None:
                if self.tok is not None:
                    self.tok.tighten(rt, "gate", blamed, zg[blamed], g.thr)
                    if blamed in zs:
                        self.tok.tighten(rt, "sem", blamed, zs[blamed], s.thr)
                else:
                    self.cal.tighten(rt, "gate", blamed, zg[blamed], g.thr, now)
                    if blamed in zs:
                        self.cal.tighten(rt, "sem", blamed, zs[blamed], s.thr, now)
        else:
            g.learn(h, pnames, False, fired=[k for k in og if k != "_req"])
            s.learn(h, False, fired=os_)
            for layer, fired, zz, b in (("gate", og, zg, g.thr), ("sem", os_, zs, s.thr)):
                for k in fired:
                    if self.tok is not None:
                        self.tok.loosen(rt, layer, k, zz[k], b)
                    else:
                        self.cal.loosen(rt, layer, k, zz[k], b, now)
        if check:
            _, _, zg2, zs2 = self.features(row)
            og2, os2 = self.verdict(rt, zg2, zs2, now)
            tl["checked"] += 1
            tl["flipped"] += bool(og2 or os2) == attack
        return flagged


def main():
    t0 = time.perf_counter()
    rows, _ = base.load()
    rnd = random.Random(base.SEED)
    day1 = [r for r in rows if r[0] < rr.DAY2]
    replay = [r for r in rows if rr.DAY2 <= r[0] < base.ATTACK_END]
    b1 = [r for r in day1 if r[1] == NORMAL]
    rnd.shuffle(b1)
    fit_rows = b1[:base.N_FIT]                                   # same draw as replay_learn.py
    used = {id(r) for r in fit_rows}
    precal_stream = [r for r in day1 if id(r) not in used]
    gate, sem = base.Gate(fpr=0.01), base.Semantic(fpr=0.01)
    fit = [base.to_request(r[2], True) for r in fit_rows]
    gate.fit(fit)
    sem.fit(fit)
    level = os.environ.get("HYB_QUANT", "0")
    quant = level in ("1", "2", "3")
    sizes = {"semantic_float32_bytes": sum(sp.MT.nbytes for sp in sem.space.values())}
    if quant:
        import quant_space
        t = time.perf_counter()
        sizes["semantic_float32_bytes"], sizes["semantic_compressed_bytes"] = quant_space.compress(
            sem, fit, binary=level in ("2", "3"), int8=level == "3")
        sizes["compress_seconds"] = round(time.perf_counter() - t, 1)
        print("compressed semantic store:", sizes, flush=True)
    balanced, tokens = {"balanced": (True, False), "tokens": (False, True), "full": (True, True)}[VARIANT]
    det = Hybrid2(gate, sem, balanced, tokens, int8=quant, packed=level in ("2", "3"), portable=level == "3",
                  exact=level == "3" and os.environ.get("HYB_EXACT") == "1")
    if os.environ.get("HYB_TRACE"):
        det.trace = []
    pre = rl.run(det, precal_stream, label=f"{VARIANT} precal")
    print(f"{VARIANT} precal: tpr={pre['tpr']:.3f} fpr={pre['fpr']:.3f} wrong={pre['wrong']}", flush=True)
    t = time.perf_counter()
    p1 = rl.run(det, replay, label=f"{VARIANT} lap1")
    sec = time.perf_counter() - t
    out = {"protocol": __doc__.split("usage:")[0].strip(), "variant": VARIANT,
           "reference_gates_learn_route": {"precal": {"tpr": 0.803, "fpr": 0.027, "wrong": 23835},
                                           "lap1": {"tpr": 0.7673, "fpr": 0.0507, "precision": 0.9714,
                                                    "wrong": 67953, "route_overrides": 171034}},
           "precal": {k: pre[k] for k in ("tpr", "fpr", "wrong")},
           "lap1": {k: p1[k] for k in ("tpr", "fpr", "precision", "wrong", "flipped_rate", "per_class_tpr", "hourly")},
           "seconds": round(sec, 1), "ms_per_request": round(sec / len(replay) * 1e3, 2),
           "route_state": det.tok.stats() if det.tok is not None else {"route_overrides": len(det.cal.o)},
           "quantised": {"0": "none", "1": "int8", "2": "binary", "3": "binary_int8_rescore"}[level], "semantic_sizes_at_fit": sizes,
           "semantic_bytes_after_lap1": sum(sp.nbytes() if hasattr(sp, "nbytes") else sp.MT.nbytes + sp._dense()[:, :sp.nL].nbytes
                                            for sp in det.sem.space.values()),
           "gate_field_weights": {k: round(v, 3) for k, v in det.gate.w.items()}}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    if det.trace is not None:
        with open(os.environ["HYB_TRACE"], "w") as f:
            f.write("".join(det.trace))
    print(f"== {VARIANT} lap 1: tpr={p1['tpr']:.4f} fpr={p1['fpr']:.4f} precision={p1['precision']:.4f} "
          f"wrong={p1['wrong']} route_state={out['route_state']} ({sec:.0f}s, {out['ms_per_request']} ms/req)", flush=True)
    out["belt"] = {}
    for lap in range(2, LAPS + 1):
        t = time.perf_counter()
        r = rl.run(det, replay, label=f"{VARIANT} lap{lap}")
        out["belt"][str(lap)] = {**{k: r[k] for k in ("tpr", "fpr", "precision", "wrong", "per_class_tpr")},
                                 "seconds": round(time.perf_counter() - t, 1),
                                 "route_state": det.tok.stats() if det.tok is not None else {"route_overrides": len(det.cal.o)}}
        print(f"== {VARIANT} lap {lap}: tpr={r['tpr']:.4f} fpr={r['fpr']:.4f} precision={r['precision']:.4f} "
              f"wrong={r['wrong']} ({time.perf_counter() - t:.0f}s)", flush=True)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)
    print("wrote", OUT, f"({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
