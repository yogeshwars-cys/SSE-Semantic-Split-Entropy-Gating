"""quant_check.py: fidelity of the compressed semantic store (quant_space.py) against the exact float32 one, SR-BH.

Fits the semantic layer on the same 10,000 normal 17 Jul requests as replay_learn.py, compresses a copy, then
compares per-field distances (compressed - exact) on 3,000 more normal 17 Jul requests and 3,000 17 Jul attacks,
and the share of requests each version's semantic layer flags at its own 1% threshold.

usage: python -I quant_check.py <data_capec_multilabel.csv> <sse dir> <int8|binary>
"""
import copy
import os
import random
import sys

import numpy as np

CSV, REAL, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [sys.argv[0], CSV, REAL, "/dev/null"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir
import run_sse_srbh as base  # noqa: E402  (first: it puts the sse dir on the path)
import quant_space  # noqa: E402
import replay_realtime as rr  # noqa: E402


def main():
    rows, _ = base.load()
    rnd = random.Random(base.SEED)
    day1 = [r for r in rows if r[0] < rr.DAY2]
    b1 = [r for r in day1 if r[1] == rr.NORMAL]
    rnd.shuffle(b1)
    fit = [base.to_request(r[2], True) for r in b1[:base.N_FIT]]   # same draw as replay_learn.py
    sem = base.Semantic(fpr=0.01)
    sem.fit(fit)
    exact = copy.deepcopy(sem)
    before, after = quant_space.compress(sem, fit, binary=MODE == "binary")
    print(f"{MODE}: semantic store float32 {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB ({before / after:.1f}x); "
          f"1% threshold exact {exact.thr:.3f}, compressed {sem.thr:.3f}")
    normal = [base.to_request(r[2], True)[0] for r in b1[base.N_FIT:base.N_FIT + 3000]]
    attacks = [base.to_request(r[2], True)[0] for r in rnd.sample([r for r in day1 if r[1] != rr.NORMAL], 3000)]
    for name, hs in (("normal", normal), ("attack", attacks)):
        err = np.array([sem.space[k].dist(v) - exact.space[k].dist(v) for h in hs for k, v in h.items() if k in sem.space])
        print(f"{name}: {len(err)} field values, distance error mean {err.mean():+.4f}, |error| p50 "
              f"{np.median(np.abs(err)):.4f} p99 {np.quantile(np.abs(err), .99):.4f}, within 0.01: "
              f"{np.mean(np.abs(err) < 0.01):.1%}; flagged by the semantic layer: exact {np.mean([exact.odd(h) for h in hs]):.1%}, "
              f"compressed {np.mean([sem.odd(h) for h in hs]):.1%}")


if __name__ == "__main__":
    main()
