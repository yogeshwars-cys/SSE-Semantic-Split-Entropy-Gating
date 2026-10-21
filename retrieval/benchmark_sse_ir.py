"""benchmark_sse_ir.py: the generalised IR SSE (sse_ir.py), 10 learning laps on training queries, then frozen test.

  data     scifact: BEIR train qrels (809) to learn from, BEIR test qrels (300) to test
           nq:      prep_nq_100k.py's slice; seeded halves of the 3,452 queries (1,726 / 1,726)
  calib    the training queries are split again, seeded 80 / 20: laps learn on the 80% ("fit"), the 20% ("val")
           picks the configuration and the lap. Test is never seen until the end.
  configs  full (split + both gates learn) at two gate step sizes, and two ablations: split only, gates only
  lap      every fit query once; each miss recalibrates at once (sse_ir.SSEIR.learn); then shrink toward start
  report   test for: untrained, the val-picked configuration at its val-picked lap, the same configuration at lap 10;
           paired bootstrap CIs; index sizes; search latency

usage: python -I benchmark_sse_ir.py <scifact|nq> <beir dir> <out.json>
"""
import json
import os
import sys
import time

import numpy as np

KIND, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [sys.argv[0], DATA, OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir
import benchmark_sse_laps as bl  # noqa: E402  (metrics, paired bootstrap)
from sse_ir import FEATURES, SSEIR  # noqa: E402

LAPS = int(os.environ.get("IR_LAPS", 10))
SEED = 20261021
CONFIGS = (("full", dict(lr=3.0, lr_split=0.05)), ("full, big split step", dict(lr=3.0, lr_split=0.2)),
           ("split only", dict(lr=0.0, lr_split=0.05)), ("gates only", dict(lr=3.0, lr_split=0.0)))
SHRINK = 0.1


def load():
    corpus = [json.loads(l) for l in open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8")]
    queries = {str(r["_id"]): r["text"] for r in map(json.loads, open(os.path.join(DATA, "queries.jsonl"), encoding="utf-8"))}

    def qrels(split):
        out = {}
        for line in open(os.path.join(DATA, "qrels", f"{split}.tsv"), encoding="utf-8").read().splitlines()[1:]:
            q, d, s = line.split("\t")
            if int(s) > 0:
                out.setdefault(q, set()).add(d)
        return out
    test = qrels("test")
    if KIND == "scifact":
        train = sorted(q for q in qrels("train") if q not in test)
        train_rel = qrels("train")
    else:
        qids = [q for q in queries if q in test]
        perm = np.random.default_rng(SEED).permutation(len(qids))
        train = [qids[i] for i in perm[: len(qids) // 2]]
        held = {qids[i] for i in perm[len(qids) // 2:]}
        train_rel, test = test, {q: v for q, v in test.items() if q in held}
    perm = np.random.default_rng(SEED + 1).permutation(len(train))
    nv = len(train) // 5
    val = [train[i] for i in perm[:nv]]
    fit = [train[i] for i in perm[nv:]]
    return corpus, queries, {q: train_rel[q] for q in train}, fit, val, test


def main():
    t0 = time.perf_counter()
    corpus, queries, train_rel, fit, val, test = load()
    print(f"{KIND}: {len(corpus):,} docs; fit {len(fit)}, val {len(val)} (training queries), test {len(test)}", flush=True)
    t = time.perf_counter()
    m = SSEIR(corpus)
    build_s = time.perf_counter() - t
    del corpus
    row = {d: i for i, d in enumerate(m.doc_ids)}
    rel = {q: np.array(sorted(row[d] for d in v if d in row), np.int64) for q, v in {**train_rel, **test}.items()}
    sizes = m.nbytes()
    print(f"built in {build_s:.0f}s; sizes MB " + str({k: round(v / 1e6, 1) for k, v in sizes.items()}), flush=True)
    qrels = {**train_rel, **test}

    def evaluate(qs, timing=False):
        rows, lat = [], []
        for q in qs:
            s = time.perf_counter()
            top = m.search(queries[q])
            lat.append((time.perf_counter() - s) * 1e3)
            rows.append(bl.metrics([m.doc_ids[int(i)] for i in top], qrels[q]))
        return (np.array(rows), lat) if timing else np.array(rows)

    start = m.state()
    te0, lat = evaluate(test, timing=True)
    va0 = evaluate(val)
    out = {"protocol": __doc__.split("usage:")[0].strip(), "dataset": KIND, "fit": len(fit), "val": len(val),
           "test": len(test), "build_seconds": round(build_s, 1), "index_mb": {k: round(v / 1e6, 2) for k, v in sizes.items()},
           "search_ms_p50": round(float(np.median(lat)), 2), "untrained": {"val": bl.summary(va0), "test": bl.summary(te0)},
           "configs": {}}
    print(f"untrained: val nDCG {va0[:, 0].mean():.4f}  test nDCG {te0[:, 0].mean():.4f}  search p50 {np.median(lat):.1f} ms", flush=True)

    best = (va0[:, 0].mean(), "untrained", 0, start)
    lap10 = {}
    for name, kw in CONFIGS:
        m.load(start)
        laps = {}
        for lap in range(1, LAPS + 1):
            t = time.perf_counter()
            kinds = {"split": 0, "recall": 0}
            for q in fit:
                k = m.learn(queries[q], rel[q], **kw)
                if k:
                    kinds[k] += 1
            m.shrink(SHRINK)
            va = evaluate(val)[:, 0].mean()
            laps[str(lap)] = {"val_ndcg10": round(float(va), 4), "split_misses": kinds["split"],
                              "recall_misses": kinds["recall"], "seconds": round(time.perf_counter() - t, 1)}
            if va > best[0]:
                best = (va, name, lap, m.state())
            print(f"{name:20s} lap {lap:2d}  misses {kinds['split']:4d} split + {kinds['recall']:4d} recall  "
                  f"val nDCG {va:.4f} ({va - va0[:, 0].mean():+.4f})  {time.perf_counter() - t:.0f}s", flush=True)
        lap10[name] = m.state()
        out["configs"][name] = {**kw, "shrink": SHRINK, "laps": laps,
                                "split_weights": dict(zip(FEATURES, np.round(m.w, 3).tolist()))}
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)

    _, name, lap, st = best
    m.load(st)
    te_best = evaluate(test)
    m.load(lap10[name] if name in lap10 else start)
    te_10 = evaluate(test)
    out["picked_on_val"] = {"config": name, "lap": lap, "val_ndcg10": round(float(best[0]), 4)}
    out["test"] = {"untrained": bl.summary(te0), f"{name} lap {lap} (val pick)": bl.summary(te_best),
                   f"{name} lap {LAPS}": bl.summary(te_10),
                   "val_pick_vs_untrained": bl.paired_ci(te0[:, 0], te_best[:, 0]),
                   f"lap{LAPS}_vs_untrained": bl.paired_ci(te0[:, 0], te_10[:, 0])}
    m.load(st)
    words = ("the", "and", "what", "who", "when", "does", "you")
    out["picked_state"] = {"split_weights": dict(zip(FEATURES, np.round(m.w, 3).tolist())),
                           "words_moved": int((m.word_gain.q != 0).sum()), "grams_moved": int((m.gram_gain.q != 0).sum()),
                           "word_gains": {w: round(float(m.word_gain.get(np.array([m.vocab[w]]))[0]), 3) for w in words if w in m.vocab}}
    out["seconds"] = round(time.perf_counter() - t0, 1)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\npicked on val: {name} lap {lap}")
    print(json.dumps(out["test"], indent=1))
    print(json.dumps(out["picked_state"], indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
