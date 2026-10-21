"""benchmark_sse_nq_laps.py: 10 training laps on NQ-100k, then freeze and test on queries never trained on.

Same technique as benchmark_sse_laps.py, on the NQ port (benchmark_sse_nq.py): the SSE entropy gate + sparse
3-gram space blended 0.65 / 0.35, trained by blame-by-diff (finetune) one lap at a time over the train half.
  split   the 3,452 NQ test queries, seeded halves: 1,726 train, 1,726 held out
  lap k   finetune(epochs=1, start_epoch=k-1): the shipped lr schedule (0.14 * 0.8^epoch) carried on; lap 3 ends
          in exactly the state of benchmark_sse_nq.py's 3-epoch fine-tune
  train   scored after every lap (fit to queries it is learning from)
  test    the held-out half is scored only after all 10 laps, frozen, on snapshots of the learned gains taken at
          laps 0, 1, 3 and 10, so nothing about the held-out half ever steers training. BM25 is the reference.
Data: prep_nq_100k.py's slice (BEIR-shaped).

usage: python -I benchmark_sse_nq_laps.py <nq-100k dir> <out.json>
"""
import json
import os
import sys
import time

import numpy as np

DATA, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir
import benchmark_sse_laps as bl  # noqa: E402  (metrics, paired bootstrap)
import benchmark_sse_nq as nq  # noqa: E402  (sparse semantic space, BM25, finetune)
from sse_retrieval import EntropyGate, tokenize  # noqa: E402

LAPS = int(os.environ.get("NQ_LAPS", 10))
SNAPSHOTS = (0, 1, 3, LAPS)
SEED = 20261021


def load():
    corpus = [json.loads(l) for l in open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8")]
    queries = {str(r["_id"]): r["text"] for r in map(json.loads, open(os.path.join(DATA, "queries.jsonl"), encoding="utf-8"))}
    qrels = {}
    for line in open(os.path.join(DATA, "qrels", "test.tsv"), encoding="utf-8").read().splitlines()[1:]:
        q, d, s = line.split("\t")
        if int(s) > 0:
            qrels.setdefault(q, set()).add(d)
    return corpus, queries, qrels


def main():
    t0 = time.perf_counter()
    corpus, queries, qrels = load()
    doc_ids = [str(r["_id"]) for r in corpus]
    id2row = {d: i for i, d in enumerate(doc_ids)}
    plain = [f"{r.get('title', '')} {r['text']}".strip() for r in corpus]
    titled = [f"{r.get('title', '')}. {r.get('title', '')}. {r['text']}".strip() for r in corpus]
    del corpus
    qids = [q for q in queries if q in qrels]
    perm = np.random.default_rng(SEED).permutation(len(qids))
    train = [qids[i] for i in perm[: len(qids) // 2]]
    held = [qids[i] for i in perm[len(qids) // 2:]]
    print(f"NQ slice: {len(doc_ids):,} passages, {len(train)} train / {len(held)} held-out queries", flush=True)

    t = time.perf_counter()
    gate, sem, bm25 = EntropyGate(titled), nq.SparseSemanticSpace(titled), nq.BM25(plain)
    docs_tok = [set(tokenize(x)) for x in titled]
    print(f"built in {time.perf_counter() - t:.0f}s, 3-gram index {sem.nbytes() / 1e6:.0f} MB", flush=True)

    def evaluate(qs):
        rows = []
        for q in qs:
            s = 0.65 * nq.nz(gate.score_all(queries[q])) + 0.35 * nq.nz(sem.score_all(queries[q]))
            rows.append(bl.metrics([doc_ids[int(i)] for i in nq.top(s)], qrels[q]))
        return np.array(rows)

    snaps = {0: (dict(gate.learned_gain), sem.gram_gain.copy())}
    tr = evaluate(train)
    out = {"protocol": __doc__.split("usage:")[0].strip(), "passages": len(doc_ids), "train_queries": len(train),
           "held_out_queries": len(held), "laps": {"0": {"train": bl.summary(tr)}}}
    print(f"lap  0                     train nDCG {tr[:, 0].mean():.4f}", flush=True)
    for lap in range(1, LAPS + 1):
        t = time.perf_counter()
        st = nq.finetune(gate, sem, docs_tok, titled, train, queries, qrels, id2row, epochs=1, start_epoch=lap - 1)
        tr = evaluate(train)
        out["laps"][str(lap)] = {"train": bl.summary(tr), "step_lr": round(0.14 * 0.8 ** (lap - 1), 4), **st,
                                 "seconds": round(time.perf_counter() - t, 1)}
        if lap in SNAPSHOTS:
            snaps[lap] = (dict(gate.learned_gain), sem.gram_gain.copy())
        print(f"lap {lap:2d}  viol {st['margin_violations']:4d}  train nDCG {tr[:, 0].mean():.4f}  "
              f"R@10 {tr[:, 1].mean():.4f}  ({time.perf_counter() - t:.0f}s)", flush=True)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)

    print("\nfrozen; held-out half, first look:", flush=True)
    bm = np.array([bl.metrics([doc_ids[int(i)] for i in nq.top(bm25.score_all(queries[q]))], qrels[q]) for q in held])
    test = {}
    for lap in SNAPSHOTS:
        gate.learned_gain, sem.gram_gain = dict(snaps[lap][0]), snaps[lap][1].copy()
        test[lap] = evaluate(held)
        print(f"  after lap {lap:2d}: nDCG {test[lap][:, 0].mean():.4f}  R@10 {test[lap][:, 1].mean():.4f}  "
              f"MRR {test[lap][:, 2].mean():.4f}", flush=True)
    print(f"  bm25        : nDCG {bm[:, 0].mean():.4f}  R@10 {bm[:, 1].mean():.4f}  MRR {bm[:, 2].mean():.4f}", flush=True)
    out["held_out"] = {"bm25": bl.summary(bm), **{f"after_lap{k}": bl.summary(v) for k, v in test.items()},
                       "paired_ndcg": {f"lap{k}_vs_lap{j}": bl.paired_ci(test[j][:, 0], test[k][:, 0])
                                       for j, k in ((0, 1), (0, 3), (0, LAPS), (1, LAPS), (3, LAPS))}}
    out["seconds"] = round(time.perf_counter() - t0, 1)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["held_out"]["paired_ndcg"], indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
