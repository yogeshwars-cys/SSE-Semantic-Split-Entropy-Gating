"""benchmark_sse_laps.py: conveyor belt for the retrieval port. Does lapping the SciFact train set make SSE better
on the train set, and does any of it carry to the held-out test set?

  lap 0   zero-shot SSE + the self-supervised split B->A pass (as benchmark_sse.py's task recipe starts)
  lap k   one supervised pass over every train query (fine_tune_supervised, epochs=1, start_epoch=k-1), then the
          stage mixing weights are re-tuned on the train subset. The pass's updates use only the gate/semantic
          blend, never the mixing weights, so lap 4 ends in exactly the state of the shipped 4-epoch recipe.
Two schedules on the same belt:
  continued   lr 0.14 * 0.8^(k-1): the shipped schedule carried on past epoch 4 (steps shrink every lap)
  constant    lr 0.14 every lap: does the belt keep moving when the steps do not shrink?
After every lap both sets are scored frozen (no online split calibration), so the numbers show only what training
changed. Train = 809 SciFact train queries (disjoint from test), test = the 300 BEIR test queries.
The repo's audit rule applies: train-set laps measure fit to queries already trained on; only test generalises.

usage: python -I benchmark_sse_laps.py <beir scifact dir> <out.json>
"""
import copy
import json
import math
import os
import sys
import time

import numpy as np

DATA, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir
from sse_retrieval import SSERetriever  # noqa: E402

LAPS = int(os.environ.get("IR_LAPS", 20))
LR = 0.14
SCHEDULES = ("continued", "constant")


def load():
    corpus = [json.loads(l) for l in open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8")]
    corpus = [{"_id": str(r["_id"]), "title": r.get("title", ""), "text": r.get("text", "")} for r in corpus]
    queries = {str(r["_id"]): r["text"] for r in map(json.loads, open(os.path.join(DATA, "queries.jsonl"), encoding="utf-8"))}

    def qrels(split):
        out = {}
        for line in open(os.path.join(DATA, "qrels", f"{split}.tsv"), encoding="utf-8").read().splitlines()[1:]:
            q, d, s = line.split("\t")
            if int(s) > 0:
                out.setdefault(q, set()).add(d)
        return out
    test = qrels("test")
    train = {q: v for q, v in qrels("train").items() if q not in test}
    return corpus, queries, train, test


def metrics(ranked, rel):
    """nDCG@10 (binary gains), Recall@10, MRR@10 for one query."""
    hits = [i for i, d in enumerate(ranked[:10]) if d in rel]
    dcg = sum(1.0 / math.log2(i + 2) for i in hits)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(10, len(rel))))
    return dcg / idcg, len(hits) / len(rel), 1.0 / (hits[0] + 1) if hits else 0.0


def evaluate(sse, queries, qrels):
    """Per-query (nDCG@10, R@10, MRR@10), frozen: the online calibrator neither reads nor learns anything."""
    rows = []
    for q, rel in qrels.items():
        idx, _ = sse._run_calibrated(queries[q], k=10, depth=150, update_cal=False, use_prf=True)
        rows.append(metrics([sse.doc_ids[int(i)] for i in idx], rel))
    return np.array(rows)


def summary(arr):
    return {"ndcg10": float(arr[:, 0].mean()), "recall10": float(arr[:, 1].mean()), "mrr10": float(arr[:, 2].mean())}


def paired_ci(a, b, n=2000, seed=0):
    """Mean of b - a over queries, with a 95% paired bootstrap interval."""
    d = b - a
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return {"delta": round(float(d.mean()), 4), "ci95": [round(float(x), 4) for x in np.quantile(boots, [0.025, 0.975])]}


def state(sse):
    g, s = sse.gate_full.learned_gain, sse.sem_full.gram_gain
    return {"terms_with_gain": len(g), "terms_at_floor": sum(v <= 0.30 + 1e-9 for v in g.values()),
            "terms_at_cap": sum(v >= 2.40 - 1e-9 for v in g.values()),
            "grams_moved": int((s != 1.0).sum()), "bridge_anchors": len(sse.learned_bridges),
            "mix": [sse.mix_stage_a, sse.mix_verify, sse.prf_weight]}


def main():
    t0 = time.perf_counter()
    corpus, queries, train, test = load()
    print(f"SciFact: {len(corpus)} docs, {len(train)} train queries, {len(test)} test queries", flush=True)
    base = SSERetriever(corpus, sem_dim=16384, split_strategy="regions", split_f=0.28)
    print(f"built in {time.perf_counter() - t0:.0f}s", flush=True)
    zs_tr, zs_te = evaluate(base, queries, train), evaluate(base, queries, test)
    out = {"protocol": __doc__.split("usage:")[0].strip(), "train_queries": len(train), "test_queries": len(test),
           "zero_shot": {"train": summary(zs_tr), "test": summary(zs_te)}}
    print(f"zero-shot       train nDCG {zs_tr[:, 0].mean():.4f}  test nDCG {zs_te[:, 0].mean():.4f}", flush=True)
    base.fine_tune_self_supervised(epochs=1, max_docs=2000)

    for sched in SCHEDULES:
        sse = copy.deepcopy(base)
        laps, per_query = {}, {}
        tr, te = evaluate(sse, queries, train), evaluate(sse, queries, test)
        laps["0"] = {"train": summary(tr), "test": summary(te)}
        per_query[0] = (tr, te)
        print(f"\n== {sched}\nlap  0          train nDCG {tr[:, 0].mean():.4f}  test nDCG {te[:, 0].mean():.4f}", flush=True)
        for lap in range(1, LAPS + 1):
            t = time.perf_counter()
            lr = LR if sched == "continued" else LR / 0.8 ** (lap - 1)   # constant: undo the 0.8^epoch decay
            st = sse.fine_tune_supervised(queries, train, epochs=1, lr=lr, start_epoch=lap - 1)
            tr, te = evaluate(sse, queries, train), evaluate(sse, queries, test)
            per_query[lap] = (tr, te)
            laps[str(lap)] = {"train": summary(tr), "test": summary(te), "step_lr": round(LR * 0.8 ** (lap - 1) if sched == "continued" else LR, 4),
                              "margin_violations": st["margin_violations_fixed"], "term_updates": st["term_gain_updates"],
                              "train_subset_ndcg10_mix_tuning": st["train_subset_ndcg10"], "state": state(sse),
                              "seconds": round(time.perf_counter() - t, 1)}
            print(f"lap {lap:2d}  viol {st['margin_violations_fixed']:4d}  train nDCG {tr[:, 0].mean():.4f}  "
                  f"test nDCG {te[:, 0].mean():.4f}  R@10 {te[:, 1].mean():.4f}  ({time.perf_counter() - t:.0f}s)", flush=True)
        prev = None
        for lap in range(1, LAPS + 1):                        # belt.py's stop rule on margin violations
            v = laps[str(lap)]["margin_violations"]
            if prev is not None and prev - v < 0.01 * prev:
                laps["belt_rule_stops_at_lap"] = lap
                break
            prev = v
        best = max(range(LAPS + 1), key=lambda k: laps[str(k)]["train"]["ndcg10"])
        out[sched] = {"laps": laps, "best_train_lap": best, "paired_vs_lap1": {
            f"lap{k}": {"train": paired_ci(per_query[1][0][:, 0], per_query[k][0][:, 0]),
                        "test": paired_ci(per_query[1][1][:, 0], per_query[k][1][:, 0])}
            for k in sorted({min(4, LAPS), best, LAPS} - {0, 1})}}
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
