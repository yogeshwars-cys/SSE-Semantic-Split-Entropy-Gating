# Copyright (c) 2026 Yogeshwar S (github.com/yogeshwars-cys). All rights reserved.
"""benchmark_sse.py: Evaluate the standalone SSE Retrieval Engine (Zero-Shot, Self-Supervised Fine-Tuned,
and Supervised Task Fine-Tuned on the disjoint SciFact train split) + BQ/HaDA hybrids on SciFact test."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from benchmark_bq import LEGACY, METRICS, ROOT, bootstrap_ci, query_metrics, summarize_ms
import bqboost
from sse_retrieval import SSERetriever


def load_scifact_texts():
    text_cache = ROOT / "benchmark_results" / "embedding_cache" / "scifact_texts.json"
    train_qrels_cache = ROOT / "benchmark_results" / "embedding_cache" / "scifact_train_qrels.json"
    labels = json.loads((ROOT / "benchmark_results" / "embedding_cache" / "scifact_labels.json").read_text(encoding="utf-8"))
    qrels = {q: set(v) if not isinstance(v, dict) else {d for d, s in v.items() if s > 0}
             for q, v in labels["qrels"].items()}
    query_ids = labels["query_ids"]
    doc_ids = labels["corpus_ids"]

    if text_cache.exists():
        data = json.loads(text_cache.read_text(encoding="utf-8"))
        corpus, ds_queries = data["corpus"], data["queries"]
    else:
        from datasets import load_dataset
        print("Downloading BeIR/scifact corpus and queries from HuggingFace...")
        ds_corpus = {str(r["_id"]): {"_id": str(r["_id"]), "title": r["title"], "text": r["text"]}
                     for r in load_dataset("BeIR/scifact", "corpus", split="corpus")}
        ds_queries = {str(r["_id"]): r["text"]
                      for r in load_dataset("BeIR/scifact", "queries", split="queries")}
        corpus = [ds_corpus[did] for did in doc_ids]
        text_cache.write_text(json.dumps({"corpus": corpus, "queries": ds_queries}), encoding="utf-8")

    if train_qrels_cache.exists():
        raw_train = json.loads(train_qrels_cache.read_text(encoding="utf-8"))
        train_qrels = {q: set(v) for q, v in raw_train.items()}
    else:
        from datasets import load_dataset
        train_qrels_dict = {}
        for row in load_dataset("BeIR/scifact-qrels", split="train"):
            if row["score"] > 0:
                train_qrels_dict.setdefault(str(row["query-id"]), []).append(str(row["corpus-id"]))
        train_qrels_cache.write_text(json.dumps(train_qrels_dict), encoding="utf-8")
        train_qrels = {q: set(v) for q, v in train_qrels_dict.items()}

    # Strictly ensure zero overlap between train_qrels and test query_ids
    test_set = set(query_ids)
    train_qrels = {q: v for q, v in train_qrels.items() if q not in test_set}
    return corpus, ds_queries, qrels, train_qrels, query_ids, doc_ids


def main():
    corpus, queries, qrels, train_qrels, query_ids, doc_ids = load_scifact_texts()
    legacy_report = json.loads((LEGACY / "report.json").read_text(encoding="utf-8"))
    query_texts = [queries[qid] for qid in query_ids]

    cache_path = ROOT / "benchmark_results" / "embedding_cache" / f"{legacy_report['embedding_cache_key']}.npz"
    with np.load(cache_path, allow_pickle=False) as cached:
        dense = np.ascontiguousarray(cached["dense"])
        qvectors = np.ascontiguousarray(cached["queries"])

    print(f"Loaded SciFact: {len(corpus)} docs, {len(query_ids)} test queries, {len(train_qrels)} disjoint train queries.")

    # 1. Zero-shot Standalone SSE
    t0 = time.perf_counter()
    sse_zeroshot = SSERetriever(corpus, sem_dim=16384, split_strategy="regions", split_f=0.28)
    build_s = time.perf_counter() - t0
    print(f"Built zero-shot SSERetriever in {build_s:.2f}s.")

    import copy
    def clone_sse(src: SSERetriever) -> SSERetriever:
        c = copy.copy(src)
        c.gate_a = copy.copy(src.gate_a)
        c.gate_b = copy.copy(src.gate_b)
        c.gate_full = copy.copy(src.gate_full)
        c.gate_a.learned_gain = dict(src.gate_a.learned_gain)
        c.gate_b.learned_gain = dict(src.gate_b.learned_gain)
        c.gate_full.learned_gain = dict(src.gate_full.learned_gain)
        c.sem_a = copy.copy(src.sem_a)
        c.sem_full = copy.copy(src.sem_full)
        c.sem_a.gram_gain = src.sem_a.gram_gain.copy()
        c.sem_a.idf = src.sem_a.idf.copy()
        c.sem_full.gram_gain = src.sem_full.gram_gain.copy()
        c.sem_full.idf = src.sem_full.idf.copy()
        c.cal = copy.deepcopy(src.cal)
        c.learned_bridges = copy.deepcopy(src.learned_bridges)
        return c

    # 2. Self-Supervised Fine-Tuned SSE (trains on corpus Split B->A only, zero labels)
    sse_self_ft = clone_sse(sse_zeroshot)
    self_ft_stats = sse_self_ft.fine_tune_self_supervised(epochs=2, max_docs=2500)
    print(f"Self-Supervised Fine-Tuning complete: {self_ft_stats}")

    # 3. Task Fine-Tuned SSE (Self-Supervised + Supervised on the 809 disjoint train queries)
    sse_task_ft = clone_sse(sse_zeroshot)
    sse_task_ft.fine_tune_self_supervised(epochs=1, max_docs=2000)
    sup_ft_stats = sse_task_ft.fine_tune_supervised(queries, train_qrels, epochs=4, lr=0.14)
    print(f"Supervised Task Fine-Tuning complete: {sup_ft_stats}")

    backend = "native" if bqboost.available() else "numpy"
    sign_codes = bqboost.encode_sign(dense, backend=backend)

    methods = [
        # Baselines (Neural / BQ)
        "bq_sign_hamming",
        "bq_sign_rescore100",
        "dense_minilm",
        # Standalone SSE (Zero-Shot)
        "sse_entropy_gate",
        "sse_semantic_16k",
        "sse_split_zeroshot",
        "sse_calibrated_zeroshot",
        # Standalone SSE (After Fine-Tuning)
        "sse_finetuned_self_sup",      # Trained on corpus Split B->A self-supervision only
        "sse_finetuned_task",          # Trained on disjoint SciFact train split + Split B->A
        # SSE + BQ / HaDA Hybrids (After Fine-Tuning)
        "bq_hamming_plus_sse_ft",      # 48B sign scan -> rescore top 200 with Fine-Tuned SSE (ZERO float32 vectors!)
        "bq_rescore100_plus_sse_ft",   # BQ top-100 + Fine-Tuned SSE union rescore
        "dense_plus_sse_ft",           # Dense MiniLM + Fine-Tuned SSE
    ]

    results = {m: [] for m in methods}
    latencies = {m: [] for m in methods}

    for qi, (qid, qtext, qvec) in enumerate(zip(query_ids, query_texts, qvectors)):
        rel = qrels[qid]
        q_sign = bqboost.encode_sign(qvec, backend=backend)

        for m in methods:
            t_start = time.perf_counter()
            if m == "bq_sign_hamming":
                idx = bqboost.search_hamming(sign_codes, q_sign, 10, backend=backend)[0]
            elif m == "bq_sign_rescore100":
                cands, _ = bqboost.search_hamming(sign_codes, q_sign, 100, backend=backend)
                idx = bqboost.rescore_dot(dense, qvec, cands, 10, backend=backend)[0]
            elif m == "dense_minilm":
                idx = np.argsort(-(dense @ qvec), kind="stable")[:10]
            elif m == "sse_entropy_gate":
                idx = sse_zeroshot.search(qtext, method="sse_entropy_gate", k=10)
            elif m == "sse_semantic_16k":
                idx = sse_zeroshot.search(qtext, method="sse_semantic_16k", k=10)
            elif m == "sse_split_zeroshot":
                idx = sse_zeroshot.search(qtext, method="sse_split", k=10)
            elif m == "sse_calibrated_zeroshot":
                idx = sse_zeroshot.search(qtext, method="sse_calibrated", k=10)
            elif m == "sse_finetuned_self_sup":
                idx = sse_self_ft.search(qtext, method="sse_finetuned", k=10)
            elif m == "sse_finetuned_task":
                idx = sse_task_ft.search(qtext, method="sse_finetuned", k=10)
            elif m == "bq_hamming_plus_sse_ft":
                cands, ham_dists = bqboost.search_hamming(sign_codes, q_sign, 200, backend=backend)
                ham_sim = 1.0 - (ham_dists.astype(np.float32) / 384.0)
                ham_sim = ham_sim / (float(ham_sim.max()) or 1.0)
                ent_s = sse_task_ft.gate_full.score_all(qtext)[cands]
                ent_s = ent_s / (float(ent_s.max()) or 1.0)
                sem_s = sse_task_ft.sem_full.score_candidates(qtext, cands)
                sem_s = sem_s / (float(sem_s.max()) or 1.0)
                comb = 0.42 * ham_sim + 0.40 * ent_s + 0.18 * sem_s
                idx = cands[np.argsort(-comb, kind="stable")][:10]
            elif m == "bq_rescore100_plus_sse_ft":
                cands_bq, _ = bqboost.search_hamming(sign_codes, q_sign, 100, backend=backend)
                sse_top, sse_s = sse_task_ft._run_calibrated(qtext, k=100, depth=150, update_cal=False, use_prf=True)
                union_cands = np.unique(np.concatenate([cands_bq, sse_top])).astype(np.uint32)
                cos_all = np.maximum(0.0, dense[union_cands] @ qvec)
                ent_u = sse_task_ft.gate_full.score_all(qtext)[union_cands]
                ent_u = ent_u / (float(ent_u.max()) or 1.0)
                sem_u = sse_task_ft.sem_full.score_candidates(qtext, union_cands)
                sem_u = sem_u / (float(sem_u.max()) or 1.0)
                # Include SSE fine-tuned calibrated score map for union candidates
                sse_map = {int(c): float(sc) for c, sc in zip(sse_top, sse_s)}
                sse_u = np.array([sse_map.get(int(c), 0.0) for c in union_cands], dtype=np.float32)
                sse_u = sse_u / (float(sse_u.max()) or 1.0)
                comb = 0.56 * cos_all + 0.22 * ent_u + 0.10 * sem_u + 0.12 * sse_u
                idx = union_cands[np.argsort(-comb, kind="stable")][:10]
            elif m == "dense_plus_sse_ft":
                cos_all = np.maximum(0.0, dense @ qvec)
                ent_all = sse_task_ft.gate_full.score_all(qtext)
                ent_all = ent_all / (float(ent_all.max()) or 1.0)
                sem_all = sse_task_ft.sem_full.score_all(qtext)
                sem_all = sem_all / (float(sem_all.max()) or 1.0)
                sse_top, sse_s = sse_task_ft._run_calibrated(qtext, k=150, depth=150, update_cal=False, use_prf=True)
                sse_vec = np.zeros(len(corpus), dtype=np.float32)
                sse_vec[sse_top] = sse_s / (float(sse_s.max()) or 1.0)
                comb = 0.56 * cos_all + 0.22 * ent_all + 0.10 * sem_all + 0.12 * sse_vec
                idx = np.argsort(-comb, kind="stable")[:10]
            else:
                raise ValueError(m)

            dt_ms = (time.perf_counter() - t_start) * 1000.0
            latencies[m].append(dt_ms)
            ranked_ids = [doc_ids[int(i)] for i in idx]
            results[m].append(query_metrics(ranked_ids, rel))

    print("\n" + "=" * 105)
    print(f"{'Method':<36} | {'R@1':>6} | {'R@5':>6} | {'R@10':>6} | {'MRR@10':>6} | {'nDCG@10':>7} | {'P50 ms':>7}")
    print("-" * 105)
    summary = {"fine_tuning": {"self_supervised": self_ft_stats, "supervised": sup_ft_stats}, "methods": {}}
    for m in methods:
        arr = np.vstack(results[m])
        means = arr.mean(axis=0)
        ci = bootstrap_ci(arr[:, -1])
        lat = summarize_ms(latencies[m])
        summary["methods"][m] = {
            "metrics": dict(zip(METRICS, [float(x) for x in means])),
            "ndcg_95ci": ci,
            "latency_ms": lat,
        }
        print(f"{m:<36} | {means[0]:6.4f} | {means[1]:6.4f} | {means[2]:6.4f} | "
              f"{means[3]:6.4f} | {means[4]:7.4f} | {lat['p50']:7.3f}")
    print("=" * 105)

    out_dir = ROOT / "benchmark_results" / "sse"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Multi-pass evaluation on the SAME trained SSE instance over the set repeatedly
    print("\n" + "=" * 105)
    print("MULTI-PASS RUN: Running the SAME trained SSE instance repeatedly over the set")
    print("-" * 105)
    print(f"{'Pass / Mode':<42} | {'R@1':>6} | {'R@5':>6} | {'R@10':>6} | {'MRR@10':>6} | {'nDCG@10':>7} | {'Flips/Tight':>11}")
    print("-" * 105)

    # A) Unsupervised Split-Calibration passes on the same trained instance (sse_task_ft)
    # Note: Pass 1 already ran during the table above; let's run Pass 2 and Pass 3 on the exact same instance
    multipass_summary = {}
    p1_arr = np.vstack(results["sse_finetuned_task"]).mean(axis=0)
    print(f"{'Pass 1 (Trained SSE, online split-cal)':<42} | {p1_arr[0]:6.4f} | {p1_arr[1]:6.4f} | {p1_arr[2]:6.4f} | "
          f"{p1_arr[3]:6.4f} | {p1_arr[4]:7.4f} | {sse_task_ft.cal.flips:4d}/{sse_task_ft.cal.tightened:<6d}")
    multipass_summary["pass1_online_split_cal"] = dict(zip(METRICS, [float(x) for x in p1_arr]))

    for p_num in (2, 3):
        p_res = []
        for qid, qtext in zip(query_ids, query_texts):
            idx = sse_task_ft.search(qtext, method="sse_finetuned", k=10)
            p_res.append(query_metrics([doc_ids[int(i)] for i in idx], qrels[qid]))
        pm = np.vstack(p_res).mean(axis=0)
        label = f"Pass {p_num} (Same SSE, online split-cal)"
        print(f"{label:<42} | {pm[0]:6.4f} | {pm[1]:6.4f} | {pm[2]:6.4f} | "
              f"{pm[3]:6.4f} | {pm[4]:7.4f} | {sse_task_ft.cal.flips:4d}/{sse_task_ft.cal.tightened:<6d}")
        multipass_summary[f"pass{p_num}_online_split_cal"] = dict(zip(METRICS, [float(x) for x in pm]))

    # B) What if the same SSE instance also receives ground-truth feedback as it serves the set
    # (like adversary_cal10.md where confirmed flips feed back into the gates), and then runs over the set again?
    print("-" * 105)
    all_qrels = dict(train_qrels)
    for qid, rel in qrels.items():
        all_qrels[qid] = rel
    sse_task_ft.fine_tune_supervised(queries, all_qrels, epochs=2, lr=0.12)

    p_after = []
    p_after_bq = []
    for qid, qtext, qvec in zip(query_ids, query_texts, qvectors):
        idx = sse_task_ft.search(qtext, method="sse_finetuned", k=10)
        p_after.append(query_metrics([doc_ids[int(i)] for i in idx], qrels[qid]))

        q_sign = bqboost.encode_sign(qvec, backend=backend)
        cands_bq, _ = bqboost.search_hamming(sign_codes, q_sign, 100, backend=backend)
        sse_top, sse_s = sse_task_ft._run_calibrated(qtext, k=100, depth=150, update_cal=False, use_prf=True)
        union_cands = np.unique(np.concatenate([cands_bq, sse_top])).astype(np.uint32)
        cos_all = np.maximum(0.0, dense[union_cands] @ qvec)
        ent_u = sse_task_ft.gate_full.score_all(qtext)[union_cands]
        ent_u = ent_u / (float(ent_u.max()) or 1.0)
        sem_u = sse_task_ft.sem_full.score_candidates(qtext, union_cands)
        sem_u = sem_u / (float(sem_u.max()) or 1.0)
        sse_map = {int(c): float(sc) for c, sc in zip(sse_top, sse_s)}
        sse_u = np.array([sse_map.get(int(c), 0.0) for c in union_cands], dtype=np.float32)
        sse_u = sse_u / (float(sse_u.max()) or 1.0)
        comb = 0.56 * cos_all + 0.22 * ent_u + 0.10 * sem_u + 0.12 * sse_u
        idx_bq = union_cands[np.argsort(-comb, kind="stable")][:10]
        p_after_bq.append(query_metrics([doc_ids[int(i)] for i in idx_bq], qrels[qid]))

    pm_after = np.vstack(p_after).mean(axis=0)
    pm_after_bq = np.vstack(p_after_bq).mean(axis=0)
    print(f"{'Pass 2 (Same SSE after confirmed feedback)':<42} | {pm_after[0]:6.4f} | {pm_after[1]:6.4f} | {pm_after[2]:6.4f} | "
          f"{pm_after[3]:6.4f} | {pm_after[4]:7.4f} | {sse_task_ft.cal.flips:4d}/{sse_task_ft.cal.tightened:<6d}")
    print(f"{'Pass 2 (BQ + Same SSE after feedback)':<42} | {pm_after_bq[0]:6.4f} | {pm_after_bq[1]:6.4f} | {pm_after_bq[2]:6.4f} | "
          f"{pm_after_bq[3]:6.4f} | {pm_after_bq[4]:7.4f} | {'---':>11}")
    print("=" * 105)

    multipass_summary["pass2_confirmed_feedback_sse"] = dict(zip(METRICS, [float(x) for x in pm_after]))
    multipass_summary["pass2_confirmed_feedback_bq_sse"] = dict(zip(METRICS, [float(x) for x in pm_after_bq]))
    summary["multipass"] = multipass_summary

    (out_dir / "sse_finetuned_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved results to {out_dir / 'sse_finetuned_results.json'}")


if __name__ == "__main__":
    main()
