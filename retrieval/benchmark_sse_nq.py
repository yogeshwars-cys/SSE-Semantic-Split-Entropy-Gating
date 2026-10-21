# Copyright (c) 2026 Yogeshwar S (github.com/yogeshwars-cys). All rights reserved.
"""benchmark_sse_nq.py: SSE vs BM25 vs MiniLM / BQ / HaDA on a 100k-passage slice of the HaDA NQ set.

Data: $SSE_NQ_DATA (default data/nq-100k) (bqboost/bench/prep_nq_100k_text.py): every test-relevant passage plus a
seeded sample of the HaDA nq-minilm-384-1m set, with the suite's own MiniLM vectors, 3,452 real NQ test queries.

Nothing here is tuned on NQ: BM25 uses Anserini defaults, fusion weights are the ones used on SciFact.
The only training is the blame-by-diff fine-tune, which learns on a seeded half of the queries and is scored
on the other half (never on the queries it learned from).

    python benchmark_sse_nq.py
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from sse_retrieval import EntropyGate, char_grams, tokenize

DATA = Path(__import__("os").environ.get("SSE_NQ_DATA", "data/nq-100k"))
DIM = 1 << 18      # sparse storage, so a wide hash space costs nothing per document


class SparseSemanticSpace:
    """SSE's 3-gram + word + bigram TF-IDF (semantic_model.py HeaderSpace), stored sparse.
    MT is CSR with one row per hash bucket, so a query reads only its own rows, as in the dense version."""

    def __init__(self, docs, dim=DIM):
        self.N, self.dim = len(docs), dim
        df = np.zeros(dim, np.float32)
        idxs, tfs = [], []
        for d in docs:
            i, c = np.unique(np.asarray(char_grams(d, dim=dim), np.int32), return_counts=True)
            df[i] += 1
            idxs.append(i)
            tfs.append(1.0 + np.log(c.astype(np.float32)))
        self.base_idf = (np.log((self.N + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
        self.gram_gain = np.ones(dim, np.float32)
        indptr = np.zeros(self.N + 1, np.int64)
        indptr[1:] = np.cumsum([len(i) for i in idxs])
        indices = np.concatenate(idxs)
        data = np.empty(len(indices), np.float32)
        for r, (i, tf) in enumerate(zip(idxs, tfs)):
            w = tf * self.base_idf[i]
            data[indptr[r]:indptr[r + 1]] = w / (np.linalg.norm(w) or 1.0)
        del idxs, tfs
        M = sp.csr_matrix((data, indices, indptr), shape=(self.N, dim))
        del data, indices
        self.MT = M.T.tocsr()
        del M

    def weights(self, text):
        i, c = np.unique(np.asarray(char_grams(text, dim=self.dim), np.int32), return_counts=True)
        w = (1.0 + np.log(c.astype(np.float32))) * self.base_idf[i] * self.gram_gain[i]
        return i, w / (np.linalg.norm(w) or 1.0)

    def score_all(self, text):
        i, w = self.weights(text)
        return np.asarray(self.MT[i].T @ w).ravel().astype(np.float32)

    def nbytes(self):
        return int(self.MT.data.nbytes + self.MT.indices.nbytes + self.MT.indptr.nbytes)


class BM25:
    def __init__(self, docs, k1=0.9, b=0.4):
        self.N = len(docs)
        toks = [Counter(tokenize(d)) for d in docs]
        dl = np.array([sum(c.values()) for c in toks], np.float32)
        avg = float(dl.mean())
        df = Counter(t for c in toks for t in c)
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        post = {}
        for i, c in enumerate(toks):
            norm = k1 * (1 - b + b * dl[i] / avg)
            for t, tf in c.items():
                p = post.setdefault(t, ([], []))
                p[0].append(i)
                p[1].append(tf * (k1 + 1) / (tf + norm))
        self.post = {t: (np.array(a, np.int32), np.array(w, np.float32)) for t, (a, w) in post.items()}

    def score_all(self, q):
        s = np.zeros(self.N, np.float32)
        for t in set(tokenize(q)):
            if t in self.post:
                ids, w = self.post[t]
                s[ids] += self.idf[t] * w
        return s


def nz(v):
    m = float(v.max()) if len(v) else 0.0
    return v / m if m > 1e-9 else v


def top(s, k=10):
    k = min(k, len(s))
    i = np.argpartition(-s, k - 1)[:k]
    return i[np.argsort(-s[i], kind="stable")]


def finetune(gate, sem, docs_tok, texts, train, queries, qrels, id2row, epochs=3, lr=0.14, start_epoch=0):
    """SSE blame-by-diff (calibration.py blame + tighten) as an offline trainer: for each training query find the
    hardest wrong document above the best right one; turn down the query terms / 3-gram buckets that matched only
    the wrong one, turn up those that matched only the right one. start_epoch continues an earlier run's lr decay,
    so a training lap is finetune(..., epochs=1, start_epoch=lap - 1)."""
    viol = upd = 0
    for ep in range(start_epoch, start_epoch + epochs):
        step = lr * (0.8 ** ep)
        for q in train:
            pos = [id2row[d] for d in qrels[q] if d in id2row]
            if not pos:
                continue
            qt = queries[q]
            s = 0.65 * nz(gate.score_all(qt)) + 0.35 * nz(sem.score_all(qt))
            best = max(pos, key=lambda r: s[r])
            neg = next((int(c) for c in top(s, 15) if int(c) not in set(pos)), None)
            if neg is None or s[neg] + 0.08 < s[best]:
                continue
            viol += 1
            pt, ntk = docs_tok[best], docs_tok[neg]
            for t in set(tokenize(qt)):
                if len(t) < 3:
                    continue
                if t in ntk and t not in pt:
                    gate.learned_gain[t] = max(0.30, gate.learned_gain.get(t, 1.0) * (1 - step))
                    upd += 1
                elif t in pt and t not in ntk:
                    gate.learned_gain[t] = min(2.40, gate.learned_gain.get(t, 1.0) * (1 + 1.25 * step))
                    upd += 1
            qg = set(char_grams(qt, dim=sem.dim))
            pg, ng = set(char_grams(texts[best], dim=sem.dim)), set(char_grams(texts[neg], dim=sem.dim))
            for g in qg:
                if g in pg and g not in ng:
                    sem.gram_gain[g] = min(2.0, sem.gram_gain[g] * (1 + 0.5 * step))
                elif g in ng and g not in pg:
                    sem.gram_gain[g] = max(0.45, sem.gram_gain[g] * (1 - 0.5 * step))
    return {"margin_violations": viol, "term_updates": upd, "terms_with_gain": len(gate.learned_gain),
            "buckets_changed": int((sem.gram_gain != 1.0).sum())}


def main():
    import bqboost
    from benchmark_bq import METRICS, ROOT, bootstrap_ci, query_metrics, summarize_ms
    t_all = time.perf_counter()
    d = json.loads((DATA / "data.json").read_text(encoding="utf-8"))
    doc_ids, qids, queries = d["corpus_ids"], d["query_ids"], d["queries"]
    qrels = {q: set(v) for q, v in d["qrels"].items()}
    plain = [f"{r['title']} {r['text']}".strip() for r in d["corpus"]]
    titled = [f"{r['title']}. {r['title']}. {r['text']}".strip() for r in d["corpus"]]
    del d
    xb = np.load(DATA / "corpus.npy")
    xq = np.load(DATA / "queries.npy")
    id2row = {x: i for i, x in enumerate(doc_ids)}
    print(f"NQ slice: {len(doc_ids):,} passages, {len(qids):,} test queries", flush=True)

    build = {}
    t = time.perf_counter(); bm25 = BM25(plain); build["bm25"] = time.perf_counter() - t
    t = time.perf_counter(); gate = EntropyGate(titled); build["sse_entropy"] = time.perf_counter() - t
    t = time.perf_counter(); sem = SparseSemanticSpace(titled); build["sse_semantic"] = time.perf_counter() - t
    docs_tok = [set(tokenize(x)) for x in titled]
    backend = "native" if bqboost.available() else "numpy"
    t = time.perf_counter(); codes = bqboost.encode_sign(xb, backend=backend); build["bq_codes"] = time.perf_counter() - t
    from bqboost.born import HaDA
    t = time.perf_counter(); hada = HaDA(xb, recall=0.95); build["hada_r0.95"] = time.perf_counter() - t
    print("built:", {k: round(v, 1) for k, v in build.items()}, f"| SSE 3-gram index {sem.nbytes() / 1e6:.0f} MB",
          f"| hada pipe {hada.describe().get('pipe', hada.describe())}", flush=True)

    rng = np.random.default_rng(20261021)
    perm = rng.permutation(len(qids))
    train = [qids[i] for i in perm[: len(qids) // 2]]
    held = set(qids[i] for i in perm[len(qids) // 2:])

    def run(tag, gate, sem, only=None):
        res, lat = {}, {}

        def add(m, ranked, q, ms):
            res.setdefault(m, []).append(query_metrics([doc_ids[int(i)] for i in ranked], qrels[q]))
            lat.setdefault(m, []).append(ms)

        for q, v in zip(qids, xq):
            if only is not None and q not in only:
                continue
            qt = queries[q]
            t0 = time.perf_counter(); s_bm = bm25.score_all(qt); t_bm = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter(); s_en = gate.score_all(qt); t_en = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter(); s_se = sem.score_all(qt); t_se = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter(); cos = xb @ v; t_de = (time.perf_counter() - t0) * 1e3
            qc = bqboost.encode_sign(v, backend=backend)
            t0 = time.perf_counter(); h200, hd = bqboost.search_hamming(codes, qc, 200, backend=backend)
            t_h = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter(); hc, _ = hada.search(v, k=100); t_ha = (time.perf_counter() - t0) * 1e3
            sse = 0.65 * nz(s_en) + 0.35 * nz(s_se)
            if tag == "base":
                add("bm25", top(s_bm), q, t_bm)
                add("dense_minilm (exact)", top(cos), q, t_de)
                add("bq_sign_hamming", h200[:10], q, t_h)
                r = bqboost.rescore_dot(xb, v, h200[:100].astype(np.uint32), 10, backend=backend)[0]
                add("bq_sign_rescore100", r, q, t_h)
                add("hada_r0.95", hc[:10], q, t_ha)
                add("sse_entropy_gate", top(s_en), q, t_en)
                add("sse_semantic_3gram", top(s_se), q, t_se)
                add("dense + bm25", top(0.62 * np.maximum(0, cos) + 0.38 * nz(s_bm)), q, t_de + t_bm)
                add("dense + sse_semantic", top(0.62 * np.maximum(0, cos) + 0.38 * nz(s_se)), q, t_de + t_se)
                hcs = np.maximum(0, xb[hc] @ v)
                add("hada top100 -> + bm25", hc[top(0.62 * hcs + 0.38 * nz(s_bm[hc]))], q, t_ha + t_bm)
                add("hada top100 -> + sse_semantic", hc[top(0.62 * hcs + 0.38 * nz(s_se[hc]))], q, t_ha + t_se)
                hs = nz(1.0 - hd.astype(np.float32) / 384.0)
                add("bq top200 -> + sse (no float rows)",
                    h200[top(0.45 * hs + 0.38 * nz(s_en[h200]) + 0.17 * nz(s_se[h200]))], q, t_h + t_en + t_se)
            add(f"sse_gate+semantic [{tag}]", top(sse), q, t_en + t_se)
            add(f"dense + sse_gate+semantic [{tag}]", top(0.62 * np.maximum(0, cos) + 0.38 * nz(sse)), q,
                t_de + t_en + t_se)
            add(f"bm25 [{tag}]", top(s_bm), q, t_bm)
            add(f"dense + bm25 [{tag}]", top(0.62 * np.maximum(0, cos) + 0.38 * nz(s_bm)), q, t_de + t_bm)
        return res, lat

    t = time.perf_counter()
    base, base_lat = run("base", gate, sem)
    print(f"base pass {time.perf_counter() - t:.0f}s", flush=True)
    held_before, _ = run("held-out, before fine-tune", gate, sem, only=held)
    t = time.perf_counter()
    ft = finetune(gate, sem, docs_tok, titled, train, queries, qrels, id2row)
    ft["seconds"] = round(time.perf_counter() - t, 1)
    ft["train_queries"], ft["held_out_queries"] = len(train), len(held)
    print("fine-tune:", ft, flush=True)
    held_after, held_lat = run("held-out, after fine-tune", gate, sem, only=held)

    def table(res, lat, ref, title):
        refv = np.array(res[ref])[:, -1]
        print(f"\n{title}\n{'method':<44} | R@1    | R@10   | MRR@10 | nDCG@10 | vs {ref} (paired 95% CI)       | p50 ms")
        out = {}
        for m, rows in res.items():
            a = np.array(rows)
            dlt = a[:, -1] - refv
            lo, hi = bootstrap_ci(dlt)
            p50 = summarize_ms(lat[m])["p50"] if m in lat else float("nan")
            print(f"{m:<44} | {a[:,0].mean():.4f} | {a[:,2].mean():.4f} | {a[:,3].mean():.4f} | {a[:,4].mean():.4f}  "
                  f"| {dlt.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] | {p50:7.2f}")
            out[m] = {"metrics": dict(zip(METRICS, a.mean(0).tolist())), "delta_vs_ref": float(dlt.mean()),
                      "delta_ci": [lo, hi], "p50_ms": p50}
        return out

    keep = {m: v for m, v in base.items() if "[" not in m}
    report = {"data": str(DATA), "passages": len(doc_ids), "queries": len(qids), "build_s": build,
              "sse_semantic_index_mb": sem.nbytes() / 1e6, "hada": hada.describe(), "fine_tune": ft}
    report["all_queries"] = table(keep, base_lat, "bm25", f"ALL {len(qids)} NQ test queries (nothing tuned on NQ)")
    merged = {**held_before, **held_after}
    merged_lat = {m: [0.0] for m in merged}
    report["held_out"] = table(merged, merged_lat, "bm25 [held-out, before fine-tune]",
                               f"HELD-OUT half ({len(held)} queries): before vs after blame-by-diff fine-tune "
                               f"on the other {len(train)}")
    out = ROOT / "benchmark_results" / "sse" / "sse_nq100k_results.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {out} ({time.perf_counter() - t_all:.0f}s total)")


if __name__ == "__main__":
    main()
