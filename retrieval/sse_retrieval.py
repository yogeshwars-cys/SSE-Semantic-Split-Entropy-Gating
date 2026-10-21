# Copyright (c) 2026 Yogeshwar S (github.com/yogeshwars-cys). All rights reserved.
"""sse_retrieval.py: Standalone SSE (Semantic Split Entropy Gating) adapted for Information Retrieval,
with offline Self-Supervised and Supervised Task Fine-Tuning (`/__train` + adversarial blame loop).

Direct architectural translation of sse/:
  1. split_lib.py     -> split_doc(): cuts each document into Part A (critical regions: title, salient
                         acronyms/numbers/claims, and prefix) and Part B (bulk body).
  2. anomaly_model.py -> EntropyGate: information-theoretic surprisal (-log2((df+0.1)/(N+0.2))),
                         Shannon character entropy, and length z-score per term/document.
  3. semantic_model.py-> SemanticSpace: hashed character 3-gram + word TF-IDF with contiguous
                         transposed matrix MT (shape: DIM x N_docs) so a query touches only MT[idx].
  4. calibration.py   -> RetrievalCalibrator + fine_tune(): split-verification is the ground truth.
                         Offline fine-tuning mines hard-negative decoys (adversaries), blames the exact
                         terms and 3-gram buckets that let the decoy beat the ground-truth document,
                         updates per-term surprisal & 3-gram IDF weights, learns co-occurrence bridge
                         associations, and tunes stage mixing weights.
"""
from __future__ import annotations

import math
import re
import statistics
import time
import zlib
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
CRIT_SENT_RE = re.compile(
    r"[^.!?]*(?:\b[A-Z]{2,}[0-9a-zA-Z-]*\b|\b\d+(?:\.\d+)?%?|"
    r"\b(?:increase|decrease|cause|induce|inhibit|reduce|require|associate|significant|effect|prevent|protect|risk)[a-z]*\b)"
    r"[^.!?]*[.!?]?"
)


# ---------------------------------------------------------------------------
# 1. split_lib.py equivalent: Split-Document Representation (A = critical, B = bulk)
# ---------------------------------------------------------------------------

def split_doc(title: str, text: str, strategy: str = "regions", f: float = 0.28) -> Tuple[str, str]:
    """Split a document into Part A (verified/scanned first) and Part B (bulk body)."""
    full = f"{title}. {text}".strip()
    if strategy == "none":
        return full, ""
    cut = max(64, int(len(text) * f))
    prefix = text[:cut]
    rest = text[cut:]
    if strategy == "prefix":
        return f"{title}. {prefix}", rest
    crit_matches = CRIT_SENT_RE.findall(rest)
    crit = " ".join(crit_matches[:3]) if crit_matches else ""
    if strategy in ("critical", "regions"):
        part_a = f"{title}. {prefix} {crit}".strip()
        return part_a, rest
    raise ValueError(f"Unknown split strategy: {strategy}")


# ---------------------------------------------------------------------------
# 2. anomaly_model.py equivalent: Information-Theory & Surprisal Gate
# ---------------------------------------------------------------------------

def shannon(s: str) -> float:
    """Character-level Shannon entropy in bits (from anomaly_model.py)."""
    if not s:
        return 0.0
    n = len(s)
    c = Counter(s)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class EntropyGate:
    """Information-theoretic lexical scorer and query-document anomaly gate.

    Ports Baseline + Gate from anomaly_model.py:
      - surprisal(df) = -log2((df + 0.1) / (N + 0.2))
      - per-document length & Shannon entropy statistics
      - inverted index over surprisal-weighted terms for sub-millisecond scoring
      - learnable per-term gain `learned_gain[term]` for task fine-tuning
    """

    def __init__(self, docs: List[str], k1: float = 1.2, b: float = 0.75):
        self.N = len(docs)
        self.k1, self.b = k1, b
        self.df: Counter = Counter()
        self.doc_lens = np.empty(self.N, dtype=np.float32)
        self.doc_ents = np.empty(self.N, dtype=np.float32)

        tokenized_docs: List[Counter] = []
        for i, d in enumerate(docs):
            toks = tokenize(d)
            counts = Counter(toks)
            tokenized_docs.append(counts)
            self.doc_lens[i] = max(len(toks), 1)
            self.doc_ents[i] = shannon(d[:512])
            for t in counts:
                self.df[t] += 1

        self.avgdl = float(self.doc_lens.mean()) or 1.0
        self.std_dl = float(self.doc_lens.std()) or 1.0
        self.mean_ent = float(self.doc_ents.mean()) or 1.0
        self.std_ent = max(float(self.doc_ents.std()), 0.1)

        # Surprisal table: -log2((df + 0.1) / (N + 0.2)) as in anomaly_model.py:36
        self.surprisal_map: Dict[str, float] = {
            t: -math.log2((cnt + 0.1) / (self.N + 0.2)) for t, cnt in self.df.items()
        }
        self.max_surprisal = -math.log2(0.1 / (self.N + 0.2))
        vals = list(self.surprisal_map.values()) or [1.0]
        self.mu_s = statistics.mean(vals)
        self.sd_s = statistics.pstdev(vals) or 1.0

        # Learnable per-term gain (initialized to 1.0, tuned during fine-tuning)
        self.learned_gain: Dict[str, float] = {}
        # Precomputed base info weight per term
        self.base_weight: Dict[str, float] = {}
        for t, s in self.surprisal_map.items():
            self.base_weight[t] = s * (1.0 + 0.15 * max(0.0, (s - self.mu_s) / self.sd_s))

        # Build compact CSR-like postings for fast surprisal-BM25 scoring
        self.postings: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        tmp_docs: Dict[str, List[int]] = {}
        tmp_tfs: Dict[str, List[float]] = {}
        for doc_id, counts in enumerate(tokenized_docs):
            dl_norm = self.k1 * (1.0 - self.b + self.b * (self.doc_lens[doc_id] / self.avgdl))
            for t, tf in counts.items():
                norm_tf = (tf * (self.k1 + 1.0)) / (tf + dl_norm)
                tmp_docs.setdefault(t, []).append(doc_id)
                tmp_tfs.setdefault(t, []).append(norm_tf)
        for t in tmp_docs:
            self.postings[t] = (
                np.asarray(tmp_docs[t], dtype=np.int32),
                np.asarray(tmp_tfs[t], dtype=np.float32),
            )

    def surprisal(self, term: str) -> float:
        return self.surprisal_map.get(term, self.max_surprisal)

    def effective_weight(self, term: str) -> float:
        return self.base_weight.get(term, self.max_surprisal) * self.learned_gain.get(term, 1.0)

    def score_all(self, q_text: str, term_weights: Optional[Dict[str, float]] = None,
                  extra_terms: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Score all N documents using information-theoretic surprisal + learned fine-tuned gains."""
        q_counts = Counter(tokenize(q_text))
        scores = np.zeros(self.N, dtype=np.float32)
        if not q_counts and not extra_terms:
            return scores
        for t, qtf in q_counts.items():
            post = self.postings.get(t)
            if post is None:
                continue
            doc_ids, norm_tfs = post
            w = term_weights.get(t, 1.0) if term_weights else 1.0
            if w <= 0.0:
                continue
            info_w = self.base_weight[t] * self.learned_gain.get(t, 1.0) * w
            scores[doc_ids] += np.float32(info_w * (1.0 + 0.25 * math.log1p(qtf))) * norm_tfs
        if extra_terms:
            for t, boost_w in extra_terms.items():
                if t in q_counts:
                    continue
                post = self.postings.get(t)
                if post is None:
                    continue
                doc_ids, norm_tfs = post
                info_w = self.base_weight[t] * self.learned_gain.get(t, 1.0) * boost_w
                scores[doc_ids] += np.float32(info_w) * norm_tfs
        return scores

    def score_candidates(self, q_text: str, candidates: np.ndarray,
                         term_weights: Optional[Dict[str, float]] = None,
                         extra_terms: Optional[Dict[str, float]] = None) -> np.ndarray:
        return self.score_all(q_text, term_weights=term_weights, extra_terms=extra_terms)[candidates]


# ---------------------------------------------------------------------------
# 3. semantic_model.py equivalent: Hashed Character 3-Gram TF-IDF Manifold (MT)
# ---------------------------------------------------------------------------

def char_grams(v: str, dim: int = 4096, fold_digits: bool = False, include_words: bool = True) -> List[int]:
    """Hashed character 3-grams (from semantic_model.py:18) + optional word/bi-word hashes."""
    low = v.lower()
    if fold_digits:
        low = re.sub(r"[0-9]", "0", low)
    s = "\x02" + re.sub(r"\s+", " ", low) + "\x03"
    b = s.encode("utf-8", errors="ignore")
    n = len(b)
    if n < 3:
        return [0]
    arr = np.frombuffer(b, dtype=np.uint8).astype(np.uint32)
    h3 = ((arr[:-2] * 73856093) ^ (arr[1:-1] * 19349663) ^ (arr[2:] * 83492791)) % dim
    out = h3.tolist()
    if include_words:
        toks = TOKEN_RE.findall(low)
        for tok in toks:
            if len(tok) >= 3:
                out.append((zlib.crc32(tok.encode()) & 0xFFFFFFFF) % dim)
        # Also hash adjacent content-word bigrams for phrase sensitivity
        content_toks = [t for t in toks if len(t) >= 4]
        for i in range(len(content_toks) - 1):
            bg = f"{content_toks[i]}_{content_toks[i + 1]}"
            out.append((zlib.crc32(bg.encode()) & 0xFFFFFFFF) % dim)
    return out or [0]


class SemanticSpace:
    """Port of HeaderSpace from semantic_model.py to document retrieval, with learnable gram gains."""

    def __init__(self, docs: List[str], dim: int = 4096, fold_digits: bool = False, include_words: bool = True):
        self.N = len(docs)
        self.dim = dim
        self.fold_digits = fold_digits
        self.include_words = include_words

        df = np.zeros(dim, dtype=np.float32)
        rows_idx: List[np.ndarray] = []
        rows_cnt: List[np.ndarray] = []

        for d in docs:
            g = char_grams(d, dim=dim, fold_digits=fold_digits, include_words=include_words)
            idx, cnt = np.unique(np.asarray(g, dtype=np.int32), return_counts=True)
            df[idx] += 1.0
            rows_idx.append(idx)
            rows_cnt.append(1.0 + np.log(cnt.astype(np.float32)))

        # Same IDF formula as semantic_model.py:32
        self.base_idf = (np.log((self.N + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
        self.gram_gain = np.ones(dim, dtype=np.float32)
        self.idf = self.base_idf.copy()

        M = np.zeros((self.N, dim), dtype=np.float32)
        for i, (idx, tf) in enumerate(zip(rows_idx, rows_cnt)):
            w = tf * self.idf[idx]
            norm = float(np.linalg.norm(w)) or 1.0
            M[i, idx] = w / norm

        self.MT = np.ascontiguousarray(M.T)

    def apply_gram_gains(self) -> None:
        self.idf = (self.base_idf * self.gram_gain).astype(np.float32)

    def weights(self, text: str, gram_mods: Optional[Dict[int, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        g = char_grams(text, dim=self.dim, fold_digits=self.fold_digits, include_words=self.include_words)
        idx, cnt = np.unique(np.asarray(g, dtype=np.int32), return_counts=True)
        tf = 1.0 + np.log(cnt.astype(np.float32))
        w = tf * self.idf[idx]
        if gram_mods:
            for pos, bucket in enumerate(idx):
                if int(bucket) in gram_mods:
                    w[pos] *= gram_mods[int(bucket)]
        norm = float(np.linalg.norm(w)) or 1.0
        return idx, (w / norm).astype(np.float32)

    def score_all(self, q_text: str, gram_mods: Optional[Dict[int, float]] = None) -> np.ndarray:
        idx, w = self.weights(q_text, gram_mods=gram_mods)
        return w @ self.MT[idx]

    def score_candidates(self, q_text: str, candidates: np.ndarray,
                         gram_mods: Optional[Dict[int, float]] = None) -> np.ndarray:
        idx, w = self.weights(q_text, gram_mods=gram_mods)
        return (w @ self.MT[idx])[candidates]

    def nbytes(self) -> int:
        return int(self.MT.nbytes + self.idf.nbytes)


# ---------------------------------------------------------------------------
# 4. calibration.py equivalent: Blame by Diff + Per-Route Threshold Tightening
# ---------------------------------------------------------------------------

def query_route(q_tokens: List[str], gate: EntropyGate, num_routes: int = 64) -> int:
    if not q_tokens:
        return 0
    anchor = max(q_tokens, key=lambda t: gate.surprisal(t))
    return (zlib.crc32(anchor.encode()) & 0xFFFFFFFF) % num_routes


def blame_false_lead(query_tokens: List[str], part_a_false: str, part_b_false: str,
                     true_winner_full: str) -> List[str]:
    """Port of blame(flagged, clean, headers, qpairs) from calibration.py:20."""
    a_toks = set(tokenize(part_a_false))
    b_toks = set(tokenize(part_b_false))
    w_toks = set(tokenize(true_winner_full))
    blamed = []
    for t in set(query_tokens):
        if len(t) >= 4 and (t in a_toks) and (t not in b_toks) and (t not in w_toks):
            blamed.append(t)
    return blamed


class RetrievalCalibrator:
    """Port of Calibrator from calibration.py:31."""

    def __init__(self, halflife: float = 75.0, margin: float = 0.18):
        self.hl = halflife
        self.margin = margin
        self.o: Dict[Tuple[int, str, str], Tuple[float, float]] = {}
        self.flips = 0
        self.blames = 0
        self.tightened = 0

    def tighten(self, rt: int, layer: str, name: str, target_val: float, base: float, now: float) -> bool:
        new_val = max(0.40, target_val - self.margin)
        cur = self.thr(rt, layer, name, base, now)
        if new_val < cur:
            self.o[(rt, layer, name)] = (new_val, now)
            self.tightened += 1
            return True
        return False

    def boost(self, rt: int, layer: str, name: str, target_val: float, base: float, now: float) -> bool:
        new_val = min(1.60, target_val + self.margin)
        cur = self.thr(rt, layer, name, base, now)
        if new_val > cur:
            self.o[(rt, layer, name)] = (new_val, now)
            self.tightened += 1
            return True
        return False

    def thr(self, rt: int, layer: str, name: str, base: float, now: float) -> float:
        t = self.o.get((rt, layer, name))
        if t is None:
            return base
        return base - (base - t[0]) * (0.5 ** ((now - t[1]) / self.hl))

    def get_term_weights(self, rt: int, q_tokens: List[str], now: float) -> Dict[str, float]:
        return {t: self.thr(rt, "gate", t, 1.0, now) for t in set(q_tokens)}


# ---------------------------------------------------------------------------
# 5. Complete Standalone SSE Retrieval Engine + Offline Fine-Tuning
# ---------------------------------------------------------------------------

class SSERetriever:
    """Standalone SSE Retrieval Engine combining all 4 layers plus offline Fine-Tuning:
      - EntropyGate (information-theoretic surprisal + Shannon entropy + learnable term gains)
      - SemanticSpace (hashed 3-gram + word/bigram TF-IDF via contiguous MT + learnable gram gains)
      - Split-Document Indexing (Part A critical/head vs Part B bulk + split-bridge vocabulary)
      - RetrievalCalibrator & Fine-Tuner (adversarial hard-negative blame + margin learning)
    """

    def __init__(self, corpus: List[Dict[str, str]], sem_dim: int = 16384, split_strategy: str = "regions",
                 split_f: float = 0.28):
        self.N = len(corpus)
        self.sem_dim = sem_dim
        self.split_strategy = split_strategy
        self.split_f = split_f
        self.doc_ids = [str(row.get("_id", i)) for i, row in enumerate(corpus)]
        self.doc_id_to_idx = {did: i for i, did in enumerate(self.doc_ids)}

        self.parts_a: List[str] = []
        self.parts_b: List[str] = []
        self.full_docs: List[str] = []
        self.a_token_sets: List[Set[str]] = []
        self.b_token_sets: List[Set[str]] = []
        self.full_token_sets: List[Set[str]] = []

        for row in corpus:
            title = row.get("title", "")
            text = row.get("text", "")
            a, b = split_doc(title, text, strategy=split_strategy, f=split_f)
            self.parts_a.append(a)
            self.parts_b.append(b)
            # Repeat title twice in full representation to preserve title salience
            full = f"{title}. {title}. {text}".strip()
            self.full_docs.append(full)
            a_set = set(tokenize(a))
            b_set = set(tokenize(b))
            self.a_token_sets.append(a_set)
            self.b_token_sets.append(b_set)
            self.full_token_sets.append(a_set | b_set)

        # Layer 1: Entropy / Surprisal Gate on Part A (head), Part B (bulk), and Full (A+B)
        self.gate_a = EntropyGate(self.parts_a)
        self.gate_b = EntropyGate(self.parts_b)
        self.gate_full = EntropyGate(self.full_docs)

        # Layer 2: Semantic 3-gram TF-IDF Space on Part A and Full (A+B)
        self.sem_a = SemanticSpace(self.parts_a, dim=sem_dim, fold_digits=False, include_words=True)
        self.sem_full = SemanticSpace(self.full_docs, dim=sem_dim, fold_digits=False, include_words=True)

        # Compact 4096-dim exact port of semantic_model.py (char 3-grams only, digits folded)
        self.sem_raw4096 = SemanticSpace(self.full_docs, dim=4096, fold_digits=True, include_words=False)

        # Layer 4: Online Calibrator + Offline Fine-Tuned Parameters
        self.cal = RetrievalCalibrator(halflife=75.0, margin=0.18)
        self.step = 0.0

        # Learnable mixing parameters (tuned during fine-tuning)
        self.mix_stage_a = (0.50, 0.25, 0.25)        # (ent_full, ent_a, sem_a)
        self.mix_verify = (0.54, 0.30, 0.08, 0.08)   # (stage_a, sem_full, ent_b, quorum)
        self.prf_weight = 0.14                       # Split-corroborated bridge expansion weight
        # Learned query-term -> document-bridge-term associations from fine-tuning
        self.learned_bridges: Dict[str, Counter] = defaultdict(Counter)

    # -----------------------------------------------------------------------
    # Offline Fine-Tuning: Self-Supervised (Split A<->B) & Supervised (Train Qrels)
    # -----------------------------------------------------------------------

    def fine_tune_self_supervised(self, epochs: int = 2, max_docs: int = 2500) -> Dict[str, float]:
        """Self-Supervised Fine-Tuning on the Corpus (ZERO labeled queries needed!):

        Uses SSE's core principle: for every document `d`, a claim extracted from Part B
        (bulk body) is used to query `gate_a` (Part A = title + critical head ONLY).
        Because `pseudo_q` comes from Part B, any term in `pseudo_q` that hits a decoy's Part A
        without appearing in `d`'s Part A is a genuine cross-half decoy term! Conversely, terms
        that appear in BOTH `d`'s Part B and `d`'s Part A are true document spine terms.
        """
        t0 = time.perf_counter()
        rng = np.random.default_rng(20261021)
        indices = rng.choice(self.N, size=min(max_docs, self.N), replace=False)

        total_flips = 0
        total_blames = 0
        total_spine_boosts = 0
        for ep in range(epochs):
            for doc_idx in indices:
                d_id = int(doc_idx)
                b_text = self.parts_b[d_id]
                if len(b_text) < 60:
                    continue
                sents = [s.strip() for s in re.split(r"[.!?]+", b_text) if len(s.strip()) >= 40]
                if not sents:
                    continue
                # Use the last (conclusion) sentence of Part B as the pseudo-query against Part A!
                pseudo_q = sents[-1][:200]
                q_toks = [t for t in tokenize(pseudo_q) if len(t) >= 4]
                if len(q_toks) < 3:
                    continue

                # Score against Part A ONLY (so Part B terms aren't trivially guaranteed in d_id)
                s_ent_a = self.gate_a.score_all(pseudo_q)
                top5 = np.argpartition(-s_ent_a, 5)[:5]
                top5 = top5[np.argsort(-s_ent_a[top5])]
                winner = int(top5[0])

                if winner != d_id:
                    total_flips += 1
                    decoy_a_toks = self.a_token_sets[winner]
                    true_a_toks = self.a_token_sets[d_id]
                    for t in set(q_toks):
                        if t in decoy_a_toks and t not in true_a_toks:
                            # Blame: term from B matched decoy's head A, not true doc's head A
                            for g in (self.gate_full, self.gate_a, self.gate_b):
                                g.learned_gain[t] = max(0.55, g.learned_gain.get(t, 1.0) * 0.96)
                            total_blames += 1
                        elif t in true_a_toks and t not in decoy_a_toks:
                            # Spine term: bridges true doc's conclusion (B) and head (A)
                            for g in (self.gate_full, self.gate_a, self.gate_b):
                                g.learned_gain[t] = min(1.65, g.learned_gain.get(t, 1.0) * 1.05)
                            total_spine_boosts += 1

        return {
            "mode": "self_supervised_split_ab",
            "epochs": epochs,
            "docs_sampled": int(len(indices)),
            "decoy_flips": total_flips,
            "term_blames": total_blames,
            "spine_boosts": total_spine_boosts,
            "seconds": round(time.perf_counter() - t0, 2),
        }

    def fine_tune_supervised(self, train_queries: Dict[str, str], train_qrels: Dict[str, Set[str]],
                             epochs: int = 4, lr: float = 0.12) -> Dict[str, float]:
        """Supervised Task Fine-Tuning on Held-Out Training Queries (analogous to `/__train` + Adversary Calibrator).

        For each training query `(q, pos_docs)`:
          1. Runs current SSE scoring to find the top hard-negative decoy `neg_doc` not in `pos_docs`.
          2. If `score(neg_doc) > score(pos_doc) - margin`, triggers SSE Blame-by-Diff:
             - Terms in `q & neg_doc` but NOT in `pos_doc` are blamed (gain tightened).
             - Terms in `q & pos_doc` but NOT in `neg_doc` are boosted (gain increased).
             - Hashed 3-gram buckets in `q & pos_doc` vs `q & neg_doc` have their `gram_gain` updated.
             - High-surprisal bridge terms in `pos_doc` (corroborated in both Part A and Part B) are
               linked to the anchor query terms in `self.learned_bridges`.
          3. Tunes the stage mixing weights `(mix_stage_a, mix_verify, prf_weight)` on the training split.
        """
        t0 = time.perf_counter()
        valid_qids = [qid for qid in train_qrels if qid in train_queries]
        total_violation_steps = 0
        total_term_updates = 0
        total_bridge_links = 0

        for ep in range(epochs):
            step_lr = lr * (0.8 ** ep)
            for qid in valid_qids:
                qtext = train_queries[qid]
                q_toks = [t for t in tokenize(qtext) if len(t) >= 3]
                if not q_toks:
                    continue
                pos_indices = [self.doc_id_to_idx[did] for did in train_qrels[qid] if did in self.doc_id_to_idx]
                if not pos_indices:
                    continue
                pos_set = set(pos_indices)

                # Score with current EntropyGate + SemanticSpace
                s_ent = self.gate_full.score_all(qtext)
                s_sem = self.sem_full.score_all(qtext)
                s_ent_n = s_ent / (float(s_ent.max()) or 1.0)
                s_sem_n = s_sem / (float(s_sem.max()) or 1.0)
                comb = 0.65 * s_ent_n + 0.35 * s_sem_n

                # Find best positive and hardest negative decoy
                best_pos = max(pos_indices, key=lambda idx: float(comb[idx]))
                top_cands = np.argpartition(-comb, 15)[:15]
                top_cands = top_cands[np.argsort(-comb[top_cands])]
                neg_cands = [int(c) for c in top_cands if int(c) not in pos_set]
                if not neg_cands:
                    continue
                hard_neg = neg_cands[0]

                # Margin violation: hard negative beats or nearly ties positive
                if float(comb[hard_neg]) + 0.08 >= float(comb[best_pos]):
                    total_violation_steps += 1
                    pos_toks = self.full_token_sets[best_pos]
                    neg_toks = self.full_token_sets[hard_neg]

                    for t in set(q_toks):
                        in_pos = t in pos_toks
                        in_neg = t in neg_toks
                        if in_neg and not in_pos:
                            # Blame: query token matched the hard-negative decoy, not the true doc!
                            for g in (self.gate_full, self.gate_a, self.gate_b):
                                g.learned_gain[t] = max(0.30, g.learned_gain.get(t, 1.0) * (1.0 - step_lr))
                            total_term_updates += 1
                        elif in_pos and not in_neg:
                            # Reward: query token uniquely identifies the positive doc!
                            for g in (self.gate_full, self.gate_a, self.gate_b):
                                g.learned_gain[t] = min(2.40, g.learned_gain.get(t, 1.0) * (1.0 + step_lr * 1.25))
                            total_term_updates += 1

                    # Update 3-gram IDF gains on the active query buckets
                    q_grams = set(char_grams(qtext, dim=self.sem_dim, fold_digits=False, include_words=True))
                    pos_grams = set(char_grams(self.full_docs[best_pos], dim=self.sem_dim, fold_digits=False, include_words=True))
                    neg_grams = set(char_grams(self.full_docs[hard_neg], dim=self.sem_dim, fold_digits=False, include_words=True))
                    for b_idx in q_grams:
                        if b_idx in pos_grams and b_idx not in neg_grams:
                            self.sem_full.gram_gain[b_idx] = min(2.0, self.sem_full.gram_gain[b_idx] * (1.0 + 0.5 * step_lr))
                            self.sem_a.gram_gain[b_idx] = min(2.0, self.sem_a.gram_gain[b_idx] * (1.0 + 0.5 * step_lr))
                        elif b_idx in neg_grams and b_idx not in pos_grams:
                            self.sem_full.gram_gain[b_idx] = max(0.45, self.sem_full.gram_gain[b_idx] * (1.0 - 0.5 * step_lr))
                            self.sem_a.gram_gain[b_idx] = max(0.45, self.sem_a.gram_gain[b_idx] * (1.0 - 0.5 * step_lr))

                # Learn Split-Corroborated Bridge Associations on the first epoch:
                # Terms that appear in BOTH Part A and Part B of the ground-truth document
                if ep == 0:
                    pos_bridge = (self.a_token_sets[best_pos] & self.b_token_sets[best_pos]) - set(q_toks)
                    high_bridge = [
                        bt for bt in pos_bridge
                        if len(bt) >= 4 and self.gate_full.surprisal(bt) > self.gate_full.mu_s
                    ]
                    # Associate the query's top-2 highest-surprisal anchor terms with the doc's bridge terms
                    q_anchors = sorted(set(q_toks), key=lambda t: -self.gate_full.surprisal(t))[:2]
                    for anc in q_anchors:
                        for bt in high_bridge[:6]:
                            self.learned_bridges[anc][bt] += 1
                            total_bridge_links += 1

            self.sem_full.apply_gram_gains()
            self.sem_a.apply_gram_gains()

        # Coordinate grid refinement of stage mixing weights on a 200-query training subset
        sample_qids = valid_qids[:200]
        best_ndcg = -1.0
        best_cfg = (self.mix_stage_a, self.mix_verify, self.prf_weight)
        candidates_cfg = [
            ((0.50, 0.25, 0.25), (0.54, 0.30, 0.08, 0.08), 0.14),
            ((0.55, 0.25, 0.20), (0.58, 0.28, 0.07, 0.07), 0.12),
            ((0.48, 0.22, 0.30), (0.50, 0.36, 0.07, 0.07), 0.15),
            ((0.52, 0.28, 0.20), (0.56, 0.30, 0.06, 0.08), 0.18),
            ((0.60, 0.20, 0.20), (0.62, 0.26, 0.06, 0.06), 0.10),
        ]
        for ma, mv, pw in candidates_cfg:
            self.mix_stage_a, self.mix_verify, self.prf_weight = ma, mv, pw
            ndcgs = []
            for qid in sample_qids:
                ranked, _ = self._run_calibrated(train_queries[qid], k=10, depth=150, update_cal=False, use_prf=True)
                r_ids = [self.doc_ids[int(i)] for i in ranked]
                rel = train_qrels[qid]
                hits = [i + 1 for i, d in enumerate(r_ids[:10]) if d in rel]
                dcg = sum(1.0 / math.log2(r + 1) for r in hits)
                idcg = sum(1.0 / math.log2(r + 2) for r in range(min(10, len(rel))))
                ndcgs.append(dcg / idcg if idcg else 0.0)
            mean_n = float(np.mean(ndcgs))
            if mean_n > best_ndcg:
                best_ndcg = mean_n
                best_cfg = (ma, mv, pw)

        self.mix_stage_a, self.mix_verify, self.prf_weight = best_cfg
        return {
            "mode": "supervised_train_qrels",
            "train_queries": len(valid_qids),
            "epochs": epochs,
            "margin_violations_fixed": total_violation_steps,
            "term_gain_updates": total_term_updates,
            "learned_bridge_links": total_bridge_links,
            "train_subset_ndcg10": round(best_ndcg, 4),
            "best_mix_stage_a": self.mix_stage_a,
            "best_mix_verify": self.mix_verify,
            "best_prf_weight": self.prf_weight,
            "seconds": round(time.perf_counter() - t0, 2),
        }

    def warmup_calibrate(self, sample_queries: List[str]) -> Dict[str, int]:
        """Run self-supervised split-calibration warmup over sample queries (no labels needed)."""
        for q in sample_queries:
            self._run_calibrated(q, k=10, update_cal=True, use_prf=False)
        return {"flips": self.cal.flips, "blames": self.cal.blames, "tightened": self.cal.tightened}

    def _run_calibrated(self, query: str, k: int = 10, depth: int = 150,
                        update_cal: bool = True, use_prf: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        self.step += 1.0
        q_toks = tokenize(query)
        rt = query_route(q_toks, self.gate_full)
        term_w = self.cal.get_term_weights(rt, q_toks, self.step)

        # Look up any fine-tuned bridge terms associated with high-surprisal query anchors
        extra_terms: Dict[str, float] = {}
        if self.learned_bridges:
            q_anchors = sorted(set(q_toks), key=lambda t: -self.gate_full.surprisal(t))[:2]
            for anc in q_anchors:
                if anc in self.learned_bridges:
                    for bt, cnt in self.learned_bridges[anc].most_common(3):
                        if cnt >= 1:
                            extra_terms[bt] = 0.18

        # Stage 1 (Front Gates + Part A): fast scan using EntropyGate + Part-A SemanticSpace
        s_ent_a = self.gate_a.score_all(query, term_weights=term_w, extra_terms=extra_terms)
        s_ent_f = self.gate_full.score_all(query, term_weights=term_w, extra_terms=extra_terms)
        s_sem_a = self.sem_a.score_all(query)

        def norm(v: np.ndarray) -> np.ndarray:
            mx = float(v.max())
            return v / mx if mx > 1e-8 else v

        w_ef, w_ea, w_sa = self.mix_stage_a
        stage_a_score = w_ef * norm(s_ent_f) + w_ea * norm(s_ent_a) + w_sa * norm(s_sem_a)

        # Select top `depth` candidates from Stage A
        cands = np.argpartition(-stage_a_score, min(depth, self.N - 1))[:depth]
        cands = cands[np.argsort(-stage_a_score[cands], kind="stable")]

        # Stage 2 (Split B / Full Semantic + Cross-Half Verification on survivors)
        sem_f_cands = self.sem_full.score_candidates(query, cands)
        sem_f_norm = sem_f_cands / (float(sem_f_cands.max()) or 1.0)
        ent_b_cands = self.gate_b.score_all(query, term_weights=term_w)[cands]
        ent_b_norm = ent_b_cands / (float(ent_b_cands.max()) or 1.0)

        # Split-Quorum / Cross-Half Agreement Bonus
        q_high = {t for t in q_toks if len(t) >= 4 and self.gate_full.surprisal(t) >= self.gate_full.mu_s * 0.85}
        quorum_bonus = np.zeros(len(cands), dtype=np.float32)
        if q_high:
            denom = float(len(q_high))
            for idx_pos, doc_id in enumerate(cands):
                did = int(doc_id)
                a_hits = q_high & self.a_token_sets[did]
                b_hits = q_high & self.b_token_sets[did]
                both = len(a_hits & b_hits) / denom
                union_ab = len(a_hits | b_hits) / denom
                quorum_bonus[idx_pos] = 0.55 * union_ab + 0.45 * both

        v_sa, v_sf, v_eb, v_qb = self.mix_verify
        verified_score = (
            v_sa * stage_a_score[cands]
            + v_sf * sem_f_norm
            + v_eb * ent_b_norm
            + v_qb * quorum_bonus
        )
        order = np.argsort(-verified_score, kind="stable")
        ranked_cands = cands[order]
        ranked_scores = verified_score[order]

        # Split-Verification Feedback & Calibration (from calibration.py):
        top_a_doc = int(cands[0])
        top_v_doc = int(ranked_cands[0])
        if update_cal and top_a_doc != top_v_doc:
            self.cal.flips += 1
            blamed = blame_false_lead(
                q_toks,
                self.parts_a[top_a_doc],
                self.parts_b[top_a_doc],
                self.full_docs[top_v_doc],
            )
            for t in blamed:
                self.cal.blames += 1
                self.cal.tighten(rt, "gate", t, term_w.get(t, 1.0), 1.0, self.step)
            win_both = self.a_token_sets[top_v_doc] & self.b_token_sets[top_v_doc]
            for t in set(q_toks) & win_both:
                if len(t) >= 4:
                    self.cal.boost(rt, "gate", t, term_w.get(t, 1.0), 1.0, self.step)

        # Optional Split-Corroborated Pseudo-Relevance Refinement (only if top-1 and top-2 agree on bridge terms)
        if use_prf and self.prf_weight > 0.0 and len(ranked_cands) >= 2:
            d0, d1 = int(ranked_cands[0]), int(ranked_cands[1])
            shared_ab = (
                (self.a_token_sets[d0] & self.b_token_sets[d0])
                & (self.a_token_sets[d1] & self.b_token_sets[d1])
            ) - set(q_toks)
            bridge_map = {
                t: 0.25 for t in shared_ab
                if len(t) >= 4 and self.gate_full.surprisal(t) > self.gate_full.mu_s
            }
            if bridge_map:
                prf_s = self.gate_full.score_all(query, term_weights=term_w, extra_terms=bridge_map)[cands]
                prf_n = prf_s / (float(prf_s.max()) or 1.0)
                comb2 = (1.0 - self.prf_weight) * verified_score + self.prf_weight * prf_n
                order2 = np.argsort(-comb2, kind="stable")
                ranked_cands = cands[order2]
                ranked_scores = comb2[order2]

        return ranked_cands[:k], ranked_scores[:k]

    def search(self, query: str, method: str = "sse_calibrated", k: int = 10) -> np.ndarray:
        """Run any of the SSE ablation or fine-tuned designs."""
        if method == "sse_semantic_raw4096":
            scores = self.sem_raw4096.score_all(query)
            return np.argsort(-scores, kind="stable")[:k]

        if method == "sse_semantic_16k":
            scores = self.sem_full.score_all(query)
            return np.argsort(-scores, kind="stable")[:k]

        if method == "sse_entropy_gate":
            scores = self.gate_full.score_all(query)
            return np.argsort(-scores, kind="stable")[:k]

        if method == "sse_gate_plus_semantic":
            s_ent = self.gate_full.score_all(query)
            s_sem = self.sem_full.score_all(query)
            s_ent = s_ent / (float(s_ent.max()) or 1.0)
            s_sem = s_sem / (float(s_sem.max()) or 1.0)
            scores = 0.65 * s_ent + 0.35 * s_sem
            return np.argsort(-scores, kind="stable")[:k]

        if method == "sse_split":
            s_ent_a = self.gate_a.score_all(query)
            s_ent_f = self.gate_full.score_all(query)
            s_sem_a = self.sem_a.score_all(query)
            stage_a = (
                0.50 * (s_ent_f / (float(s_ent_f.max()) or 1.0))
                + 0.25 * (s_ent_a / (float(s_ent_a.max()) or 1.0))
                + 0.25 * (s_sem_a / (float(s_sem_a.max()) or 1.0))
            )
            depth = min(150, self.N)
            cands = np.argpartition(-stage_a, depth - 1)[:depth]
            cands = cands[np.argsort(-stage_a[cands], kind="stable")]
            sem_f = self.sem_full.score_candidates(query, cands)
            sem_f = sem_f / (float(sem_f.max()) or 1.0)
            comb = 0.62 * stage_a[cands] + 0.38 * sem_f
            return cands[np.argsort(-comb, kind="stable")][:k]

        if method == "sse_calibrated":
            ranked, _ = self._run_calibrated(query, k=k, depth=150, update_cal=True, use_prf=False)
            return ranked

        if method == "sse_finetuned":
            ranked, _ = self._run_calibrated(query, k=k, depth=150, update_cal=True, use_prf=True)
            return ranked

        raise ValueError(f"Unknown SSE method: {method}")
