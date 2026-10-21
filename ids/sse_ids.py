"""sse_ids.py: Semantic Split Entropy (SSE) Gating adapted and fixed for Network IDS (CIC-IDS2017).

Compares:
  1. UnfixedSSEIDS: faithful port of sse/ (anomaly_model.py + semantic_model.py + calibration.py)
     showing why linear mean/std, digit-folding ('[0-9]' -> '0'), global pooling across ports, and scalar
     threshold lowering (thr = z - 0.25) fail on network flow IDS benchmarks.
  2. FixedSSEIDS: preserves the exact 4-part SSE architecture (EntropyGate -> SemanticSpace -> SplitVerifier ->
     BlameCalibrator) while fixing the three root causes:
       - Fix 1 (Per-Route Bit-Scale & Stream Surprisal in EntropyGate):
         Uses sign(x)*log2(1+|x|), exact TCP/initiation tuple surprisal, and sliding-window stream entropy
         calibrated per route (Destination Port) rather than pooling Port 80 with random ephemeral ports.
       - Fix 2 (Per-Route Template Manifold in SemanticSpace & SplitVerifier):
         Matches Part-A (forward initiation) and Part-B (backward server response + timing + A->B coupling)
         against deduplicated normal templates of the SAME route (1 - max_cos / min L2 to normal manifold),
         preserving numeric values instead of folding digits to '0'.
       - Fix 3 (Closed-Loop Split -> Front Gate Blame Manifold Calibration):
         When a quorum of flows slips past the Part-A Front Gates (a Miss) and is caught by SplitVerifier
         on Part B (using ZERO ground-truth labels), BlameCalibrator extracts their Part-A profile and adds it
         to the route's blamed Part-A manifold Bad_A[route]. Subsequent flows matching Bad_A[route] closer than
         Norm_A[route] are caught immediately at the Front Gate without raising benign FPR on that port.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

ROUTE_COL = 0  # Destination Port

# Part A: Client / Forward initiation features observable at request/flow start
PART_A_COLS = [
    2,   # Total Fwd Packets
    4,   # Total Length of Fwd Packets
    6,   # Fwd Packet Length Max
    7,   # Fwd Packet Length Min
    8,   # Fwd Packet Length Mean
    9,   # Fwd Packet Length Std
    30,  # Fwd PSH Flags
    32,  # Fwd URG Flags
    34,  # Fwd Header Length
    43,  # FIN Flag Count
    44,  # SYN Flag Count
    45,  # RST Flag Count
    46,  # PSH Flag Count
    47,  # ACK Flag Count
    48,  # URG Flag Count
    53,  # Avg Fwd Segment Size
    62,  # Subflow Fwd Packets
    63,  # Subflow Fwd Bytes
    66,  # Init_Win_bytes_forward
    68,  # act_data_pkt_fwd
    69,  # min_seg_size_forward
]

# Part B: Server / Backward response & full-flow temporal outcome (used by SplitVerifier)
PART_B_COLS = [
    1,   # Flow Duration
    3,   # Total Backward Packets
    5,   # Total Length of Bwd Packets
    10,  # Bwd Packet Length Max
    11,  # Bwd Packet Length Min
    12,  # Bwd Packet Length Mean
    13,  # Bwd Packet Length Std
    14,  # Flow Bytes/s
    15,  # Flow Packets/s
    16,  # Flow IAT Mean
    17,  # Flow IAT Std
    18,  # Flow IAT Max
    19,  # Flow IAT Min
    21,  # Fwd IAT Mean
    22,  # Fwd IAT Std
    23,  # Fwd IAT Max
    26,  # Bwd IAT Mean
    27,  # Bwd IAT Std
    28,  # Bwd IAT Max
    35,  # Bwd Header Length
    36,  # Fwd Packets/s
    37,  # Bwd Packets/s
    39,  # Max Packet Length
    40,  # Packet Length Mean
    41,  # Packet Length Std
    42,  # Packet Length Variance
    51,  # Down/Up Ratio
    52,  # Average Packet Size
    54,  # Avg Bwd Segment Size
    64,  # Subflow Bwd Packets
    65,  # Subflow Bwd Bytes
    67,  # Init_Win_bytes_backward
    70,  # Active Mean
    74,  # Idle Mean
]


def signed_log2(x: np.ndarray) -> np.ndarray:
    """Information-scale bit transform: sign(x) * log2(1 + |x|)."""
    return np.sign(x) * np.log2(1.0 + np.abs(x))


def min_l2_dist(X: np.ndarray, T: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Fast chunked Euclidean distance from each row of X (N, D) to nearest template in T (M, D)."""
    x2 = np.sum(X * X, axis=1, keepdims=True)
    t2 = np.sum(T * T, axis=1)
    out = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), chunk):
        sl = X[i : i + chunk]
        d2 = np.maximum(0.0, x2[i : i + chunk] + t2 - 2.0 * (sl @ T.T))
        out[i : i + chunk] = np.sqrt(d2.min(axis=1))
    return out


# ============================================================================
# 1. UNFIXED BASELINE (Direct port of sse/ behavior)
# ============================================================================

class UnfixedSSEIDS:
    """Faithful port of sse/{anomaly_model,semantic_model,calibration}.py."""

    def __init__(self, fpr: float = 0.02):
        self.fpr = fpr
        self.mu_a: Optional[np.ndarray] = None
        self.sd_a: Optional[np.ndarray] = None
        self.mu_raw: Optional[np.ndarray] = None
        self.sd_raw: Optional[np.ndarray] = None
        self.thr_gate: float = 0.0
        self.mu_b: Optional[np.ndarray] = None
        self.sd_b: Optional[np.ndarray] = None
        self.thr_split: float = 0.0

    def _raw_scores(self, xa: np.ndarray, ports: np.ndarray) -> np.ndarray:
        z_val = np.abs(xa - self.mu_a) / self.sd_a
        max_z = z_val.max(axis=1)
        port_counts = np.array([self.port_counts.get(int(p), 0) for p in ports], dtype=np.float32)
        s_port = -np.log2((port_counts + 0.1) / (self.n_base + 0.2))
        str_len = np.floor(np.log10(np.maximum(1.0, np.abs(xa)))).sum(axis=1)
        z_len = np.abs(str_len - self.len_mu) / max(self.len_sd, 2.0)
        return np.stack([s_port, max_z, z_len], axis=1)

    def fit(self, X_benign: np.ndarray) -> None:
        half = len(X_benign) // 2
        b1, b2 = X_benign[:half], X_benign[half:]
        ports1 = b1[:, ROUTE_COL].astype(np.int64)
        xa1 = b1[:, PART_A_COLS]
        self.n_base = len(b1)
        self.port_counts = Counter(int(p) for p in ports1)
        self.mu_a = xa1.mean(axis=0)
        self.sd_a = np.maximum(xa1.std(axis=0), 2.0)
        str_len1 = np.floor(np.log10(np.maximum(1.0, np.abs(xa1)))).sum(axis=1)
        self.len_mu = float(str_len1.mean())
        self.len_sd = float(max(str_len1.std(), 2.0))

        raw1 = self._raw_scores(xa1, ports1)
        self.mu_raw = raw1.mean(axis=0)
        self.sd_raw = np.maximum(raw1.std(axis=0), 1e-3)

        ports2 = b2[:, ROUTE_COL].astype(np.int64)
        xa2 = b2[:, PART_A_COLS]
        raw2 = self._raw_scores(xa2, ports2)
        scores2 = ((raw2 - self.mu_raw) / self.sd_raw).max(axis=1)
        self.thr_gate = float(np.quantile(scores2, 1.0 - self.fpr))

        xb1 = b1[:, PART_B_COLS]
        self.mu_b = xb1.mean(axis=0)
        self.sd_b = np.maximum(xb1.std(axis=0), 2.0)
        xb2 = b2[:, PART_B_COLS]
        sb2 = (np.abs(xb2 - self.mu_b) / self.sd_b).max(axis=1)
        self.thr_split = float(np.quantile(sb2, 1.0 - self.fpr))

    def evaluate_stream(self, X: np.ndarray, calibrate: bool = False) -> np.ndarray:
        ports = X[:, ROUTE_COL].astype(np.int64)
        xa = X[:, PART_A_COLS]
        xb = X[:, PART_B_COLS]
        raw = self._raw_scores(xa, ports)
        gate_scores = ((raw - self.mu_raw) / self.sd_raw).max(axis=1)
        if not calibrate:
            return gate_scores > self.thr_gate

        per_feat_z = (np.abs(xa - self.mu_a) / self.sd_a - self.mu_raw[1]) / self.sd_raw[1]
        split_scores = (np.abs(xb - self.mu_b) / self.sd_b).max(axis=1)
        cal_thr: Dict[Tuple[int, int], float] = {}
        preds = np.zeros(len(X), dtype=bool)
        for i in range(len(X)):
            rt = int(ports[i])
            if gate_scores[i] > self.thr_gate:
                preds[i] = True
                continue
            odd_cal = False
            for j in range(xa.shape[1]):
                t = cal_thr.get((rt, j))
                if t is not None and per_feat_z[i, j] > t:
                    odd_cal = True
                    break
            if odd_cal:
                preds[i] = True
                continue
            if split_scores[i] > self.thr_split:
                preds[i] = True
                top_j = int(np.argmax(per_feat_z[i]))
                new_t = float(per_feat_z[i, top_j]) - 0.25
                old_t = cal_thr.get((rt, top_j), self.thr_gate)
                if new_t < old_t:
                    cal_thr[(rt, top_j)] = new_t
        return preds


# ============================================================================
# 2. FIXED SSE IDS (EntropyGate + SemanticSpace + SplitVerifier + BlameCalibrator)
# ============================================================================

class FixedSSEIDS:
    """Fixed Semantic Split Entropy (SSE) IDS with per-route template manifolds & closed-loop blame."""

    def __init__(self, fpr: float = 0.015, quorum: int = 2):
        self.fpr = fpr
        self.quorum = quorum

        self.n_base: int = 0
        self.major_ports: set = set()
        self.mu_a: Optional[np.ndarray] = None
        self.sd_a: Optional[np.ndarray] = None
        self.mu_b: Optional[np.ndarray] = None
        self.sd_b: Optional[np.ndarray] = None

        # EntropyGate (Part A): discrete header-tuple surprisal + sliding-window burst surprisal
        self.tuple_counts_a: Counter = Counter()
        self.route_counts: Counter = Counter()
        self.thr_gate_route: Dict[int, float] = {}
        self.thr_gate_global: float = 0.0

        # SemanticSpace (Part A): per-route deduplicated normal template manifold T_A[route]
        self.templates_a_route: Dict[int, np.ndarray] = {}
        self.templates_a_global: Optional[np.ndarray] = None
        self.thr_sem_route: Dict[int, float] = {}
        self.thr_sem_global: float = 0.0

        # SplitVerifier (Part B + A->B coupling): per-route normal response manifold T_B[route] + tuple surprisal
        self.templates_b_route: Dict[int, np.ndarray] = {}
        self.templates_b_global: Optional[np.ndarray] = None
        self.tuple_counts_b: Counter = Counter()
        self.thr_split_route: Dict[int, float] = {}
        self.thr_split_global: float = 0.0

        self.reset_calibration()

    def reset_calibration(self) -> None:
        # route -> list of blamed Part-A vectors learned from SplitVerifier misses
        self.bad_templates_a: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.bad_templates_a_mat: Dict[int, np.ndarray] = {}
        self.bad_tuples_a: Dict[int, set] = defaultdict(set)
        self.pending_quorum: Dict[int, List[ Tuple[np.ndarray, Tuple[int, ...]] ]] = defaultdict(list)
        self.stats = {
            "gate_caught": 0,
            "sem_caught": 0,
            "cal_caught": 0,
            "split_caught": 0,
            "split_flips": 0,
            "signatures_learned": 0,
        }

    @staticmethod
    def _route_bucket(ports: np.ndarray, major_ports: set) -> np.ndarray:
        out = np.empty(len(ports), dtype=np.int64)
        for i, p in enumerate(ports):
            pi = int(p)
            if pi in major_ports:
                out[i] = pi
            elif pi < 1024:
                out[i] = -1  # other well-known service port
            else:
                out[i] = -2  # ephemeral client port
        return out

    @staticmethod
    def _part_a_features(X: np.ndarray) -> np.ndarray:
        xa = signed_log2(np.nan_to_num(X[:, PART_A_COLS], nan=0.0, posinf=1e9, neginf=-1e9))
        # Exact structural ratios in bit space:
        bytes_per_pkt = xa[:, 1:2] - xa[:, 0:1]
        hdr_per_pkt = xa[:, 8:9] - xa[:, 0:1]
        act_ratio = xa[:, 19:20] - xa[:, 0:1]
        # Exact TCP stack fingerprint features (scaled finely so 20 vs 32 and 8192 vs 29200 are sharp)
        win_fwd_exact = np.nan_to_num(X[:, 66:67], 0.0) / 8192.0
        min_seg_exact = np.nan_to_num(X[:, 69:70], 0.0) / 20.0
        return np.hstack([xa, bytes_per_pkt, hdr_per_pkt, act_ratio, win_fwd_exact, min_seg_exact]).astype(np.float32)

    @staticmethod
    def _part_b_features(X: np.ndarray) -> np.ndarray:
        xb = signed_log2(np.nan_to_num(X[:, PART_B_COLS], nan=0.0, posinf=1e9, neginf=-1e9))
        xa = signed_log2(np.nan_to_num(X[:, PART_A_COLS], nan=0.0, posinf=1e9, neginf=-1e9))
        # Joint A->B coupling invariants (how Part B responds to Part A):
        byte_asym = xb[:, 2:3] - xa[:, 1:2]
        pkt_asym = xb[:, 1:2] - xa[:, 0:1]
        dur_per_pkt = xb[:, 0:1] - np.maximum(xa[:, 0:1], xb[:, 1:2])
        win_gap = xb[:, 31:32] - xa[:, 18:19]
        iat_ratio = xb[:, 15:16] - xb[:, 0:1]
        # Zero-payload established connection indicator (FwdPkts >= 3, BwdPkts >= 1, FwdBytes == 0, BwdBytes == 0)
        empty_est = ((X[:, 2] >= 3) & (X[:, 3] >= 1) & (X[:, 4] == 0) & (X[:, 5] == 0)).astype(np.float32)[:, None] * 4.0
        return np.hstack([xb, byte_asym, pkt_asym, dur_per_pkt, win_gap, iat_ratio, empty_est]).astype(np.float32)

    @staticmethod
    def _make_tuples_a(X: np.ndarray) -> List[Tuple[int, ...]]:
        p = X[:, 0].astype(np.int64)
        fp = np.clip(X[:, 2], 0, 64).astype(np.int64)
        fb = np.round(signed_log2(np.maximum(0.0, X[:, 4])) * 2.0).astype(np.int64)
        fmax = np.round(signed_log2(np.maximum(0.0, X[:, 6])) * 2.0).astype(np.int64)
        wf = X[:, 66].astype(np.int64)
        ms = X[:, 69].astype(np.int64)
        return list(zip(p.tolist(), fp.tolist(), fb.tolist(), fmax.tolist(), wf.tolist(), ms.tolist()))

    @staticmethod
    def _make_tuples_b(X: np.ndarray) -> List[Tuple[int, ...]]:
        p = X[:, 0].astype(np.int64)
        bp = np.clip(X[:, 3], 0, 64).astype(np.int64)
        bb = np.round(signed_log2(np.maximum(0.0, X[:, 5])) * 2.0).astype(np.int64)
        bmax = np.round(signed_log2(np.maximum(0.0, X[:, 10])) * 2.0).astype(np.int64)
        wb = X[:, 67].astype(np.int64)
        dur = np.round(signed_log2(np.maximum(0.0, X[:, 1]))).astype(np.int64)
        return list(zip(p.tolist(), bp.tolist(), bb.tolist(), bmax.tolist(), wb.tolist(), dur.tolist()))

    @staticmethod
    def _dedup_templates(Z: np.ndarray, max_k: int = 1200) -> np.ndarray:
        if len(Z) == 0:
            return Z
        _, uniq_idx = np.unique(np.round(Z * 3.0).astype(np.int32), axis=0, return_index=True)
        if len(uniq_idx) > max_k:
            rng = np.random.default_rng(42)
            uniq_idx = rng.choice(uniq_idx, max_k, replace=False)
        return np.ascontiguousarray(Z[uniq_idx], dtype=np.float32)

    def _compute_gate_scores(self, X: np.ndarray, tuples_a: List[Tuple[int, ...]]) -> np.ndarray:
        """EntropyGate score: Part-A tuple surprisal + sliding-window burst concentration."""
        n = len(X)
        denom = float(self.n_base) + 0.2
        surp_a = np.array(
            [-np.log2((self.tuple_counts_a.get(t, 0) + 0.1) / denom) for t in tuples_a],
            dtype=np.float32,
        )
        W = 32
        burst_bonus = np.zeros(n, dtype=np.float32)
        ports = X[:, 0].astype(np.int64)
        thash = np.array([hash(t) for t in tuples_a], dtype=np.int64)
        for i in range(n):
            s = max(0, i - W + 1)
            win_len = i - s + 1
            if win_len >= 8:
                n_ports = len(set(ports[s : i + 1].tolist()))
                if n_ports >= 12 and X[i, 4] <= 6:
                    burst_bonus[i] = max(burst_bonus[i], float(n_ports) * 0.6)
                if surp_a[i] > 9.5:
                    rep = int(np.sum(thash[s : i + 1] == thash[i]))
                    if rep >= 4:
                        burst_bonus[i] = max(burst_bonus[i], float(rep) * 0.45)
        return surp_a + burst_bonus

    def _compute_sem_scores(self, za: np.ndarray, rbuckets: np.ndarray) -> np.ndarray:
        """SemanticSpace score: Part-A L2 distance to nearest normal template on the same route."""
        dists = min_l2_dist(za, self.templates_a_global)
        for rb, T_sub in self.templates_a_route.items():
            mask = rbuckets == rb
            if np.any(mask):
                dists[mask] = min_l2_dist(za[mask], T_sub)
        return dists

    def _compute_split_scores(
        self,
        X: np.ndarray,
        zb: np.ndarray,
        rbuckets: np.ndarray,
        tuples_b: List[Tuple[int, ...]],
    ) -> np.ndarray:
        """SplitVerifier score: Part-B manifold distance + Part-B response tuple surprisal + burst anomaly."""
        dists_b = min_l2_dist(zb, self.templates_b_global)
        for rb, T_sub in self.templates_b_route.items():
            mask = rbuckets == rb
            if np.any(mask):
                dists_b[mask] = min_l2_dist(zb[mask], T_sub)

        denom = float(self.n_base) + 0.2
        surp_b = np.array(
            [-np.log2((self.tuple_counts_b.get(t, 0) + 0.1) / denom) for t in tuples_b],
            dtype=np.float32,
        )
        surp_bonus = np.maximum(0.0, (surp_b - 13.5) * 0.35)

        n = len(X)
        W = 40
        thash_b = np.array([hash(t) for t in tuples_b], dtype=np.int64)
        burst_b = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if surp_b[i] > 10.0:
                s = max(0, i - W + 1)
                rep = int(np.sum(thash_b[s : i + 1] == thash_b[i]))
                if rep >= 5:
                    burst_b[i] = float(rep) * 0.25

        return dists_b + surp_bonus + burst_b
    def fit(self, X_benign: np.ndarray) -> None:
        """Fit all layers exclusively on BENIGN flows (split 60/40 into template bank & quantile calibration)."""
        split_idx = int(len(X_benign) * 0.6)
        b1, b2 = X_benign[:split_idx], X_benign[split_idx:]
        self.n_base = len(b1)

        ports1 = b1[:, ROUTE_COL].astype(np.int64)
        self.route_counts = Counter(int(p) for p in ports1)
        self.major_ports = {p for p, c in self.route_counts.items() if c >= 80}

        rb1 = self._route_bucket(ports1, self.major_ports)
        fa1 = self._part_a_features(b1)
        fb1 = self._part_b_features(b1)

        self.mu_a = fa1.mean(axis=0)
        self.sd_a = np.maximum(fa1.std(axis=0), 0.20)
        self.mu_b = fb1.mean(axis=0)
        self.sd_b = np.maximum(fb1.std(axis=0), 0.20)

        za1 = (fa1 - self.mu_a) / self.sd_a
        zb1 = (fb1 - self.mu_b) / self.sd_b

        # Build per-route normal template banks (Part A and Part B)
        self.templates_a_global = self._dedup_templates(za1, max_k=1500)
        self.templates_b_global = self._dedup_templates(zb1, max_k=1500)
        for rb in set(rb1.tolist()):
            m = rb1 == rb
            if m.sum() >= 30:
                self.templates_a_route[rb] = self._dedup_templates(za1[m], max_k=1200)
                self.templates_b_route[rb] = self._dedup_templates(zb1[m], max_k=1200)

        # Build discrete tuple frequency tables on b1
        self.tuple_counts_a = Counter(self._make_tuples_a(b1))
        self.tuple_counts_b = Counter(self._make_tuples_b(b1))

        # Calibrate per-route thresholds on held-out benign b2
        ports2 = b2[:, ROUTE_COL].astype(np.int64)
        rb2 = self._route_bucket(ports2, self.major_ports)
        za2 = (self._part_a_features(b2) - self.mu_a) / self.sd_a
        zb2 = (self._part_b_features(b2) - self.mu_b) / self.sd_b
        tuples_a2 = self._make_tuples_a(b2)
        tuples_b2 = self._make_tuples_b(b2)

        gate2 = self._compute_gate_scores(b2, tuples_a2)
        sem2 = self._compute_sem_scores(za2, rb2)
        split2 = self._compute_split_scores(b2, zb2, rb2, tuples_b2)

        q_target = 1.0 - self.fpr
        self.thr_gate_global = float(np.quantile(gate2, q_target)) + 0.5
        self.thr_sem_global = float(np.quantile(sem2, q_target)) * 1.15 + 0.15
        self.thr_split_global = float(np.quantile(split2, q_target)) * 1.15 + 0.15

        for rb in set(rb2.tolist()):
            m = rb2 == rb
            if m.sum() >= 30:
                # Add a small safety margin above the held-out quantile to prevent drift false positives
                self.thr_gate_route[rb] = float(np.quantile(gate2[m], q_target)) + 0.5
                self.thr_sem_route[rb] = float(np.quantile(sem2[m], q_target)) * 1.15 + 0.15
                self.thr_split_route[rb] = float(np.quantile(split2[m], q_target)) * 1.15 + 0.15

        self.reset_calibration()

    def _blame_and_calibrate(
        self,
        rb: int,
        batch: List[Tuple[np.ndarray, Tuple[int, ...]]],
    ) -> None:
        """Blame the Part-A templates and discrete tuples of a Split-confirmed quorum."""
        vecs = np.vstack([b[0] for b in batch])
        mean_vec = vecs.mean(axis=0).astype(np.float32)

        existing = self.bad_templates_a[rb]
        if len(existing) == 0:
            existing.append(mean_vec)
            self.bad_templates_a_mat[rb] = np.vstack(existing)
            self.stats["signatures_learned"] += 1
        else:
            mat = self.bad_templates_a_mat[rb]
            d_min = float(np.sqrt(np.sum((mat - mean_vec) ** 2, axis=1).min()))
            if d_min > 0.35 and len(existing) < 64:
                existing.append(mean_vec)
                self.bad_templates_a_mat[rb] = np.vstack(existing)
                self.stats["signatures_learned"] += 1

        # Also record any Part-A discrete tuple that was not a frequent benign staple on this route
        for _, tup in batch:
            if self.tuple_counts_a.get(tup, 0) < 25:
                self.bad_tuples_a[rb].add(tup)

    def evaluate_stream(
        self,
        X: np.ndarray,
        mode: str = "full_online",
    ) -> np.ndarray:
        """Evaluate SSE on a stream of flows X."""
        ports = X[:, ROUTE_COL].astype(np.int64)
        rbuckets = self._route_bucket(ports, self.major_ports)
        tuples_a = self._make_tuples_a(X)

        gate_scores = self._compute_gate_scores(X, tuples_a)
        thr_gate_vec = np.array([self.thr_gate_route.get(int(rb), self.thr_gate_global) for rb in rbuckets], dtype=np.float32)

        if mode == "gate_only":
            return gate_scores > thr_gate_vec

        za = (self._part_a_features(X) - self.mu_a) / self.sd_a
        sem_scores = self._compute_sem_scores(za, rbuckets)
        thr_sem_vec = np.array([self.thr_sem_route.get(int(rb), self.thr_sem_global) for rb in rbuckets], dtype=np.float32)

        if mode == "gate_plus_sem":
            return (gate_scores > thr_gate_vec) | (sem_scores > thr_sem_vec)

        zb = (self._part_b_features(X) - self.mu_b) / self.sd_b
        tuples_b = self._make_tuples_b(X)
        split_scores = self._compute_split_scores(X, zb, rbuckets, tuples_b)
        thr_split_vec = np.array([self.thr_split_route.get(int(rb), self.thr_split_global) for rb in rbuckets], dtype=np.float32)

        if mode == "split_uncalibrated":
            return (gate_scores > thr_gate_vec) | (sem_scores > thr_sem_vec) | (split_scores > thr_split_vec)

        preds = np.zeros(len(X), dtype=bool)
        for i in range(len(X)):
            rb = int(rbuckets[i])

            # 1. Front Gate A: EntropyGate eats high-surprisal / burst rubble
            if gate_scores[i] > thr_gate_vec[i]:
                preds[i] = True
                self.stats["gate_caught"] += 1
                continue

            # 2. Front Gate B: SemanticSpace eats off-manifold Part-A requests
            d_norm_a = float(sem_scores[i])
            if d_norm_a > thr_sem_vec[i]:
                preds[i] = True
                self.stats["sem_caught"] += 1
                continue

            # 3. Calibrated Front Gate: check if Part A matches a Split-blamed template closer than normal manifold
            tup_a = tuples_a[i]
            if tup_a in self.bad_tuples_a.get(rb, ()):
                preds[i] = True
                self.stats["cal_caught"] += 1
                continue

            bad_mat = self.bad_templates_a_mat.get(rb)
            if bad_mat is not None and d_norm_a > 0.25:
                z_row = za[i]
                d_bad = float(np.sqrt(np.sum((bad_mat - z_row) ** 2, axis=1).min()))
                if d_bad < d_norm_a * 0.85 and d_bad < 1.5:
                    preds[i] = True
                    self.stats["cal_caught"] += 1
                    continue

            # 4. Passed Front Gates -> Hand to SplitVerifier (Part B response + A->B coupling check)
            if split_scores[i] > thr_split_vec[i]:
                preds[i] = True
                self.stats["split_caught"] += 1
                buf = self.pending_quorum[rb]
                buf.append((za[i], tup_a))
                if len(buf) >= self.quorum:
                    self.stats["split_flips"] += 1
                    self._blame_and_calibrate(rb, buf)
                    self.pending_quorum[rb] = []

        return preds
