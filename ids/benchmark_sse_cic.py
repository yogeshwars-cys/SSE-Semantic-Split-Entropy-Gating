"""benchmark_sse_cic.py: Evaluate Unfixed vs Fixed SSE IDS on official CIC-IDS2017 CSV datasets.

Fit protocol (strictly zero-shot anomaly / closed-loop self-calibration):
  - Training set: first 40,000 BENIGN flows of each capture file (0 attack labels seen during fit).
  - Test stream: remaining benign flows + all attack flows in chronological stream order.
  - At test time, NO ground-truth labels are ever shown to SplitVerifier or BlameCalibrator.
"""
from __future__ import annotations

import csv
import json
import pathlib
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from sse_ids import FixedSSEIDS, UnfixedSSEIDS

DATA_DIR = pathlib.Path(r"D:\dev-cache\datasets\cic-ids2017")
FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
]


def load_cic_csv(path: pathlib.Path, max_test_rows: int = 120000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load CIC-IDS2017 CSV into (X_benign_train, X_test, test_labels) preserving chronological order.

    Training baseline is drawn from calm BENIGN segments (not interleaved mid-attack teardown packets),
    matching the gates' baseline of normal traffic.
    """
    all_vals = []
    all_labels = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        _ = next(reader)  # header
        for row in reader:
            if len(row) < 79:
                continue
            label = row[78].strip().replace("\ufffd", "-")
            try:
                vals = [float(x) if x not in ("Infinity", "NaN", "inf", "-inf", "") else 0.0 for x in row[:78]]
            except ValueError:
                continue
            all_vals.append(vals)
            all_labels.append(label)

    n_total = len(all_labels)
    is_atk = np.array([l != "BENIGN" for l in all_labels], dtype=np.float32)
    # A row is in a calm normal period if it is BENIGN and no attack occurs in the local 21-flow window
    atk_near = np.convolve(is_atk, np.ones(21, dtype=np.float32), mode="same") > 0
    calm_benign_idx = [i for i in range(n_total) if all_labels[i] == "BENIGN" and not atk_near[i]]

    n_train_target = min(36000, len(calm_benign_idx) // 2)
    # Take evenly spaced calm segments across the capture's normal periods for baseline + threshold calibration
    step_tr = max(1, len(calm_benign_idx) // n_train_target)
    train_set = set(calm_benign_idx[::step_tr][:n_train_target])

    benign_train = [all_vals[i] for i in sorted(train_set)]
    test_rows = [all_vals[i] for i in range(n_total) if i not in train_set]
    test_labels = [all_labels[i] for i in range(n_total) if i not in train_set]

    if len(test_rows) > max_test_rows:
        step = len(test_rows) / float(max_test_rows)
        idx = [int(i * step) for i in range(max_test_rows)]
        test_rows = [test_rows[i] for i in idx]
        test_labels = [test_labels[i] for i in idx]

    X_train = np.array(benign_train, dtype=np.float32)
    X_test = np.array(test_rows, dtype=np.float32)
    return X_train, X_test, test_labels


def compute_metrics(preds: np.ndarray, y_true: np.ndarray, labels: List[str]) -> Dict:
    tp = int(np.sum(preds & y_true))
    fp = int(np.sum(preds & ~y_true))
    tn = int(np.sum(~preds & ~y_true))
    fn = int(np.sum(~preds & y_true))

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * tpr / max(prec + tpr, 1e-12)
    acc = (tp + tn) / max(len(y_true), 1)

    per_attack = {}
    attack_types = sorted({l for l in labels if l != "BENIGN"})
    for at in attack_types:
        mask = np.array([l == at for l in labels], dtype=bool)
        per_attack[at] = {
            "count": int(mask.sum()),
            "recall": float(preds[mask].mean()) if mask.any() else 0.0,
        }

    return {
        "precision": float(prec),
        "recall_tpr": float(tpr),
        "fpr": float(fpr),
        "f1": float(f1),
        "accuracy": float(acc),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "per_attack": per_attack,
    }


def main() -> None:
    all_results = {}
    print("=" * 112)
    print(f"{'Dataset / Method':<46} | {'Prec':>6} | {'Recall':>6} | {'FPR':>6} | {'F1':>6} | {'Acc':>6} | {'Gate/Sem/Cal/Split Breakdown'}")
    print("=" * 112)

    for fname in FILES:
        fpath = DATA_DIR / fname
        short = fname.split(".")[0].replace("-WorkingHours", "").replace("-workingHours", "")
        t0 = time.perf_counter()
        X_train, X_test, labels = load_cic_csv(fpath)
        y_true = np.array([l != "BENIGN" for l in labels], dtype=bool)
        n_ben = int((~y_true).sum())
        n_atk = int(y_true.sum())
        print(f"\n[{short}] Train Benign: {len(X_train):,d} | Test Stream: {len(X_test):,d} (Benign: {n_ben:,d}, Attack: {n_atk:,d}) [{time.perf_counter()-t0:.1f}s]")
        print("-" * 112)

        ds_res = {}

        # 1. Unfixed SSE (Gates only)
        unfixed = UnfixedSSEIDS(fpr=0.02)
        unfixed.fit(X_train)
        p_unf_gate = unfixed.evaluate_stream(X_test, calibrate=False)
        m_unf_gate = compute_metrics(p_unf_gate, y_true, labels)
        ds_res["unfixed_gates_only"] = m_unf_gate
        print(f"  {'1. Unfixed SSE (Gates only, digit-folded)':<44} | {m_unf_gate['precision']:6.4f} | {m_unf_gate['recall_tpr']:6.4f} | {m_unf_gate['fpr']:6.4f} | {m_unf_gate['f1']:6.4f} | {m_unf_gate['accuracy']:6.4f} | ---")

        # 2. Unfixed SSE + Scalar Calibration (thr = z - 0.25)
        p_unf_cal = unfixed.evaluate_stream(X_test, calibrate=True)
        m_unf_cal = compute_metrics(p_unf_cal, y_true, labels)
        ds_res["unfixed_scalar_cal"] = m_unf_cal
        print(f"  {'2. Unfixed SSE + Scalar Cal (z - 0.25 bug)':<44} | {m_unf_cal['precision']:6.4f} | {m_unf_cal['recall_tpr']:6.4f} | {m_unf_cal['fpr']:6.4f} | {m_unf_cal['f1']:6.4f} | {m_unf_cal['accuracy']:6.4f} | scalar port lockouts")

        # 3. Fixed SSE Ablations
        import pickle
        fixed = FixedSSEIDS(fpr=0.015, quorum=2)
        t_fit = time.perf_counter()
        fixed.fit(X_train)
        fit_s = time.perf_counter() - t_fit

        p_fg_only = fixed.evaluate_stream(X_test, mode="gate_only")
        m_fg_only = compute_metrics(p_fg_only, y_true, labels)
        ds_res["fixed_entropy_gate_only"] = m_fg_only
        print(f"  {'3. Fixed SSE: EntropyGate only (Part A)':<44} | {m_fg_only['precision']:6.4f} | {m_fg_only['recall_tpr']:6.4f} | {m_fg_only['fpr']:6.4f} | {m_fg_only['f1']:6.4f} | {m_fg_only['accuracy']:6.4f} | eats high-surprisal rubble")

        p_fgs = fixed.evaluate_stream(X_test, mode="gate_plus_sem")
        m_fgs = compute_metrics(p_fgs, y_true, labels)
        ds_res["fixed_gate_plus_sem"] = m_fgs
        print(f"  {'4. Fixed SSE: Gate + Semantic (Part A only)':<44} | {m_fgs['precision']:6.4f} | {m_fgs['recall_tpr']:6.4f} | {m_fgs['fpr']:6.4f} | {m_fgs['f1']:6.4f} | {m_fgs['accuracy']:6.4f} | + Part-A 4096-d TF-IDF")

        # 5. Full Closed-Loop Fixed SSE (Pass 1 online stream)
        fixed.reset_calibration()
        t_ev = time.perf_counter()
        p_full_p1 = fixed.evaluate_stream(X_test, mode="full_online")
        ev_s = time.perf_counter() - t_ev
        state_bytes = len(pickle.dumps(fixed.__dict__, protocol=pickle.HIGHEST_PROTOCOL))
        perf = {
            "fit_seconds": round(fit_s, 2),
            "full_online_seconds": round(ev_s, 2),
            "us_per_flow": round(ev_s / len(X_test) * 1e6, 1),
            "flows_per_second": int(len(X_test) / ev_s),
            "state_bytes_after_pass1": state_bytes,
        }
        ds_res["perf"] = perf
        print(f"  PERF: fit {perf['fit_seconds']}s | full-loop {perf['us_per_flow']} us/flow ({perf['flows_per_second']:,d} flows/s) | state {state_bytes/1e6:.2f} MB")
        m_full_p1 = compute_metrics(p_full_p1, y_true, labels)
        st_p1 = dict(fixed.stats)
        m_full_p1["stage_stats"] = st_p1
        ds_res["fixed_full_online_pass1"] = m_full_p1
        bd_p1 = f"G:{st_p1['gate_caught']} S:{st_p1['sem_caught']} Cal:{st_p1['cal_caught']} Sp:{st_p1['split_caught']} (sigs:{st_p1['signatures_learned']})"
        print(f"  {'5. Fixed SSE: Full Closed-Loop (Pass 1)':<44} | {m_full_p1['precision']:6.4f} | {m_full_p1['recall_tpr']:6.4f} | {m_full_p1['fpr']:6.4f} | {m_full_p1['f1']:6.4f} | {m_full_p1['accuracy']:6.4f} | {bd_p1}")

        # 6. Pass 2 with the Split-modified Front Gates (shows misses promoted to Front Gate!)
        fixed.stats = {k: 0 for k in fixed.stats}
        p_full_p2 = fixed.evaluate_stream(X_test, mode="full_online")
        m_full_p2 = compute_metrics(p_full_p2, y_true, labels)
        st_p2 = dict(fixed.stats)
        m_full_p2["stage_stats"] = st_p2
        ds_res["fixed_full_online_pass2"] = m_full_p2
        bd_p2 = f"G:{st_p2['gate_caught']} S:{st_p2['sem_caught']} Cal:{st_p2['cal_caught']} Sp:{st_p2['split_caught']}"
        print(f"  {'6. Fixed SSE: Warmed Front Gates (Pass 2)':<44} | {m_full_p2['precision']:6.4f} | {m_full_p2['recall_tpr']:6.4f} | {m_full_p2['fpr']:6.4f} | {m_full_p2['f1']:6.4f} | {m_full_p2['accuracy']:6.4f} | {bd_p2}")

        # Print per-attack recall comparison (Unfixed Gates vs Fixed Part-A Gates vs Fixed Closed-Loop SSE)
        print("  Per-Attack Recall Breakdown:")
        for at in m_full_p1["per_attack"]:
            cnt = m_full_p1["per_attack"][at]["count"]
            r_unf = m_unf_gate["per_attack"][at]["recall"]
            r_fgs = m_fgs["per_attack"][at]["recall"]
            r_p1 = m_full_p1["per_attack"][at]["recall"]
            r_p2 = m_full_p2["per_attack"][at]["recall"]
            print(f"    - {at:<28} (n={cnt:6d}): Unfixed={r_unf*100:5.1f}% -> PartA-Gates={r_fgs*100:5.1f}% -> Full-SSE(P1)={r_p1*100:5.1f}% -> Warmed(P2)={r_p2*100:5.1f}%")

        all_results[short] = ds_res

    out_path = pathlib.Path("benchmark_results/sse/sse_cic_ids2017_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print("\nSaved detailed results to", out_path)


if __name__ == "__main__":
    main()
