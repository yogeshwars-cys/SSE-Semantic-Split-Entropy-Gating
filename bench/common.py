"""common.py: metrics, thresholds and result files shared by every IDS benchmark runner.

Every method is scored the same way on a dataset's test set:
  verdict metrics  detection (TPR), false alarms (FPR), precision, F1, wrong verdicts, per-class detection, at the
                   method's own operating point (see `threshold` below), and
  ranking metrics  ROC-AUC and detection at 1% / 5% / 10% false alarms, thresholds read off the test normals (an
                   oracle setting that no deployed detector has; it compares score quality, not operating points).
Operating points:
  unsupervised     fit on normal traffic only; threshold at the (1 - fpr) quantile of scores on held-out normal
                   traffic (the SSE gates' own protocol: fit on one half of the normal sample, threshold on the other)
  supervised       probability >= 0.5 from a model trained on labelled traffic, attacks included
  rule-based WAF   CRS inbound anomaly score >= 5 (the CRS default blocking threshold)
"""
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "bench")


def auc(y, s):
    """ROC-AUC with ties averaged (Mann-Whitney U)."""
    y = np.asarray(y, bool)
    s = np.asarray(s, np.float64)
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(ss):
        j = np.searchsorted(ss, ss[i], side="right")
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    npos, nneg = y.sum(), (~y).sum()
    if not npos or not nneg:
        return float("nan")
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def tpr_at_fpr(y, s, fpr):
    y = np.asarray(y, bool)
    s = np.asarray(s, np.float64)
    if y.all() or not y.any():                                   # no normals (Kitsune Mirai) or no attacks
        return float("nan")
    thr = np.quantile(s[~y], 1 - fpr)
    return float((s[y] > thr).mean())


def verdicts(y, pred, classes=None):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    prec = tp / max(1, tp + fp)
    tpr = tp / max(1, tp + fn)
    out = {"tpr": tpr, "fpr": fp / max(1, fp + tn), "precision": prec,
           "f1": 2 * prec * tpr / (prec + tpr) if prec + tpr else 0.0, "wrong": fp + fn,
           "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    if classes is not None:
        pc = {}
        for c, p in zip(classes, pred):
            for k in (c if isinstance(c, (list, tuple)) else [c]):
                if k:
                    h = pc.setdefault(k, [0, 0])
                    h[0] += bool(p)
                    h[1] += 1
        out["per_class_tpr"] = {k: h[0] / h[1] for k, h in sorted(pc.items())}
    return out


def ranking(y, s):
    return {"auc": auc(y, s), **{f"tpr@fpr{int(f * 100)}%": tpr_at_fpr(y, s, f) for f in (0.01, 0.05, 0.10)}}


def report(y, score, pred, classes=None, **extra):
    """Verdict + ranking metrics of one method on one test set."""
    out = verdicts(y, pred, classes)
    if score is not None:
        out.update(ranking(y, score))
    out.update(extra)
    return out


def quantile_threshold(s_holdout, fpr):
    return float(np.quantile(np.asarray(s_holdout, np.float64), 1 - fpr))


class Timer:
    def __init__(self):
        self.t = time.perf_counter()

    def __call__(self):
        t, self.t = self.t, time.perf_counter()
        return self.t - t


def save(path, obj):
    path = os.path.join(RESULTS, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    return path


def line(name, r):
    s = f"  {name:34s} tpr={r['tpr']:.4f} fpr={r['fpr']:.4f} prec={r['precision']:.4f} f1={r['f1']:.4f} wrong={r['wrong']}"
    if "auc" in r:
        s += f" auc={r['auc']:.4f} tpr@1%={r['tpr@fpr1%']:.4f}"
    return s
