"""Run the SSE gates (sse/anomaly_model.py + sse/semantic_model.py) on SR-BH 2020.

SR-BH 2020: 907k real HTTP requests to a WordPress honeypot, 12 days, 13 CAPEC labels.
Protocol (zero-shot, benign-only training):
  * temporal split: benign requests from the first 60% of the capture fit the gates,
    everything evaluated comes from the last 40% (later traffic, unseen scanners);
  * thresholds come from held-out benign traffic only;
  * response_* columns are never read (status code / length would leak the label);
  * results at fixed false-alarm rates (1/2/5%) and ROC-AUC, plus the gates' own thresholds.

usage: python -I run_sse_srbh.py <data_capec_multilabel.csv> <sse dir> <out.json>
variants: stock = headers + parameter names (as SSE core ships it);
          fields = also @method/@path/@query/@body as value-scored fields (same algorithm, more inputs).
"""
import csv
import re
import json
import random
import sys
import time
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

CSV, REAL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, REAL)
from anomaly_model import Gate  # noqa: E402
from semantic_model import Semantic  # noqa: E402

SEED = 7
N_FIT = int(__import__("os").environ.get("NFIT", 10000))          # benign requests to fit (the gates split it half fit / half threshold)
N_EVAL_BENIGN = int(__import__("os").environ.get("NEV", 20000))
N_EVAL_ATTACK = int(__import__("os").environ.get("NEV", 20000))
PER_CLASS_MIN = 500
UNCLASSIFIED = "anomalous, no CAPEC class"     # top up rare classes so every class has a usable sample

HEADER_COLS = {
    "request_user_agent": "user-agent", "request_referer": "referer", "request_host": "host",
    "request_origin": "origin", "request_cookie": "cookie", "request_content_type": "content-type",
    "request_accept": "accept", "request_accept_language": "accept-language",
    "request_accept_encoding": "accept-encoding", "request_do_not_track": "dnt",
    "request_connection": "connection",
}


def to_request(row, with_fields=False):
    """(headers, param names) as SSE core's gates take them."""
    p = urllib.parse.urlparse(row["request_http_request"])
    h = {name: row[c] for c, name in HEADER_COLS.items() if row.get(c)}
    params = list(urllib.parse.parse_qs(p.query, keep_blank_values=True))
    body = row.get("request_body", "") or ""
    if body and "application/x-www-form-urlencoded" in h.get("content-type", ""):
        params += list(urllib.parse.parse_qs(body, keep_blank_values=True))
    if with_fields:
        h["@method"] = row["request_http_method"]
        h["@path"] = p.path or "/"
        if p.query:
            h["@query"] = urllib.parse.unquote_plus(p.query)
        if body:
            h["@body"] = urllib.parse.unquote_plus(body)
    return h, params


def load():
    csv.field_size_limit(10 ** 9)
    rows = []
    with open(CSV, encoding="utf-8", errors="replace", newline="") as f:
        rd = csv.DictReader(f)
        label_cols = [c for c in rd.fieldnames if re.match(r"^\d+ - ", c)]
        for row in rd:
            try:
                ts = datetime.strptime(row["timestamp"], "%d/%b/%Y:%H:%M:%S %z").timestamp()
            except ValueError:
                continue
            if row["000 - Normal"] == "1":
                labels = ["000 - Normal"]
            else:                             # anomalous; 272k rows carry no CAPEC class at all
                labels = [c for c in label_cols if c != "000 - Normal" and row[c] == "1"] or [UNCLASSIFIED]
            keep = {k: row[k] for k in ("request_http_method", "request_http_request", "request_body",
                                        "src_ip", *HEADER_COLS)}
            rows.append((ts, labels, keep))
    rows.sort(key=lambda t: t[0])
    return rows, [c for c in label_cols if c != "000 - Normal"] + [UNCLASSIFIED]


def tpr_at(benign, attack, fpr):
    thr = np.quantile(benign, 1 - fpr)
    return float((attack > thr).mean()), thr


def auc(benign, attack):
    s = np.concatenate([benign, attack])
    ranks = s.argsort().argsort().astype(np.float64) + 1
    # average ranks over ties
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    r = np.empty_like(ranks)
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    na, nb = len(attack), len(benign)
    return float((r[nb:].sum() - na * (na + 1) / 2) / (na * nb))


def survival(ref, x):
    """Empirical P(benign score >= x) using the held-out benign reference."""
    ref = np.sort(ref)
    return (len(ref) - np.searchsorted(ref, x, side="left") + 1) / (len(ref) + 1)


ATTACK_END = datetime.strptime("20/Jul/2020:00:00:00 +0100", "%d/%b/%Y:%H:%M:%S %z").timestamp()


def run(rows, attack_cols, with_path, protocol):
    rnd = random.Random(SEED)
    window = [r for r in rows if r[0] < ATTACK_END]          # 17-19 Jul: every attack is in here
    if protocol == "time_disjoint":                          # fit on the clean week after, test on 17-19 Jul
        fit_pool = [r for r in rows if r[0] >= ATTACK_END and r[1] == ["000 - Normal"]]
        late = window
    else:                                                    # same window, disjoint requests
        b = [r for r in window if r[1] == ["000 - Normal"]]
        rnd.shuffle(b)
        fit_pool, held = b[:len(b) // 2], {id(r) for r in b[len(b) // 2:]}
        late = [r for r in window if r[1] != ["000 - Normal"] or id(r) in held]
    fit = [to_request(r[2], with_path) for r in rnd.sample(fit_pool, min(N_FIT, len(fit_pool)))]

    late_benign = [r for r in late if r[1] == ["000 - Normal"]]
    late_attack = [r for r in late if r[1] != ["000 - Normal"]]
    ev_b = rnd.sample(late_benign, min(N_EVAL_BENIGN, len(late_benign)))
    ev_a = rnd.sample(late_attack, min(N_EVAL_ATTACK, len(late_attack)))
    chosen = {id(r) for r in ev_a}
    by_class = defaultdict(list)
    for r in late_attack:
        for c in r[1]:
            by_class[c].append(r)
    for c in attack_cols:                     # top up rare classes
        have = sum(1 for r in ev_a if c in r[1])
        extra = [r for r in by_class[c] if id(r) not in chosen]
        rnd.shuffle(extra)
        for r in extra[:max(0, PER_CLASS_MIN - have)]:
            ev_a.append(r)
            chosen.add(id(r))

    t0 = time.perf_counter()
    eg = Gate(fpr=0.01); eg.fit(fit)
    sg = Semantic(fpr=0.01); sg.fit(fit)
    fit_s = time.perf_counter() - t0
    mem_mb = sum(s.MT.nbytes for s in sg.space.values()) / 1e6

    # held-out benign reference for score normalisation = the gates' own threshold half
    ref = fit[len(fit) // 2:]
    ref_e = np.array([eg.score(*r) for r in ref]); ref_s = np.array([sg.score(*r) for r in ref])

    def scores(rs):
        reqs = [to_request(r[2], with_path) for r in rs]
        e, s, lat = [], [], []
        for q in reqs:
            t = time.perf_counter()
            e.append(eg.score(*q)); s.append(sg.score(*q))
            lat.append((time.perf_counter() - t) * 1e3)
        return np.array(e), np.array(s), np.array(lat)

    be, bs, blat = scores(ev_b)
    ae, as_, alat = scores(ev_a)
    def norm(ref, x):                         # 0 at the benign median, 1 at the benign 99th percentile
        med, q99 = np.median(ref), np.quantile(ref, 0.99)
        return (x - med) / ((q99 - med) or 1.0)
    comb = lambda e, s: np.maximum(norm(ref_e, e), norm(ref_s, s))  # noqa: E731
    bc, ac = comb(be, bs), comb(ae, as_)

    res = {"fit_requests": len(fit), "fit_seconds": round(fit_s, 2), "semantic_matrices_mb": round(mem_mb, 1),
           "semantic_fields": sorted(sg.space),
           "eval_benign": len(ev_b), "eval_attack": len(ev_a),
           "latency_ms": {"p50": float(np.percentile(np.r_[blat, alat], 50)),
                          "p95": float(np.percentile(np.r_[blat, alat], 95)),
                          "p99": float(np.percentile(np.r_[blat, alat], 99))}}

    # 1. as shipped: each gate at its own 1% threshold, OR cascade
    pe_b, ps_b, pe_a, ps_a = be > eg.thr, bs > sg.thr, ae > eg.thr, as_ > sg.thr
    pc_b, pc_a = pe_b | ps_b, pe_a | ps_a
    def conf(pb, pa):
        tp, fn, fp, tn = int(pa.sum()), int((~pa).sum()), int(pb.sum()), int((~pb).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        return {"tpr": rec, "fpr": fp / (fp + tn), "precision": prec,
                "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    res["as_shipped"] = {"entropy_gate": conf(pe_b, pe_a), "semantic_gate": conf(ps_b, ps_a),
                         "cascade": conf(pc_b, pc_a)}

    # 2. fixed false-alarm rates on later benign traffic, and AUC
    res["matched_fpr"] = {}
    # naive reference: total URI + body length (no learning at all)
    blen = np.array([len(r[2]["request_http_request"] or "") + len(r[2]["request_body"] or "") for r in ev_b], float)
    alen = np.array([len(r[2]["request_http_request"] or "") + len(r[2]["request_body"] or "") for r in ev_a], float)
    for name, b, a in (("baseline_uri_body_length", blen, alen), ("entropy_gate", be, ae),
                       ("semantic_gate", bs, as_), ("combined", bc, ac)):
        res["matched_fpr"][name] = {"auc": auc(b, a),
                                    **{f"tpr@fpr{int(f*100)}%": tpr_at(b, a, f)[0] for f in (0.01, 0.02, 0.05)}}

    # 3. per class (multi-label: a request counts for every class it carries), combined score
    res["per_class"] = {}
    thr = {f: np.quantile(bc, 1 - f) for f in (0.01, 0.05)}
    for c in attack_cols:
        m = np.array([c in r[1] for r in ev_a])
        if m.sum() == 0:
            continue
        res["per_class"][c] = {"n": int(m.sum()), "auc": auc(bc, ac[m]),
                               "tpr@fpr1%": float((ac[m] > thr[0.01]).mean()),
                               "tpr@fpr5%": float((ac[m] > thr[0.05]).mean()),
                               "cascade_as_shipped_tpr": float(pc_a[m].mean())}
    return res


def main():
    t0 = time.perf_counter()
    rows, attack_cols = load()
    lab = Counter(c for r in rows for c in r[1])
    out = {"dataset": "SR-BH 2020 (data_capec_multilabel.csv, Harvard Dataverse doi:10.7910/DVN/OGOIXX)",
           "rows": len(rows), "label_counts": dict(lab),
           "src_ips_all_rows": Counter(r[2]["src_ip"] for r in rows).most_common(3),
           "load_seconds": round(time.perf_counter() - t0, 1),
           "split": "time_disjoint: fit on benign 20-29 Jul, eval 17-19 Jul; same_window: 17-19 Jul benign halved"}
    print(json.dumps({k: out[k] for k in ("rows", "label_counts", "src_ips_all_rows")}, indent=1), flush=True)
    for protocol in ("time_disjoint", "same_window"):
        for variant, wp in (("sse_stock", False), ("sse_plus_request_fields", True)):
            key = f"{protocol}/{variant}"
            print(f"\n=== {key}", flush=True)
            out[key] = run(rows, attack_cols, wp, protocol)
            print(json.dumps({k: out[key][k] for k in ("eval_benign", "eval_attack", "as_shipped", "matched_fpr")},
                             indent=1), flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
