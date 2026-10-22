"""master_table.py: every method on every IDS dataset, one table set: results/bench/MASTER.md (+ MASTER.json).

Rows: C++ SSE (exact-route variant, lap 1 and lap 5; token-only lap 1), the two CRS rule sets at their best paranoia
level (fewest wrong verdicts, picked on the test set, which favours them), the character models, and the tabular
baselines. Columns: the 12 test sets. "-" = not applicable (rule sets and character models read HTTP only) or not run.

usage: python -I bench/master_table.py
"""
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "bench")
DATASETS = [("srbh", "SR-BH"), ("csic", "CSIC"), ("cic_Friday-Afternoon-DDos", "CIC DDoS"),
            ("cic_Friday-Afternoon-PortScan", "CIC Scan"), ("cic_Thursday-Morning-WebAttacks", "CIC Web"),
            ("cic_Wednesday", "CIC DoS"), ("unsw", "UNSW"), ("kitsune_arp_mitm", "K ARP"), ("kitsune_mirai", "K Mirai*"),
            ("kitsune_os_scan", "K Scan"), ("kitsune_ssdp_flood", "K SSDP"), ("kitsune_syn_dos", "K SYN")]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def lap(j, k):
    l = j["lap1"] if k == 1 else j["belt"][str(k)]
    p, r = l["tp"] / max(1, l["tp"] + l["fp"]), l["tp"] / max(1, l["tp"] + l["fn"])
    return {**l, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def sse(d, variant, k):
    j = load(f"{RES}/{d}/sse_cpp{variant}.json")
    if not j:
        return None
    r = lap(j, k)
    r["size_mb"] = (j["semantic_bytes_after_lap1"] + j["route_state"]["bytes"]) / 1e6
    r["us"] = j["frozen_after_lap1"]["us_per_request_mean"]
    return r


def waf(d, engine):
    j = load(f"{RES}/{d}/{engine}.json")
    if not j:
        return None
    pls = [k for k in j if k.startswith("pl")]
    if not pls:
        return None
    k = min(pls, key=lambda k: j[k]["wrong"])
    r = dict(j[k])
    r["note"] = k.upper()
    r["size_mb"] = r.get("peak_rss_mb")
    r["us"] = (r.get("ms_per_request") or 0) * 1e3
    return r


def model(d, file, name):
    j = load(f"{RES}/{d}/{file}.json")
    r = j.get(name) if j else None
    if not r:
        return None
    r = dict(r)
    r["size_mb"] = r["model_bytes"] / 1e6
    r["us"] = r["us_per_record_single"]
    return r


ROWS = [("SSE C++ binary, exact routes, lap 1", lambda d: sse(d, "_exact", 1)),
        ("SSE C++ binary, exact routes, lap 5", lambda d: sse(d, "_exact", 5)),
        ("SSE C++ binary, token routes only, lap 1", lambda d: sse(d, "", 1)),
        ("ModSecurity + CRS 3.3.5 (best PL)", lambda d: waf(d, "modsec")),
        ("Coraza + CRS 4.25 (best PL)", lambda d: waf(d, "coraza")),
        ("char-CNN (supervised)", lambda d: model(d, "charnn", "char_cnn")),
        ("char-BiLSTM (supervised)", lambda d: model(d, "charnn", "char_bilstm")),
        ("Isolation Forest", lambda d: model(d, "isolation_forest", "isolation_forest")),
        ("One-Class SVM", lambda d: model(d, "ocsvm", "ocsvm")),
        ("KitNET (Kitsune)", lambda d: model(d, "kitnet", "kitnet")),
        ("Random Forest (supervised)", lambda d: model(d, "random_forest", "random_forest")),
        ("XGBoost (supervised)", lambda d: model(d, "xgboost", "xgboost")),
        ("CatBoost (supervised)", lambda d: model(d, "catboost", "catboost"))]


def main():
    grid = {name: {d: f(d) for d, _ in DATASETS} for name, f in ROWS}
    head = "| Method | " + " | ".join(s for _, s in DATASETS) + " |"
    sep = "|---|" + "---|" * len(DATASETS)

    def table(title, cell):
        out = [f"## {title}", "", head, sep]
        for name, _ in ROWS:
            out.append(f"| {name} | " + " | ".join(cell(grid[name][d]) if grid[name][d] else "-" for d, _ in DATASETS) + " |")
        return out + [""]

    lines = ["# IDS benchmark: master table", "",
             "Same test rows for every method on a dataset. SSE learns online from the label of every wrong verdict "
             "(an oracle teacher: an upper bound of the loop), laps 2-5 replay the same stream. Unsupervised baselines "
             "see normal traffic only, thresholded at 1% false alarms on held-out normal traffic; supervised baselines "
             "train on labelled attacks (SR-BH: 17 Jul; UNSW: official training set; others: 2-fold cross-validation "
             "over the test stream, optimistic on flow captures). CRS rows: paranoia level with the fewest wrong verdicts, "
             "picked on the test set. *Kitsune Mirai's test stream has no normal traffic (false alarms undefined). "
             "Protocols: bench/*.py docstrings.", ""]
    lines += table("F1", lambda r: f"{r['f1']:.3f}")
    lines += table("Detection / false alarms", lambda r: f"{r['tpr']:.1%} / {r['fpr']:.1%}" + (f" ({r['note']})" if r.get("note") else ""))
    lines += table("Wrong verdicts", lambda r: f"{r['wrong']:,}")
    lines += ["## Size and speed (median over the datasets a method ran on)", "",
              "| Method | model / state size | us per record | datasets |", "|---|---|---|---|"]
    for name, _ in ROWS:
        rs = [r for r in grid[name].values() if r]
        if not rs:
            lines.append(f"| {name} | - | - | 0 |")
            continue
        sz = [r["size_mb"] for r in rs if r.get("size_mb") is not None]
        us = [r["us"] for r in rs if r.get("us")]
        lines.append(f"| {name} | {statistics.median(sz):.2f} MB | {statistics.median(us):.0f} | {len(rs)} |"
                     if sz and us else f"| {name} | - | - | {len(rs)} |")
    lines += ["", "SSE size = semantic store after lap 1 + threshold tables; its time = inference only (frozen pass), "
              "one core. CRS size = process peak RSS, time = ms/request x 1000. Others: pickled / saved model, one "
              "record at a time. Runs shared the machine, so times carry contention."]
    open(os.path.join(RES, "MASTER.md"), "w").write("\n".join(lines) + "\n")
    json.dump(grid, open(os.path.join(RES, "MASTER.json"), "w"), indent=1, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
