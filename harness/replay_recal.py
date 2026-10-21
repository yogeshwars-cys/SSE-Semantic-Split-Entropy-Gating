"""Real-time replay of SR-BH 2020 (17-19 Jul, every attack) through SSE with recalibration on every miss.

A request is odd if the statistical gate or the semantic layer flags it, or if the Calibrator's per-(route,
field) thresholds do. When an attack gets through (a miss), the teacher fires, blames a field and tightens both
layers for it on that route. SR-BH has no response bodies to diff, so blame takes the fallback: the field the
statistical gate scored highest.

The teacher here is the dataset label, standing in for the split verifier's verdict. In deployment the verifier
confirms only a subset of misses, so learning from every miss is the upper bound of what recalibration can do.
Time is the request timestamp, so the half-life decays in data time.

usage: python -I replay_recal.py <data_capec_multilabel.csv> <sse dir> <out.json>
"""
import json
import random
import sys
import time
import urllib.parse
from collections import Counter, defaultdict

import numpy as np

CSV, REAL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [sys.argv[0], CSV, REAL, OUT]
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))  # -I drops the script dir
import run_sse_srbh as base  # noqa: E402  (loader, request builder, split boundary)
from calibration import Calibrator  # noqa: E402

HALFLIVES = (120.0, 600.0, 3600.0)
NORMAL = ["000 - Normal"]


def main():
    t0 = time.perf_counter()
    rows, attack_cols = base.load()
    rnd = random.Random(base.SEED)
    window = [r for r in rows if r[0] < base.ATTACK_END]
    benign = [r for r in window if r[1] == NORMAL]
    rnd.shuffle(benign)
    fit_pool, held = benign[:len(benign) // 2], {id(r) for r in benign[len(benign) // 2:]}
    fit = [base.to_request(r[2], True) for r in rnd.sample(fit_pool, min(base.N_FIT, len(fit_pool)))]
    stream = [r for r in window if r[1] != NORMAL or id(r) in held]          # already in time order
    gate, sem = base.Gate(fpr=0.01), base.Semantic(fpr=0.01)
    gate.fit(fit); sem.fit(fit)
    print(f"fit on {len(fit)} benign; replaying {len(stream)} requests "
          f"({sum(r[1] != NORMAL for r in stream)} attacks) in time order", flush=True)

    names = ["static"] + [f"recal_hl{int(h)}s" for h in HALFLIVES]
    cals = {f"recal_hl{int(h)}s": Calibrator(halflife=h) for h in HALFLIVES}
    tally = {n: Counter() for n in names}
    per_class = {n: defaultdict(Counter) for n in names}
    hourly = {n: defaultdict(Counter) for n in names}
    lat = []
    t_start = stream[0][0]

    def cal_odd(c, rt, zg, zs, pnames, now):                 # per-field override check (as in the calibration loop)
        return (c.odd(rt, "gate", zg, gate.thr, now) or c.odd(rt, "param", {n: 0.0 for n in pnames}, gate.thr, now)
                or c.odd(rt, "sem", zs, sem.thr, now))

    for i, (now, labels, row) in enumerate(stream):
        h, pnames = base.to_request(row, True)
        rt = urllib.parse.urlparse(row["request_http_request"] or "/").path or "/"
        attack = labels != NORMAL
        t = time.perf_counter()
        zs = sem.header_z(h)
        static = gate.odd(h, pnames) or max(list(zs.values()) + [0.0]) > sem.thr
        zg = gate.header_z(h) if not static else None
        lat.append((time.perf_counter() - t) * 1e3)
        hour = int((now - t_start) // 3600)
        for n in names:
            flagged = static or (n != "static" and cal_odd(cals[n], rt, zg, zs, pnames, now))
            k = ("tp" if flagged else "fn") if attack else ("fp" if flagged else "tn")
            tally[n][k] += 1
            hourly[n][hour][k] += 1
            if attack:
                for c in labels:
                    per_class[n][c]["hit" if flagged else "miss"] += 1
            if not attack or flagged or n == "static":
                continue
            # miss -> teacher: blame and tighten (fallback blame: highest statistical-gate field)
            c = cals[n]
            if not zg:
                tally[n]["blame_no_field"] += 1
                continue
            name = max(zg, key=zg.get)
            tally[n]["blames"] += 1
            tally[n]["tightened"] += c.tighten(rt, "gate", name, zg[name], gate.thr, now)
            if name in zs:
                tally[n]["tightened"] += c.tighten(rt, "sem", name, zs[name], sem.thr, now)
            # verify: the request just missed must now be caught
            tally[n]["verified" if cal_odd(c, rt, zg, zs, pnames, now) else "verify_fail"] += 1
        if i % 50000 == 0:
            print(f"  {i}/{len(stream)}  " + "  ".join(
                f"{n}: tpr={tally[n]['tp'] / max(1, tally[n]['tp'] + tally[n]['fn']):.3f} "
                f"fpr={tally[n]['fp'] / max(1, tally[n]['fp'] + tally[n]['tn']):.3f}" for n in names), flush=True)

    out = {"stream_requests": len(stream), "fit_benign": len(fit), "seconds": round(time.perf_counter() - t0, 1),
           "teacher": "dataset label on every miss (upper bound of split-verifier feedback)",
           "blame": "fallback: field with the highest statistical-gate z",
           "gate_latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}}
    for n in names:
        tl = tally[n]
        tp, fn, fp, tn = tl["tp"], tl["fn"], tl["fp"], tl["tn"]
        out[n] = {"tpr": tp / (tp + fn), "fpr": fp / (fp + tn), "precision": tp / max(1, tp + fp),
                  **{k: tl[k] for k in ("blames", "tightened", "verified", "verify_fail", "blame_no_field")},
                  "calibration_entries": len(cals[n].o) if n in cals else 0,
                  "per_class_tpr": {c: v["hit"] / (v["hit"] + v["miss"]) for c, v in sorted(per_class[n].items())},
                  "hourly": {hr: {"tpr": v["tp"] / max(1, v["tp"] + v["fn"]), "fpr": v["fp"] / max(1, v["fp"] + v["tn"]),
                                  "attacks": v["tp"] + v["fn"], "benign": v["fp"] + v["tn"]}
                             for hr, v in sorted(hourly[n].items())}}
        print(n, {k: out[n][k] for k in ("tpr", "fpr", "precision", "blames", "tightened", "verified", "verify_fail",
                                         "blame_no_field", "calibration_entries")}, flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
