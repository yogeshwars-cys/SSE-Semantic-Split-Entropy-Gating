"""SSE with real-time recalibration on every wrong verdict, precalibrated on the previous day.

Every request is judged per field: the statistical gate's per-field z (plus "_req", the gate's whole-request score)
and the semantic layer's per-field z, each against its own (route, layer, field) threshold, the global 1% threshold
where nothing has been learned. After each verdict the teacher corrects it at once:
  missed attack -> tighten the blamed field (fallback blame: highest statistical z) on that route, both layers,
                   to just below the attack's score;
  false alarm   -> loosen every field that fired on that route to just above the benign score.
No half-life: a correction stands until a later wrong verdict moves it. Every correction is re-checked on the
same request (verified = that request now gets the right verdict).
"Training" is precalibration: the same loop run over 17 Jul traffic, carried into the 18-19 Jul replay.
Teacher = dataset label (in deployment, the split verifier supplies these verdicts).

usage: python -I replay_realtime.py <data_capec_multilabel.csv> <sse dir> <out.json>
"""
import copy
import json
import os
import random
import sys
import time
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

CSV, REAL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [sys.argv[0], CSV, REAL, OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir
import run_sse_srbh as base  # noqa: E402
from calibration import Calibrator  # noqa: E402

NORMAL = ["000 - Normal"]
DAY2 = datetime.strptime("18/Jul/2020:00:00:00 +0100", "%d/%b/%Y:%H:%M:%S %z").timestamp()


def main():
    t0 = time.perf_counter()
    rows, _ = base.load()
    rnd = random.Random(base.SEED)
    day1 = [r for r in rows if r[0] < DAY2]
    replay = [r for r in rows if DAY2 <= r[0] < base.ATTACK_END]
    b1 = [r for r in day1 if r[1] == NORMAL]
    rnd.shuffle(b1)
    fit_rows = b1[:base.N_FIT]
    used = {id(r) for r in fit_rows}
    precal_stream = [r for r in day1 if id(r) not in used]        # time order, gate-fit requests left out
    gate, sem = base.Gate(fpr=0.01), base.Semantic(fpr=0.01)
    gate.fit([base.to_request(r[2], True) for r in fit_rows])
    sem.fit([base.to_request(r[2], True) for r in fit_rows])

    def features(row):
        h, pnames = base.to_request(row, True)
        zg = gate.header_z(h)
        zg["_req"] = gate.score(h, pnames)
        return zg, sem.header_z(h)

    def verdict(c, rt, zg, zs, now):
        return c.over(rt, "gate", zg, gate.thr, now), c.over(rt, "sem", zs, sem.thr, now)

    def correct(c, tl, rt, zg, zs, attack, og, os_, now):
        """Real-time recalibration after a wrong verdict, then re-check the same request."""
        if attack:                                               # miss
            fields = {k: v for k, v in zg.items() if k != "_req"}
            name = max(fields, key=fields.get) if fields else "_req"
            tl["tightened"] += c.tighten(rt, "gate", name, zg[name], gate.thr, now)
            if name in zs:
                tl["tightened"] += c.tighten(rt, "sem", name, zs[name], sem.thr, now)
        else:                                                    # false alarm
            for k in og:
                tl["loosened"] += c.loosen(rt, "gate", k, zg[k], gate.thr, now)
            for k in os_:
                tl["loosened"] += c.loosen(rt, "sem", k, zs[k], sem.thr, now)
        g2, s2 = verdict(c, rt, zg, zs, now)
        tl["verified" if bool(g2 or s2) == attack else "verify_fail"] += 1

    # precalibration: the same loop over 17 Jul
    tp = time.perf_counter()
    pre_cal, pre_tl, pre_routes = Calibrator(halflife=None), Counter(), set()
    for now, labels, row in precal_stream:
        rt = urllib.parse.urlparse(row["request_http_request"] or "/").path or "/"
        attack = labels != NORMAL
        zg, zs = features(row)
        og, os_ = verdict(pre_cal, rt, zg, zs, now)
        if bool(og or os_) != attack:
            pre_tl["wrong"] += 1
            correct(pre_cal, pre_tl, rt, zg, zs, attack, og, os_, now)
        pre_tl["tp" if attack and (og or os_) else "fn" if attack else "fp" if (og or os_) else "tn"] += 1
        if attack:
            pre_routes.add(rt)
    pre_s = time.perf_counter() - tp
    print(f"precalibrated on {len(precal_stream)} requests of 17 Jul in {pre_s:.0f}s: {dict(pre_tl)}, "
          f"{len(pre_cal.o)} overrides", flush=True)

    cals = {"realtime_cold": Calibrator(halflife=None), "precal_frozen": copy.deepcopy(pre_cal),
            "precal_realtime": copy.deepcopy(pre_cal), "tighten_only_hl120": Calibrator(halflife=120.0)}
    learns = {"realtime_cold", "precal_realtime"}
    names = ["static", *cals]
    tally = {n: Counter() for n in names}
    per_class = {n: defaultdict(Counter) for n in names}
    by_route = {n: Counter() for n in names}
    hourly = {n: defaultdict(Counter) for n in names}
    lat, t_start = [], replay[0][0]
    print(f"replay {len(replay)} requests of 18-19 Jul ({sum(r[1] != NORMAL for r in replay)} attacks)", flush=True)

    for i, (now, labels, row) in enumerate(replay):
        rt = urllib.parse.urlparse(row["request_http_request"] or "/").path or "/"
        attack = labels != NORMAL
        t = time.perf_counter()
        zg, zs = features(row)
        lat.append((time.perf_counter() - t) * 1e3)
        static = zg["_req"] > gate.thr or max(list(zs.values()) + [0.0]) > sem.thr
        hour = int((now - t_start) // 3600)
        for n in names:
            c = cals.get(n)
            if n == "static":
                flagged = static
            elif n == "tighten_only_hl120":                     # previous design: decaying, tighten on misses only
                flagged = static or c.odd(rt, "gate", zg, gate.thr, now) or c.odd(rt, "sem", zs, sem.thr, now)
            else:
                og, os_ = verdict(c, rt, zg, zs, now)
                flagged = bool(og or os_)
            k = ("tp" if flagged else "fn") if attack else ("fp" if flagged else "tn")
            tally[n][k] += 1
            hourly[n][hour][k] += 1
            if attack:
                by_route[n][("route_seen_day1" if rt in pre_routes else "route_new", flagged)] += 1
                for cl in labels:
                    per_class[n][cl]["hit" if flagged else "miss"] += 1
            if flagged == attack:
                continue
            tally[n]["wrong"] += 1
            if n in learns:
                correct(c, tally[n], rt, zg, zs, attack, og, os_, now)
            elif n == "tighten_only_hl120" and attack:
                fields = {f: v for f, v in zg.items() if f != "_req"}
                name = max(fields, key=fields.get)
                c.tighten(rt, "gate", name, zg[name], gate.thr, now)
                if name in zs:
                    c.tighten(rt, "sem", name, zs[name], sem.thr, now)
        if i % 50000 == 0:
            print(f"  {i}/{len(replay)}  " + "  ".join(
                f"{n}: tpr={tally[n]['tp'] / max(1, tally[n]['tp'] + tally[n]['fn']):.3f} "
                f"fpr={tally[n]['fp'] / max(1, tally[n]['fp'] + tally[n]['tn']):.3f}" for n in names), flush=True)

    out = {"protocol": __doc__.split("usage:")[0].strip(), "precal_requests": len(precal_stream),
           "precal_seconds": round(pre_s, 1), "precal": dict(pre_tl), "precal_overrides": len(pre_cal.o),
           "replay_requests": len(replay), "seconds": round(time.perf_counter() - t0, 1),
           "feature_latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}}
    for n in names:
        tl, br = tally[n], by_route[n]
        out[n] = {"tpr": tl["tp"] / (tl["tp"] + tl["fn"]), "fpr": tl["fp"] / (tl["fp"] + tl["tn"]),
                  "precision": tl["tp"] / max(1, tl["tp"] + tl["fp"]),
                  **{k: tl[k] for k in ("wrong", "tightened", "loosened", "verified", "verify_fail")},
                  "overrides": len(cals[n].o) if n in cals else 0,
                  "tpr_route_seen_day1": br[("route_seen_day1", True)] / max(1, br[("route_seen_day1", True)] + br[("route_seen_day1", False)]),
                  "tpr_route_new": br[("route_new", True)] / max(1, br[("route_new", True)] + br[("route_new", False)]),
                  "attacks_route_seen_day1": br[("route_seen_day1", True)] + br[("route_seen_day1", False)],
                  "per_class_tpr": {c: v["hit"] / (v["hit"] + v["miss"]) for c, v in sorted(per_class[n].items())},
                  "hourly": {hr: {"tpr": v["tp"] / max(1, v["tp"] + v["fn"]), "fpr": v["fp"] / max(1, v["fp"] + v["tn"]),
                                  "attacks": v["tp"] + v["fn"], "benign": v["fp"] + v["tn"]}
                             for hr, v in sorted(hourly[n].items())}}
        print(n, {k: out[n][k] for k in ("tpr", "fpr", "precision", "wrong", "tightened", "loosened", "verified",
                                         "verify_fail", "overrides", "tpr_route_seen_day1", "tpr_route_new")}, flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
