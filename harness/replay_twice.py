"""Did the real-time instance adapt? Precalibrate on 17 Jul, replay 18-19 Jul learning on every wrong verdict
(pass 1), then replay the very same 18-19 Jul stream again (pass 2): once frozen, once still learning.
Pass 2 re-sees requests it was already corrected on, so it measures adaptation, not generalisation.
The trained calibration state after pass 1 is saved (pickle) for reuse.

usage: python -I replay_twice.py <data_capec_multilabel.csv> <sse dir> <out.json>
"""
import copy
import json
import os
import pickle
import random
import sys
import time
import urllib.parse
from collections import Counter, defaultdict

CSV, REAL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [sys.argv[0], CSV, REAL, OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir
import replay_realtime as rr  # noqa: E402  (split boundaries)
import run_sse_srbh as base  # noqa: E402
from calibration import Calibrator  # noqa: E402

NORMAL = rr.NORMAL


def main():
    t0 = time.perf_counter()
    rows, _ = base.load()
    rnd = random.Random(base.SEED)
    day1 = [r for r in rows if r[0] < rr.DAY2]
    replay = [r for r in rows if rr.DAY2 <= r[0] < base.ATTACK_END]
    b1 = [r for r in day1 if r[1] == NORMAL]
    rnd.shuffle(b1)
    fit_rows = b1[:base.N_FIT]                                   # same draw as replay_realtime.py
    used = {id(r) for r in fit_rows}
    precal_stream = [r for r in day1 if id(r) not in used]
    gate, sem = base.Gate(fpr=0.01), base.Semantic(fpr=0.01)
    gate.fit([base.to_request(r[2], True) for r in fit_rows])
    sem.fit([base.to_request(r[2], True) for r in fit_rows])

    def prep(stream):                                            # features once; both passes reuse them
        out = []
        for now, labels, row in stream:
            h, pnames = base.to_request(row, True)
            zg = gate.header_z(h)
            zg["_req"] = gate.score(h, pnames)
            rt = urllib.parse.urlparse(row["request_http_request"] or "/").path or "/"
            out.append((now, labels != NORMAL, rt, zg, sem.header_z(h), labels))
        return out

    def run(c, stream, learn, hourly=False):
        tl, pc, hr = Counter(), defaultdict(Counter), defaultdict(Counter)
        t_start = stream[0][0]
        for now, attack, rt, zg, zs, labels in stream:
            og, os_ = c.over(rt, "gate", zg, gate.thr, now), c.over(rt, "sem", zs, sem.thr, now)
            flagged = bool(og or os_)
            k = ("tp" if flagged else "fn") if attack else ("fp" if flagged else "tn")
            tl[k] += 1
            if hourly:
                hr[int((now - t_start) // 3600)][k] += 1
            if attack:
                for cl in labels:
                    pc[cl]["hit" if flagged else "miss"] += 1
            if flagged == attack or not learn:
                continue
            tl["wrong_corrected"] += 1
            if attack:
                fields = {f: v for f, v in zg.items() if f != "_req"}
                name = max(fields, key=fields.get) if fields else "_req"
                c.tighten(rt, "gate", name, zg[name], gate.thr, now)
                if name in zs:
                    c.tighten(rt, "sem", name, zs[name], sem.thr, now)
            else:
                for f in og:
                    c.loosen(rt, "gate", f, zg[f], gate.thr, now)
                for f in os_:
                    c.loosen(rt, "sem", f, zs[f], sem.thr, now)
        res = {"tpr": tl["tp"] / max(1, tl["tp"] + tl["fn"]), "fpr": tl["fp"] / max(1, tl["fp"] + tl["tn"]),
               "precision": tl["tp"] / max(1, tl["tp"] + tl["fp"]), "wrong": tl["fn"] + tl["fp"],
               "corrected": tl["wrong_corrected"], "overrides": len(c.o),
               "per_class_tpr": {k: v["hit"] / (v["hit"] + v["miss"]) for k, v in sorted(pc.items())}}
        if hourly:
            res["hourly"] = {h: {"tpr": v["tp"] / max(1, v["tp"] + v["fn"]), "fpr": v["fp"] / max(1, v["fp"] + v["tn"])}
                             for h, v in sorted(hr.items())}
        return res

    pre = prep(precal_stream)
    cal = Calibrator(halflife=None)
    out = {"protocol": __doc__.split("usage:")[0].strip(), "precal": run(cal, pre, learn=True)}
    print("precal", {k: out["precal"][k] for k in ("tpr", "fpr", "wrong", "overrides")}, flush=True)
    rep = prep(replay)
    print(f"replay features ready: {len(rep)} requests, {sum(a for _, a, *_ in rep)} attacks", flush=True)
    out["pass1_learning"] = run(cal, rep, learn=True, hourly=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(OUT)), "trained_calibrator.pkl"), "wb") as f:
        pickle.dump({"overrides": cal.o, "margin": cal.margin, "gate_thr": gate.thr, "sem_thr": sem.thr}, f)
    out["pass2_frozen"] = run(copy.deepcopy(cal), rep, learn=False, hourly=True)
    out["pass2_learning"] = run(cal, rep, learn=True, hourly=True)
    out["pass3_learning"] = run(cal, rep, learn=True)          # one more lap: does it keep converging?
    out["seconds"] = round(time.perf_counter() - t0, 1)
    for k in ("pass1_learning", "pass2_frozen", "pass2_learning", "pass3_learning"):
        print(k, {m: round(out[k][m], 4) if isinstance(out[k][m], float) else out[k][m]
                  for m in ("tpr", "fpr", "precision", "wrong", "corrected", "overrides")}, flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
