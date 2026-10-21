"""SSE whose gates recalibrate themselves from every failure, precalibrated on 17 Jul, replayed on 18-19 Jul.

Each wrong verdict updates the gates' own knowledge, on every route:
  missed attack -> statistical gate: the blamed field (highest gate z) gets more sensitive;
                   semantic layer:   the oddest field's grams that are rare in normal traffic weigh more;
  false alarm   -> statistical gate: the request joins the baseline, fields that fired get less sensitive;
                   semantic layer:   every field that fired stores the value as normal and drops its gram boosts.
Detectors (same split and seed as replay_realtime.py, so static / thresholds-only numbers there compare directly):
  gates_learn        gates learn, global thresholds unchanged
  gates_learn_route  gates learn + the real-time (route, field) threshold overrides on top
Every 100th correction is re-checked on the same request ("flipped" = that request now gets the right verdict).
Teacher = dataset label (stands in for the split verifier's verdict).

usage: python -I replay_learn.py <data_capec_multilabel.csv> <sse dir> <out.json>
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
import replay_realtime as rr  # noqa: E402
import run_sse_srbh as base  # noqa: E402
from calibration import Calibrator  # noqa: E402

NORMAL = rr.NORMAL


class Detector:
    def __init__(self, gate, sem, route_overrides):
        self.gate, self.sem = copy.deepcopy(gate), copy.deepcopy(sem)
        self.cal = Calibrator(halflife=None)
        self.route = route_overrides

    def features(self, row):
        h, pnames = base.to_request(row, True)
        zg = self.gate.header_z(h)
        zg["_req"] = self.gate.score(h, pnames)
        return h, pnames, zg, self.sem.header_z(h)

    def verdict(self, rt, zg, zs, now):
        return self.cal.over(rt, "gate", zg, self.gate.thr, now), self.cal.over(rt, "sem", zs, self.sem.thr, now)

    def step(self, now, attack, rt, row, tl, learn=True, check=False):
        h, pnames, zg, zs = self.features(row)
        og, os_ = self.verdict(rt, zg, zs, now)
        flagged = bool(og or os_)
        if flagged == attack or not learn:
            return flagged
        tl["corrected"] += 1
        g, s = self.gate, self.sem
        if attack:
            fields = {k: v for k, v in zg.items() if k != "_req"}
            blamed = max(fields, key=fields.get) if fields else None
            g.learn(h, pnames, True, blamed=blamed)
            s.learn(h, True, zs=zs)
            if self.route and blamed is not None:
                self.cal.tighten(rt, "gate", blamed, zg[blamed], g.thr, now)
                if blamed in zs:
                    self.cal.tighten(rt, "sem", blamed, zs[blamed], s.thr, now)
        else:
            g.learn(h, pnames, False, fired=[k for k in og if k != "_req"])
            s.learn(h, False, fired=os_)
            if self.route:
                for k in og:
                    self.cal.loosen(rt, "gate", k, zg[k], g.thr, now)
                for k in os_:
                    self.cal.loosen(rt, "sem", k, zs[k], s.thr, now)
        if check:
            _, _, zg2, zs2 = self.features(row)
            og2, os2 = self.verdict(rt, zg2, zs2, now)
            tl["checked"] += 1
            tl["flipped"] += bool(og2 or os2) == attack
        return flagged

    def state(self):
        g, s = self.gate, self.sem
        return {"gate_field_weights": dict(g.w), "baseline_N": g.base.N,
                "semantic_boosted_grams": {k: int((sp.gain > 1).sum()) for k, sp in s.space.items()},
                "semantic_learned_normal_values": {k: sum(b[4] for b in sp.blocks) + len(sp.pending)
                                                   for k, sp in s.space.items()},
                "route_overrides": len(self.cal.o)}


def run(det, stream, learn=True, label=""):
    tl, pc, hr = Counter(), defaultdict(Counter), defaultdict(Counter)
    t_start, t0 = stream[0][0], time.perf_counter()
    for i, (now, labels, row) in enumerate(stream):
        rt = urllib.parse.urlparse(row["request_http_request"] or "/").path or "/"
        attack = labels != NORMAL
        flagged = det.step(now, attack, rt, row, tl, learn=learn, check=tl["corrected"] % 100 == 0)
        k = ("tp" if flagged else "fn") if attack else ("fp" if flagged else "tn")
        tl[k] += 1
        hr[int((now - t_start) // 3600)][k] += 1
        if attack:
            for c in labels:
                pc[c]["hit" if flagged else "miss"] += 1
        if i % 50000 == 0 and i:
            print(f"    {label} {i}/{len(stream)} tpr={tl['tp'] / max(1, tl['tp'] + tl['fn']):.3f} "
                  f"fpr={tl['fp'] / max(1, tl['fp'] + tl['tn']):.3f} {time.perf_counter() - t0:.0f}s", flush=True)
    return {"tpr": tl["tp"] / max(1, tl["tp"] + tl["fn"]), "fpr": tl["fp"] / max(1, tl["fp"] + tl["tn"]),
            "precision": tl["tp"] / max(1, tl["tp"] + tl["fp"]), "wrong": tl["fn"] + tl["fp"],
            "corrected": tl["corrected"], "checked": tl["checked"],
            "flipped_rate": tl["flipped"] / max(1, tl["checked"]),
            "per_class_tpr": {c: v["hit"] / (v["hit"] + v["miss"]) for c, v in sorted(pc.items())},
            "hourly": {h: {"tpr": v["tp"] / max(1, v["tp"] + v["fn"]), "fpr": v["fp"] / max(1, v["fp"] + v["tn"])}
                       for h, v in sorted(hr.items())}}


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
    out = {"protocol": __doc__.split("usage:")[0].strip(),
           "reference_same_split": "replay_realtime.py: static tpr 0.559 fpr 0.594; thresholds-only real-time "
                                   "(precalibrated) tpr 0.657 fpr 0.390"}
    here = os.path.dirname(os.path.abspath(OUT))
    only = os.environ.get("SSE_DETECTOR")                       # run one detector per process if set
    for name, route in (("gates_learn", False), ("gates_learn_route", True)):
        if only and name != only:
            continue
        det = Detector(gate, sem, route)
        print(f"== {name}: precalibrating on {len(precal_stream)} requests of 17 Jul", flush=True)
        pre = run(det, precal_stream, label="precal")
        print(f"   precal tpr={pre['tpr']:.3f} fpr={pre['fpr']:.3f} wrong={pre['wrong']}", flush=True)
        p1 = run(det, replay, label="pass1")
        out[name] = {"precal": {k: pre[k] for k in ("tpr", "fpr", "wrong", "flipped_rate")}, "pass1": p1,
                     "state_after_pass1": det.state()}
        with open(os.path.join(here, f"trained_{name}.pkl"), "wb") as f:
            pickle.dump(det, f)
        print(f"   pass1 {name}: " + str({k: round(p1[k], 4) if isinstance(p1[k], float) else p1[k]
                                          for k in ("tpr", "fpr", "precision", "wrong", "flipped_rate")}), flush=True)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
