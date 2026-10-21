"""Tell the SR-BH lap-3 scanner it is in a new environment: rebaseline on CSIC, keep the attack knowledge.

rebaseline = fit fresh gates on the new site's normal traffic (same 10,000 CSIC requests as the brand-new
scanner), then copy over what is not site-specific:
  gains    the semantic layer's learned per-3-gram weights, per field (same hashing, so buckets line up)
  weights  the statistical gate's learned per-field sensitivities
and drop what is: SR-BH baseline statistics, stored SR-BH values, IDF, route overrides. Thresholds are then
re-set on the held-out half of the fit sample (1% false alarms), since the copied knowledge changes how normal
traffic scores. Same stream, order and fit sample as csic_lap.py, so rows compare directly with the brand-new
scanner. One lap each, frozen and learning (hybrid: gates learn + real-time route thresholds).

usage: python -I csic_rebaseline.py <csic dir> <sse dir> <srbh scanner .pkl> <out.json>
"""
import json
import os
import pickle
import random
import statistics
import sys
import time

CSIC, REAL, PKL, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.argv = [sys.argv[0], CSIC, REAL, PKL, OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir
import csic_lap as cl  # noqa: E402
import replay_learn as rl  # noqa: E402
import run_sse_srbh as base  # noqa: E402
import __main__  # noqa: E402


def reset_thresholds(gate, sem, fit):
    """Redo the fit-time normalisation and 1% thresholds with the transferred knowledge in place."""
    half = len(fit) // 2
    raw = [gate.base.raw(h, p, gate.w) for h, p in fit[:half]]
    gate.mu = [statistics.mean(c) for c in zip(*raw)]
    gate.sd = [statistics.pstdev(c) or 1 for c in zip(*raw)]
    ns = sorted(gate.score(h, p) for h, p in fit[half:])
    gate.thr = ns[max(int(len(ns) * (1 - gate.fpr)) - 1, 0)]
    for sp in sem.space.values():
        sp.ver += 1
        sp._dc.clear()
    d = {k: [] for k in sem.space}
    for h, _ in fit[half:]:
        for k, v in h.items():
            if k in sem.space:
                d[k].append(sem.space[k].dist(v))
    sem.stat = {k: (statistics.mean(x), max(statistics.pstdev(x), 0.02)) for k, x in d.items() if x}
    ns = sorted(sem.score(h, p) for h, p in fit[half:])
    sem.thr = ns[max(int(len(ns) * (1 - sem.fpr)) - 1, 0)]


def main():
    t0 = time.perf_counter()
    train = cl.parse(os.path.join(CSIC, "normalTrafficTraining.txt"))
    normal = cl.parse(os.path.join(CSIC, "normalTrafficTest.txt"))
    attack = cl.parse(os.path.join(CSIC, "anomalousTrafficTest.txt"))
    rnd = random.Random(cl.SEED)                                    # identical draw to csic_lap.py
    stream = [(0.0, cl.NORMAL, r) for r in normal] + [(0.0, cl.ATTACK, r) for r in attack]
    rnd.shuffle(stream)
    stream = [(float(i), lab, r) for i, (_, lab, r) in enumerate(stream)]
    fit = [base.to_request(r, True) for r in rnd.sample(train, base.N_FIT)]
    __main__.Detector = rl.Detector
    with open(PKL, "rb") as f:
        old = pickle.load(f)
    shared = sorted(set(old.sem.space))
    out = {"protocol": __doc__.split("usage:")[0].strip(), "scanner": os.path.basename(PKL),
           "transferred_gate_weights": {k: round(v, 3) for k, v in old.gate.w.items()},
           "transferred_boosted_grams": {k: int((sp.gain > 1).sum()) for k, sp in old.sem.space.items()}}

    def build(gains, weights):
        gate, sem = base.Gate(fpr=0.01), base.Semantic(fpr=0.01)
        gate.fit(fit); sem.fit(fit)
        if weights:
            gate.w = dict(old.gate.w)
        if gains:
            for k in shared:
                if k in sem.space:
                    sem.space[k].gain = old.sem.space[k].gain.copy()
        if gains or weights:
            reset_thresholds(gate, sem, fit)
        return gate, sem

    variants = (("brand_new", False, False), ("rebaselined_gains+weights", True, True),
                ("rebaselined_gains_only", True, False), ("rebaselined_weights_only", False, True))
    for name, gains, weights in variants:
        gate, sem = build(gains, weights)
        out[name] = {"gate_thr": gate.thr, "sem_thr": sem.thr}
        for mode, learn in (("frozen", False), ("lap", True)):
            t = time.perf_counter()
            r = rl.run(rl.Detector(gate, sem, True), stream, learn=learn, label=f"{name} {mode}")
            out[name][mode] = {**cl.summarise(r), "curve_per_3600_requests": r["hourly"]}
            print(f"{name:28s} {mode:6s} tpr={r['tpr']:.4f} fpr={r['fpr']:.4f} precision={r['precision']:.4f} "
                  f"wrong={r['wrong']} ({time.perf_counter() - t:.0f}s)", flush=True)
            with open(OUT, "w") as f:
                json.dump(out, f, indent=1)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
