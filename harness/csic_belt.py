"""CSIC 2010 conveyor belt: each scanner keeps scanning the same 61,065-request CSIC stream, learning every lap.

  brand_new          fresh SSE hybrid fit on CSIC normal training traffic
  rebaselined_gains  SR-BH lap-3 scanner rebaselined on CSIC (fresh baseline + carried 3-gram gains)
  srbh_as_is         SR-BH lap-3 scanner carried over unchanged
Same stream, order and fit sample as csic_lap.py / csic_rebaseline.py. Lap 1 is the unseen-traffic figure;
later laps show convergence on this belt.

usage: python -I csic_belt.py <csic dir> <sse dir> <srbh scanner .pkl> <out.json>
"""
import copy
import json
import os
import pickle
import random
import sys
import time

CSIC, REAL, PKL, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.argv = [sys.argv[0], CSIC, REAL, PKL, OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir
import csic_rebaseline as cr  # noqa: E402  (first: it reads the 4-argument argv, then imports csic_lap)
import csic_lap as cl  # noqa: E402
import replay_learn as rl  # noqa: E402
import run_sse_srbh as base  # noqa: E402
import __main__  # noqa: E402

LAPS = int(os.environ.get("CSIC_LAPS", 8))


def main():
    train = cl.parse(os.path.join(CSIC, "normalTrafficTraining.txt"))
    normal = cl.parse(os.path.join(CSIC, "normalTrafficTest.txt"))
    attack = cl.parse(os.path.join(CSIC, "anomalousTrafficTest.txt"))
    rnd = random.Random(cl.SEED)
    stream = [(0.0, cl.NORMAL, r) for r in normal] + [(0.0, cl.ATTACK, r) for r in attack]
    rnd.shuffle(stream)
    stream = [(float(i), lab, r) for i, (_, lab, r) in enumerate(stream)]
    fit = [base.to_request(r, True) for r in rnd.sample(train, base.N_FIT)]
    __main__.Detector = rl.Detector
    with open(PKL, "rb") as f:
        old = pickle.load(f)

    def fresh(gains):
        gate, sem = base.Gate(fpr=0.01), base.Semantic(fpr=0.01)
        gate.fit(fit); sem.fit(fit)
        if gains:
            for k in old.sem.space:
                if k in sem.space:
                    sem.space[k].gain = old.sem.space[k].gain.copy()
            cr.reset_thresholds(gate, sem, fit)
        return rl.Detector(gate, sem, True)

    scanners = {"brand_new": fresh(False), "rebaselined_gains": fresh(True), "srbh_as_is": copy.deepcopy(old)}
    out = {"protocol": __doc__.split("usage:")[0].strip(), "stream": len(stream), "laps": LAPS}
    for name, det in scanners.items():
        out[name] = {}
        for lap in range(1, LAPS + 1):
            t = time.perf_counter()
            r = rl.run(det, stream, learn=True, label=f"{name} lap{lap}")
            out[name][str(lap)] = cl.summarise(r)
            print(f"{name:18s} lap {lap}: tpr={r['tpr']:.4f} fpr={r['fpr']:.4f} precision={r['precision']:.4f} "
                  f"wrong={r['wrong']} ({time.perf_counter() - t:.0f}s)", flush=True)
            with open(OUT, "w") as f:
                json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
