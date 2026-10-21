"""Conveyor belt: the same trained scanner keeps scanning the same looping belt and keeps calibrating.

Loads the scanner saved at the end of lap 1 by replay_learn.py (trained_<detector>.pkl) and runs laps 2, 3, ...
over the identical 18-19 Jul stream (384,629 requests), learning from every wrong verdict on every lap. Stops when
a lap removes less than 1% of the previous lap's mistakes, or after MAX_LAPS. Lap 1 is the only figure for items
never seen before; later laps show how fast and how far the scanner converges on this belt.

usage: python -I belt.py <data_capec_multilabel.csv> <sse dir> <detector> <out.json>
"""
import json
import os
import pickle
import sys
import time

CSV, REAL, NAME, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.argv = [sys.argv[0], CSV, REAL, OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir
import replay_learn as rl  # noqa: E402
import replay_realtime as rr  # noqa: E402
import run_sse_srbh as base  # noqa: E402
import __main__  # noqa: E402

MAX_LAPS = int(os.environ.get("BELT_MAX_LAPS", 6))


def main():
    here = os.path.dirname(os.path.abspath(OUT))
    __main__.Detector = rl.Detector                    # lap 1 pickled the scanner from replay_learn's __main__
    with open(os.path.join(here, f"trained_{NAME}.pkl"), "rb") as f:
        det = pickle.load(f)
    rows, _ = base.load()
    belt = [r for r in rows if rr.DAY2 <= r[0] < base.ATTACK_END]
    lap1 = json.load(open(os.path.join(here, "learn_results.json" if NAME == "gates_learn" else "learn_route_results.json")))[NAME]["pass1"]
    out = {"detector": NAME, "belt_requests": len(belt), "laps": {"1": {k: lap1[k] for k in ("tpr", "fpr", "precision", "wrong")}}}
    print(f"lap 1 (from replay_learn): tpr={lap1['tpr']:.4f} fpr={lap1['fpr']:.4f} wrong={lap1['wrong']}", flush=True)
    prev = lap1["wrong"]
    for lap in range(2, MAX_LAPS + 1):
        t = time.perf_counter()
        r = rl.run(det, belt, learn=True, label=f"lap{lap}")
        out["laps"][str(lap)] = {**{k: r[k] for k in ("tpr", "fpr", "precision", "wrong", "flipped_rate")},
                                 "seconds": round(time.perf_counter() - t, 1), "per_class_tpr": r["per_class_tpr"]}
        print(f"lap {lap}: tpr={r['tpr']:.4f} fpr={r['fpr']:.4f} precision={r['precision']:.4f} wrong={r['wrong']} "
              f"({time.perf_counter() - t:.0f}s)", flush=True)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)
        with open(os.path.join(here, f"belt_{NAME}_lap{lap}.pkl"), "wb") as f:   # snapshot every lap
            pickle.dump(det, f)
        if prev - r["wrong"] < 0.01 * prev:
            out["converged_at_lap"] = lap
            break
        prev = r["wrong"]
    out["state"] = det.state()
    with open(os.path.join(here, f"belt_{NAME}.pkl"), "wb") as f:
        pickle.dump(det, f)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
