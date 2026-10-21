"""One lap of CSIC 2010 (a different web app: an e-shop, not WordPress) through two SSE hybrids.

  new_hybrid    brand-new SSE: gates fit on CSIC normal training traffic (10,000 requests, benign only)
  srbh_hybrid   the scanner trained on SR-BH (precalibration + belt laps), carried over unchanged
Both are hybrids (gates learn from every wrong verdict + real-time route thresholds) and keep learning
during the lap. A frozen pass of each (no learning) is reported for reference.
Requests are mapped onto the SR-BH field schema (the same 11 header columns + @method/@path/@query/@body) so the
carried-over scanner sees the fields it was built on. CSIC has no timestamps: normal test and attack requests are
interleaved in a seeded random order. Teacher = dataset label.

usage: python -I csic_lap.py <csic dir> <sse dir> <srbh scanner .pkl> <out.json>
"""
import copy
import json
import os
import pickle
import random
import re
import sys
import time
import urllib.parse

CSIC, REAL, PKL, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.argv = [sys.argv[0], "unused.csv", REAL, OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir
import replay_learn as rl  # noqa: E402
import run_sse_srbh as base  # noqa: E402
import __main__  # noqa: E402

NORMAL = ["000 - Normal"]
ATTACK = ["csic_attack"]
SEED = 11
COLS = {v: k for k, v in base.HEADER_COLS.items()}          # header name -> SR-BH column


def parse(path):
    text = open(path, encoding="latin-1").read()
    out = []
    for blk in re.split(r"\n(?=(?:GET|POST|PUT) http)", text):
        lines = blk.strip("\n").split("\n")
        if not lines or not re.match(r"(GET|POST|PUT) http", lines[0]):
            continue
        method, url = lines[0].split(" ")[:2]
        u = urllib.parse.urlparse(url)
        row = {"request_http_method": method, "request_http_request": u.path + ("?" + u.query if u.query else ""),
               "request_body": "", "src_ip": "", **{c: "" for c in base.HEADER_COLS}}
        i = 1
        while i < len(lines) and lines[i].strip():
            k, _, v = lines[i].partition(":")
            col = COLS.get(k.strip().lower())
            if col:
                row[col] = v.strip()
            i += 1
        row["request_body"] = "\n".join(l for l in lines[i:] if l.strip()).strip()
        out.append(row)
    return out


def summarise(r):
    return {k: r[k] for k in ("tpr", "fpr", "precision", "wrong", "corrected")}


def main():
    t0 = time.perf_counter()
    train = parse(os.path.join(CSIC, "normalTrafficTraining.txt"))
    normal = parse(os.path.join(CSIC, "normalTrafficTest.txt"))
    attack = parse(os.path.join(CSIC, "anomalousTrafficTest.txt"))
    rnd = random.Random(SEED)
    stream = [(0.0, NORMAL, r) for r in normal] + [(0.0, ATTACK, r) for r in attack]
    rnd.shuffle(stream)
    stream = [(float(i), lab, r) for i, (_, lab, r) in enumerate(stream)]   # one request per "second"; no decay used
    fit = [base.to_request(r, True) for r in rnd.sample(train, base.N_FIT)]
    gate, sem = base.Gate(fpr=0.01), base.Semantic(fpr=0.01)
    gate.fit(fit); sem.fit(fit)
    __main__.Detector = rl.Detector
    only = os.environ.get("CSIC_ONLY")                          # "new" or "srbh": run one scanner per process
    srbh = None
    if only != "new":
        with open(PKL, "rb") as f:
            srbh = pickle.load(f)
    print(f"CSIC: {len(train)} train, {len(normal)} normal test, {len(attack)} attacks; stream {len(stream)}; "
          f"scanner {os.path.basename(PKL)}", flush=True)

    out = {"protocol": __doc__.split("usage:")[0].strip(), "scanner": os.path.basename(PKL),
           "stream": len(stream), "normal": len(normal), "attacks": len(attack)}
    runs = (("new_hybrid frozen", lambda: rl.Detector(gate, sem, True), False),
            ("srbh_hybrid frozen", lambda: copy.deepcopy(srbh), False),
            ("new_hybrid", lambda: rl.Detector(gate, sem, True), True),
            ("srbh_hybrid", lambda: copy.deepcopy(srbh), True))
    for name, make, learn in runs:
        if only and not name.startswith(only):
            continue
        t = time.perf_counter()
        r = rl.run(make(), stream, learn=learn, label=name)
        out[name] = {**summarise(r), "seconds": round(time.perf_counter() - t, 1),
                     "curve_per_3600_requests": r["hourly"]}            # "now" = request index, so 1 bucket = 3,600
        print(f"{name:20s} tpr={r['tpr']:.4f} fpr={r['fpr']:.4f} precision={r['precision']:.4f} wrong={r['wrong']} "
              f"({time.perf_counter() - t:.0f}s)", flush=True)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
