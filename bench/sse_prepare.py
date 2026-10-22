"""Write an HTTP benchmark stream as the CSV that `cpp/replay_srbh --prepared` reads: the SR-BH columns plus
split (fit / precal / test), label (0 / 1) and class, in stream order.

  csic  fit = the 10,000 normal training requests csic_lap.py fits on; no precalibration; test = the 61,065
        normal + anomalous test requests in csic_lap.py's seeded order

usage: python -I bench/sse_prepare.py csic <out.csv>
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import http_data  # noqa: E402

name, out = sys.argv[1], sys.argv[2]
d = http_data.load(name)
cols = ["request_http_method", "request_http_request", "request_body", *http_data.HEADER_COLS]
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["split", "label", "class", *cols])
    for r in d["fit_normal"]:
        w.writerow(["fit", 0, "", *[r.get(c) or "" for c in cols]])
    for r, attack, cls in d["test"]:
        w.writerow(["test", int(attack), ";".join(cls), *[r.get(c) or "" for c in cols]])
print(len(d["fit_normal"]), "fit +", len(d["test"]), "test ->", out)
