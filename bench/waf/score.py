"""Score the WAF runs (bench/waf/run_all.sh) on the HTTP test streams: results/bench/<dataset>/<engine>.json.

Per paranoia level: verdicts at the CRS default blocking threshold (inbound anomaly score >= 5), ranking metrics
from the score itself, ms/request and peak memory from the run log. "best" = the paranoia level with the highest F1,
picked on the test set itself (no other data exists for a rule set; this favours the baseline).

usage: python -I bench/waf/score.py <modsec|coraza> [dataset ...]
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402
import http_data  # noqa: E402

ENGINE = sys.argv[1]
DATASETS = sys.argv[2:] or ["srbh", "csic"]
LABEL = {"modsec": "ModSecurity 3.0.12 + OWASP CRS 3.3.5", "coraza": "Coraza 3.8.1 + OWASP CRS 4.25"}[ENGINE]

for name in DATASETS:
    d = http_data.load(name)
    y = np.array([a for _, a, _ in d["test"]])
    cls = [c for _, _, c in d["test"]]
    out = {"engine": LABEL, "dataset": name, "threshold": "inbound anomaly score >= 5 (CRS default)"}
    for pl in (1, 2, 3, 4):
        base = os.path.join(common.RESULTS, "waf", f"{ENGINE}_{name}_pl{pl}")
        if not os.path.exists(base + ".done"):
            print(f"{name} PL{pl}: not finished")
            continue
        s = np.array([int(x.split()[0]) for x in open(base + ".txt")], np.float64)
        log = open(base + ".log").read()
        ms = re.search(r"\(([\d.]+) ms/request\)", log)
        rss = re.search(r"peak RSS (\d+) kB", log)
        out[f"pl{pl}"] = common.report(y, s, s >= 5, cls, ms_per_request=float(ms.group(1)) if ms else None,
                                       peak_rss_mb=int(rss.group(1)) / 1024 if rss else None)
        print(common.line(f"{name} {ENGINE} PL{pl}", out[f"pl{pl}"]), flush=True)
    done = [k for k in out if k.startswith("pl")]
    if done:
        out["best"] = max(done, key=lambda k: out[k]["f1"])
        print(f"  best: {out['best']}")
    common.save(f"{name}/{ENGINE}.json", out)
