"""One baseline across every dataset: results/bench/<METHOD>.md (from results/bench/<dataset>/<method>.json).

usage: python -I bench/method_summary.py <isolation_forest|ocsvm|kitnet|random_forest|xgboost|catboost>
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "bench")
M = sys.argv[1]
order = ["srbh", "csic"] + sorted(os.path.basename(p) for p in glob.glob(f"{RES}/cic_*")) + ["unsw"] + \
    sorted(os.path.basename(p) for p in glob.glob(f"{RES}/kitsune_*"))
lines = [f"# {M}", "", "Same test rows as SSE (results/bench/SSE_5LAPS*.md); protocol in bench/tabular_baselines.py.", "",
         "| Dataset | detection | false alarms | precision | F1 | wrong | ROC-AUC | det. @1% FA | model size | us/record (batch / single) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for d in order:
    p = f"{RES}/{d}/{M}.json"
    r = json.load(open(p)).get(M) if os.path.exists(p) else None
    if not r:
        lines.append(f"| {d} | not run | | | | | | | | |")
        continue
    lines.append(f"| {d} | {r['tpr']:.1%} | {r['fpr']:.1%} | {r['precision']:.1%} | {r['f1']:.3f} | {r['wrong']:,} | "
                 f"{r['auc']:.3f} | {r['tpr@fpr1%']:.1%} | {r['model_bytes'] / 1e6:.2f} MB | "
                 f"{r['us_per_record_batch']:.1f} / {r['us_per_record_single']:.0f} |")
out = os.path.join(RES, f"{M.upper()}.md")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
