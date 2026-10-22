"""Summary of the C++ SSE 5-lap runs (bench/run_sse.sh): results/bench/SSE_5LAPS.md

usage: python -I bench/sse_summary.py [variant]   (variant "exact": sse_cpp_exact.json, compared with the base run)
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "bench")
VAR = sys.argv[1] if len(sys.argv) > 1 else ""
SUF = f"_{VAR}" if VAR else ""


def f1(l):
    p = l["tp"] / max(1, l["tp"] + l["fp"])
    r = l["tp"] / max(1, l["tp"] + l["fn"])
    return 2 * p * r / (p + r) if p + r else 0.0


order = ["srbh", "csic"] + sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{RES}/cic_*/sse_cpp{SUF}.json")) + \
    ["unsw"] + sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{RES}/kitsune_*/sse_cpp{SUF}.json"))
title = "C++ SSE, fully binary semantic store + int8 rescore, 5 belt laps" + \
    (", with the exact-route threshold layer" if VAR == "exact" else "")
lines = [f"# {title}", "",
         "Lap 1 is traffic never seen; laps 2-5 replay the same test stream, learning every lap (convergence, not",
         "generalisation). Teacher = dataset label. Cells: detection / false alarms (F1), wrong verdicts below.", "",
         "| Dataset | test records (attacks) | lap 1 | lap 2 | lap 3 | lap 4 | lap 5 | us/record (frozen) | store after lap 1 |",
         "|---|---|---|---|---|---|---|---|---|"]
for d in order:
    p = f"{RES}/{d}/sse_cpp{SUF}.json"
    if not os.path.exists(p):
        lines.append(f"| {d} | not run | | | | | | | |")
        continue
    j = json.load(open(p))
    laps = [j["lap1"]] + [j["belt"][str(k)] for k in range(2, 6) if str(k) in j.get("belt", {})]
    l1 = j["lap1"]
    cells = [f"{l['tpr']:.1%} / {l['fpr']:.1%} ({f1(l):.3f})<br>{l['wrong']:,} wrong" for l in laps]
    cells += [""] * (5 - len(cells))
    n, a = l1["tp"] + l1["fn"] + l1["fp"] + l1["tn"], l1["tp"] + l1["fn"]
    store = (j["semantic_bytes_after_lap1"] + j["route_state"]["bytes"]) / 1e6
    lines.append(f"| {d} | {n:,} ({a:,}) | " + " | ".join(cells) +
                 f" | {j['frozen_after_lap1']['us_per_request_mean']:.0f} | {store:.1f} MB |")
if VAR:                                                          # wrong verdicts against the base (token-only) run
    lines += ["", "Wrong verdicts, base (token thresholds only) -> this variant:", "",
              "| Dataset | lap 1 | lap 5 |", "|---|---|---|"]
    for d in order:
        a, b = f"{RES}/{d}/sse_cpp.json", f"{RES}/{d}/sse_cpp{SUF}.json"
        if os.path.exists(a) and os.path.exists(b):
            ja, jb = json.load(open(a)), json.load(open(b))
            w = lambda j, k: j["lap1"]["wrong"] if k == 1 else j["belt"]["5"]["wrong"]  # noqa: E731
            lines.append(f"| {d} | {w(ja, 1):,} -> {w(jb, 1):,} | {w(ja, 5):,} -> {w(jb, 5):,} |")
out = os.path.join(RES, f"SSE_5LAPS{SUF}.md")
os.makedirs(RES, exist_ok=True)
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
