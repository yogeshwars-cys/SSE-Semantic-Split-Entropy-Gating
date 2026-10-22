#!/bin/sh
# One tabular baseline (bench/tabular_baselines.py) on every dataset, JOBS at a time (one thread each), then its
# summary table results/bench/<METHOD>.md.
# usage: [JOBS=4] sh bench/run_method.sh <isolation_forest|ocsvm|kitnet|random_forest|xgboost|catboost>
cd "$(dirname "$0")/.."
M=$1
S=data/sse_streams
{ echo srbh; echo csic; for f in $S/cic_*.npz $S/unsw.npz $S/kitsune_*.npz; do basename "$f" .npz; done; } |
  BENCH_THREADS=1 xargs -P "${JOBS:-4}" -I{} sh -c "python3 -I bench/tabular_baselines.py {} $M > results/bench/logs/$M.{}.log 2>&1 || echo FAILED {}"
python3 -I bench/method_summary.py $M
