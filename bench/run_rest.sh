#!/bin/sh
# Every remaining baseline: OC-SVM, KitNET, Random Forest, XGBoost, CatBoost on all datasets (2 jobs, 1 thread each)
# and char-CNN / char-BiLSTM on SR-BH and CSIC (in parallel), then the per-method tables and the master table.
# usage: sh bench/run_rest.sh   (from the repo root)
cd "$(dirname "$0")/.."
mkdir -p results/bench/logs
for d in srbh csic; do                                           # HTTP feature caches first (shared by every method)
  python3 -I -c "import sys; sys.argv = ['x', '$d']; sys.path.insert(0, 'bench'); import tabular_baselines as t; t.load()"
done
for d in srbh csic; do BENCH_THREADS=1 python3 -I bench/char_nn.py $d > results/bench/logs/charnn.$d.log 2>&1 & done
S=data/sse_streams
for m in ocsvm kitnet random_forest xgboost catboost; do
  { echo srbh; echo csic; for f in $S/cic_*.npz $S/unsw.npz $S/kitsune_*.npz; do basename "$f" .npz; done; } | sed "s/$/ $m/"
done | BENCH_THREADS=1 xargs -P 2 -L 1 sh -c 'python3 -I bench/tabular_baselines.py $0 $1 > results/bench/logs/$1.$0.log 2>&1 || echo "FAILED $1 $0"'
wait
for m in ocsvm kitnet random_forest xgboost catboost; do python3 -I bench/method_summary.py $m > /dev/null; done
python3 -I bench/master_table.py > /dev/null
echo "rest done"
