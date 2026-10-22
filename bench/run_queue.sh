#!/bin/sh
# The current benchmark queue in one process (finished WAF runs are skipped):
#   Python exact-route check on SR-BH (in parallel)  ->  Coraza + CRS 4.25  ->  Isolation Forest on all datasets
# usage: sh bench/run_queue.sh   (from the repo root)
cd "$(dirname "$0")/.."
T=data/traces
( HYB_QUANT=3 HYB_EXACT=1 HYB_TRACE=$T/py_srbh_exact.txt python3 -I harness/replay_hybrid2.py \
    data/sr-bh-2020/data_capec_multilabel.csv sse results/srbh/hybrid2/full_binary_int8_exact.json \
    > results/srbh/hybrid2/full_binary_int8_exact.log 2>&1
  head -c "$(wc -c < $T/py_srbh_exact.txt)" $T/srbh_exact.txt | cmp - $T/py_srbh_exact.txt \
    && echo "parity: C++ and Python exact-route runs give the same verdict on every SR-BH precal + lap 1 request" \
    || echo "parity: C++ and Python exact-route verdicts DIFFER" ) > results/bench/parity_exact.log 2>&1 &
ENGINES=coraza JOBS=3 sh bench/waf/run_all.sh > results/bench/waf/run_coraza.log 2>&1
mkdir -p results/bench/logs
JOBS=3 sh bench/run_method.sh isolation_forest > results/bench/logs/run_isolation_forest.log 2>&1
wait
echo "queue done"
