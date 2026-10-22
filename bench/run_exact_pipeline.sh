#!/bin/sh
# After the ModSecurity runs: the exact-route variant of C++ SSE, 5 laps on every dataset (4 at a time), plus the
# Python reference on SR-BH lap 1 (HYB_QUANT=3 HYB_EXACT=1) and a request-by-request comparison of the two.
# usage: sh bench/run_exact_pipeline.sh   (from the repo root)
cd "$(dirname "$0")/.."
while [ ! -f results/bench/srbh/modsec.json ] || [ ! -f results/bench/csic/modsec.json ]; do sleep 30; done
T=data/traces; mkdir -p $T
HYB_QUANT=3 HYB_EXACT=1 HYB_TRACE=$T/py_srbh_exact.txt python3 -I harness/replay_hybrid2.py \
  data/sr-bh-2020/data_capec_multilabel.csv sse results/srbh/hybrid2/full_binary_int8_exact.json \
  > results/srbh/hybrid2/full_binary_int8_exact.log 2>&1 &
PY=$!
JOBS=3 VARIANT=exact TRACE_DIR=$T sh bench/run_sse.sh
wait $PY
head -c "$(wc -c < $T/py_srbh_exact.txt)" $T/srbh_exact.txt | cmp - $T/py_srbh_exact.txt \
  && echo "parity: C++ and Python exact-route runs give the same verdict on every SR-BH precal + lap 1 request" \
  || echo "parity: C++ and Python exact-route verdicts DIFFER"
grep "== full lap 1" results/srbh/hybrid2/full_binary_int8_exact.log
