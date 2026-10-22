#!/bin/sh
# C++ SSE (fully binary semantic store + int8 rescore), 5 belt laps, on every IDS dataset, JOBS datasets at a time
# (default 4; each run is single-threaded and deterministic, so results do not depend on JOBS, timings do).
# SR-BH uses its native protocol; the others are prepared streams (bench/sse_prepare.py, bench/flow_prepare.py).
# Datasets that already have results/bench/<dataset>/sse_cpp.json are skipped.
# VARIANT=exact adds the exact-route threshold layer (SSE_EXACT=1); TRACE_DIR=<dir> writes per-request verdicts.
# Results: results/bench/<dataset>/sse_cpp[_<variant>].json (+ .log); summary: results/bench/SSE_5LAPS[_<variant>].md
# usage: [JOBS=4] [VARIANT=exact] [TRACE_DIR=dir] sh bench/run_sse.sh   (from the repo root)
cd "$(dirname "$0")/.."
make -s -C cpp
SUF=${VARIANT:+_$VARIANT}
[ "$VARIANT" = exact ] && export SSE_EXACT=1
export SUF TRACE_DIR
S=data/sse_streams
[ -f data/csic_stream.csv ] || python3 -I bench/sse_prepare.py csic data/csic_stream.csv
[ -f $S/unsw.csv ] || python3 -I bench/flow_prepare.py unsw $S
[ -f $S/cic_Wednesday.csv ] || python3 -I bench/flow_prepare.py cic $S
[ -f $S/kitsune_syn_dos.csv ] || python3 -I bench/flow_prepare.py kitsune $S
{
  echo "srbh data/sr-bh-2020/data_capec_multilabel.csv"
  echo "csic --prepared data/csic_stream.csv"
  for f in $S/cic_*.csv $S/unsw.csv $S/kitsune_*.csv; do echo "$(basename "$f" .csv) --prepared $f"; done
} | xargs -P "${JOBS:-4}" -L 1 sh -c '
  d=$0; o=results/bench/$d; r=$o/sse_cpp$SUF; [ -f $r.json ] && exit 0; mkdir -p $o
  tr=""; [ -n "$TRACE_DIR" ] && tr=$TRACE_DIR/$d$SUF.txt
  ./cpp/replay_srbh "$@" $r.json "$tr" 5 > $r.log 2>&1 || echo "FAILED $d"
  echo "== $d"; grep "== full lap" $r.log'
python3 -I bench/sse_summary.py $VARIANT
