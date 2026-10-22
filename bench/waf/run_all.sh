#!/bin/sh
# WAF configurations on both HTTP test streams, paranoia 1-4, then scoring (bench/waf/score.py).
# ENGINES: modsec (ModSecurity + CRS 3.3.5) and / or coraza (Coraza + CRS 4.25). Finished runs (.done) are skipped.
# usage: [ENGINES="modsec coraza"] [JOBS=4] sh bench/waf/run_all.sh   (from anywhere)
cd "$(dirname "$0")/../.."
for pl in 1 2 3 4; do sed "s/@PL@/$pl/g" bench/waf/crs.conf.in > data/crs_pl$pl.conf; done
for d in srbh csic; do [ -f data/bench_${d}_req.bin ] || python3 -I bench/waf/export_requests.py $d data/bench_${d}_req.bin; done
mkdir -p results/bench/waf
for e in ${ENGINES:-modsec coraza}; do for d in srbh csic; do for pl in 1 2 3 4; do echo "$e $d $pl"; done; done; done |
  xargs -P "${JOBS:-4}" -L 1 sh -c 'out=results/bench/waf/$0_$1_pl$2; [ -f $out.done ] && exit 0;
    if [ $0 = modsec ]; then ./bench/waf/modsec_eval data/crs_pl$2.conf data/bench_$1_req.bin $out.txt 2> $out.log;
    else ./bench/waf/coraza/coraza_eval $2 data/bench_$1_req.bin $out.txt 2> $out.log; fi && touch $out.done'
for e in ${ENGINES:-modsec coraza}; do python3 -I bench/waf/score.py $e; done
