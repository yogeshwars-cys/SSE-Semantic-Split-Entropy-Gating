# Datasets (not included)

Each dataset stays under its own licence and terms. Download them into this folder.

## Web attack detection and IDS

| Dataset | Used by | Source |
|---|---|---|
| SR-BH 2020 | `harness/run_sse_srbh.py`, `replay_*.py`, `belt.py` | Harvard Dataverse, doi:10.7910/DVN/OGOIXX (`data_capec_multilabel.csv`, 436 MB) |
| CSIC 2010 | `harness/csic_*.py`, cache benchmarks (Part I, 1.3) | `normalTrafficTraining.txt`, `normalTrafficTest.txt`, `anomalousTrafficTest.txt` |
| CIC-IDS2017 | `ids/benchmark_sse_cic.py` | official CSVs (`c01dsnap/CIC-IDS2017`) |
| BEIR SciFact, NQ-100k | `retrieval/benchmark_sse*.py` | BEIR; NQ slice built with bqboost (`SSE_NQ_DATA` points at it) |

## Cache benchmarks (BENCHMARKS.md, Part I)

| Dataset | Used in | Source |
|---|---|---|
| NASA-HTTP 1995 access log (Jul 1995) | 1.3 real-world trace performance | https://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html |
| Wikimedia CDN 2019 caching trace (`cache-t-00`) | 1.4 A | https://wikitech.wikimedia.org/wiki/Data_Platform/Data_Lake/Traffic/Caching |
| intoli user-agents (real User-Agent frequencies) | 1.4 B | https://github.com/intoli/user-agents |
| SecLists Param Miner header names (1,084) | 1.4 C | https://github.com/danielmiessler/SecLists (`Discovery/Web-Content/BurpSuite-ParamMiner/lowercase-headers`) |

## Download

```
mkdir -p data/sr-bh-2020 data/csic-2010 data/cache
curl -L -o data/sr-bh-2020/data_capec_multilabel.csv "https://dataverse.harvard.edu/api/access/datafile/6319496"
for f in normalTrafficTest.txt anomalousTrafficTest.txt; do
  curl -L -o data/csic-2010/$f "https://gitlab.fing.edu.uy/gsi/web-application-attacks-datasets/-/raw/master/csic_2010/$f"
done
curl -L -o data/csic-2010/normalTrafficTraining.txt \
  "https://raw.githubusercontent.com/baksakal/HTTP-DATASET-CSIC-2010-MACHINE-LEARNING-GUI-AND-SERVER/master/normalTrafficTraining.txt"

curl -L -o data/cache/NASA_access_log_Jul95.gz "https://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz"
curl -L -o data/cache/cache-t-00.gz "https://analytics.wikimedia.org/published/datasets/caching/2019/text/cache-t-00.gz"
curl -L -o data/cache/user-agents.json.gz "https://raw.githubusercontent.com/intoli/user-agents/master/src/user-agents.json.gz"
curl -L -o data/cache/lowercase-headers \
  "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Discovery/Web-Content/BurpSuite-ParamMiner/lowercase-headers"
```

Expected sizes: SR-BH 907,814 rows; CSIC 36,000 / 36,000 / 25,065 requests; SecLists 1,084 header names.
