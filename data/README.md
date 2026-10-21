# Datasets (not included)

Each dataset stays under its own licence and terms. Download them into this folder:

| Dataset | Used by | Source |
|---|---|---|
| SR-BH 2020 | `harness/run_sse_srbh.py`, `replay_*.py`, `belt.py` | Harvard Dataverse, doi:10.7910/DVN/OGOIXX (`data_capec_multilabel.csv`, 436 MB) |
| CSIC 2010 | `harness/csic_*.py` | `normalTrafficTraining.txt`, `normalTrafficTest.txt`, `anomalousTrafficTest.txt` |
| CIC-IDS2017 | `ids/benchmark_sse_cic.py` | official CSVs (`c01dsnap/CIC-IDS2017`) |
| BEIR SciFact, NQ-100k | `retrieval/benchmark_sse*.py` | BEIR; NQ slice built with bqboost (`SSE_NQ_DATA` points at it) |

```
mkdir -p data/sr-bh-2020 data/csic-2010
curl -L -o data/sr-bh-2020/data_capec_multilabel.csv "https://dataverse.harvard.edu/api/access/datafile/6319496"
for f in normalTrafficTest.txt anomalousTrafficTest.txt; do
  curl -L -o data/csic-2010/$f "https://gitlab.fing.edu.uy/gsi/web-application-attacks-datasets/-/raw/master/csic_2010/$f"
done
curl -L -o data/csic-2010/normalTrafficTraining.txt \
  "https://raw.githubusercontent.com/baksakal/HTTP-DATASET-CSIC-2010-MACHINE-LEARNING-GUI-AND-SERVER/master/normalTrafficTraining.txt"
```

Expected sizes: SR-BH 907,814 rows; CSIC 36,000 / 36,000 / 25,065 requests.
