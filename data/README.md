# Datasets (not included)

Each dataset stays under its own licence and terms. Download them into this folder.

## Web attack detection and IDS

| Dataset | Used by | Source |
|---|---|---|
| SR-BH 2020 | `harness/run_sse_srbh.py`, `replay_*.py`, `belt.py` | Harvard Dataverse, doi:10.7910/DVN/OGOIXX (`data_capec_multilabel.csv`, 436 MB) |
| CSIC 2010 | `harness/csic_*.py`, cache benchmarks (Part I, 1.3) | `normalTrafficTraining.txt`, `normalTrafficTest.txt`, `anomalousTrafficTest.txt` |
| CIC-IDS2017 | `ids/benchmark_sse_cic.py`, `bench/flow_prepare.py cic` | official MachineLearningCSV files (Hugging Face `c01dsnap/CIC-IDS2017`) into `data/cic-ids2017/` |
| UNSW-NB15 | `bench/flow_prepare.py unsw` | official partition (Hugging Face `Mireu-Lab/UNSW-NB15`; its `test.csv` is the official training set and `train.csv` the testing set) into `data/unsw-nb15/` |
| Kitsune Network Attack | `bench/flow_prepare.py kitsune` | UCI dataset 516; per-packet features and labels of ARP MitM, Mirai, OS Scan, SSDP Flood, SYN DoS into `data/kitsune/` (`bench/zip_stream_extract.py`, pcaps skipped) |
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

# IDS benchmark streams (bench/run_sse.sh)
for f in Monday-WorkingHours Tuesday-WorkingHours Wednesday-workingHours Thursday-WorkingHours-Morning-WebAttacks \
         Thursday-WorkingHours-Afternoon-Infilteration Friday-WorkingHours-Morning \
         Friday-WorkingHours-Afternoon-PortScan Friday-WorkingHours-Afternoon-DDos; do
  curl -L -o "data/cic-ids2017/$f.pcap_ISCX.csv" "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/$f.pcap_ISCX.csv"
done
for f in train.csv test.csv; do
  curl -L -o data/unsw-nb15/$f "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main/$f"
done
curl -sS "https://archive.ics.uci.edu/static/public/516/kitsune+network+attack+dataset.zip" |
  python -I bench/zip_stream_extract.py data/kitsune mirai/ syn_dos/ ssdp_flood/ os_scan/ arp_mitm/
```

Expected sizes: SR-BH 907,814 rows; CSIC 36,000 / 36,000 / 25,065 requests; SecLists 1,084 header names.
