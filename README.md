# SSE: Semantic Split Entropy Gating

A small, CPU-only anomaly gate that learns from its own mistakes. Two cheap gates score every item against normal
traffic, and a calibration loop retrains them on every wrong verdict:

- **Statistical gate** (`sse/anomaly_model.py`): surprisal of field names, length and entropy z-scores per field.
- **Semantic gate** (`sse/semantic_model.py`): hashed character 3-gram TF-IDF distance to the nearest normal value.
- **Calibration** (`sse/calibration.py`): per-(route, field) thresholds, plus the gates' own updates. A miss sharpens
  the blamed field and its rare 3-grams; a false alarm adds the value to the normal baseline.

About 380 lines, numpy only, ~1 ms per request on one core, fitted on normal traffic only.

## Headline results

| Benchmark | Setting | Result |
|---|---|---|
| Web cache poisoning defence (6 attack vectors, Docker) | gates + semantic layer + split-delivery | **0% of victim views harmed on every vector** (unprotected cache: 70–100%) |
| Adaptive attacker with an offline probe, 10 min | split verdicts feeding the gates | attacker pass rate **98.1% → 7.0%**, 0% harmed, 3.1% of views held |
| Reverse-proxy tradeoff index, 10 architectures | performance vs security and header compatibility | **TI 95.8, ranked first**, at +6% origin load |
| SR-BH 2020 (907k real honeypot HTTP requests, 13 attack classes) | unseen traffic, learning from every wrong verdict | **76.7% detection, 5.1% false alarms, 97.1% precision** |
| SR-BH 2020 | frozen gates (no learning) | 24% detection at 1% false alarms |
| CSIC 2010 (a different web app) | brand-new scanner, one learning lap | 87.8% detection, 9.7% false alarms |
| CIC-IDS2017 | network flows, closed loop | F1 0.94–0.99 on DDoS / PortScan / DoS; 0.03 on web attacks |
| BEIR SciFact | retrieval, supervised fine-tune | nDCG@10 0.719 (dense MiniLM: 0.645) |

All figures, methods, caveats and exclusions are in [BENCHMARKS.md](BENCHMARKS.md). The cache benchmarks ran in a
caching reverse proxy that is not part of this repo; its result files are in `results/cache/`. In the HTTP benchmarks
the teacher is the dataset label, so they are upper bounds of what the loop can do with a verifier.

## Layout

| Path | Contents |
|---|---|
| `sse/` | the gates and the calibration loop |
| `harness/` | reproducible SR-BH 2020 and CSIC 2010 runs (zero-shot, real-time recalibration, gate learning, conveyor-belt laps, cross-site transfer) |
| `ids/` | CIC-IDS2017 flow IDS and its benchmark |
| `retrieval/` | the retrieval port and its SciFact / NQ benchmarks |
| `results/` | every JSON result and run log behind BENCHMARKS.md (`cache/`, `srbh/`, `csic/`, `ids/`, `retrieval/`) |
| `docs/figures/` | figures used in BENCHMARKS.md |
| `data/` | how to fetch the datasets (not included) |
| `demo/` | an authorised closed-lab, three-laptop live demo of the cache-poisoning defence (Part I) — see [`demo/README.md`](demo/README.md) |

## Reproduce (HTTP benchmarks)

Fetch the data as described in [data/README.md](data/README.md), then from the repo root:

```
python -I harness/run_sse_srbh.py   data/sr-bh-2020/data_capec_multilabel.csv sse results/srbh/sse_srbh2020_results.json
python -I harness/replay_realtime.py data/sr-bh-2020/data_capec_multilabel.csv sse results/srbh/realtime_results.json
SSE_DETECTOR=gates_learn_route python -I harness/replay_learn.py data/sr-bh-2020/data_capec_multilabel.csv sse results/srbh/learn_route_results.json
python -I harness/belt.py           data/sr-bh-2020/data_capec_multilabel.csv sse gates_learn_route results/srbh/belt_gates_learn_route.json
CSIC_ONLY=new python -I harness/csic_lap.py data/csic-2010 sse none results/csic/csic_new_results.json
```

Runs are deterministic (fixed seeds, time-ordered replay). Trained scanners are saved as `.pkl` files next to the
output (95–500 MB, not committed).

## Licence

Proprietary. See [LICENSE](LICENSE).
