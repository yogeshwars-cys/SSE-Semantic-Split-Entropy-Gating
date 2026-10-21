# Split-delivery caching on real open datasets (Docker). n=30000

## NASA-HTTP 1995: 30000 real requests, 2286 distinct URLs, top-3 URLs = 15.6% of traffic, mean size 21.4 KB

### NASA-HTTP 1995, benign replay

| design | views | req/s | hit ratio | clean | saved | HARMED | harmed % | p50 ms | p99 ms | origin reqs | origin verify | origin health |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vulnerable cache | 30000 | 606 | 0.904 | 30000 | 0 | 0 | 0.00% | 1.8 | 47.9 | 2888 | 0 | 0 |
| v1 full screen + browser flag | 30000 | 576 | 0.902 | 30000 | 0 | 0 | 0.00% | 2.6 | 49.1 | 2928 | 0 | 0 |
| ghost cache (hardened) + gate | 30000 | 577 | 0.902 | 30000 | 0 | 0 | 0.00% | 2.6 | 49.3 | 2926 | 0 | 26 |
| split: regions first + late check | 30000 | 608 | 0.904 | 30000 | 0 | 0 | 0.00% | 2.1 | 48.3 | 2881 | 0 | 0 |
| STACKED: allowlist + no-cache-errors + split regions | 30000 | 603 | 0.904 | 30000 | 0 | 0 | 0.00% | 2.2 | 48.3 | 2894 | 0 | 0 |

### NASA-HTTP 1995, under attack (attacker poisons the 3 hottest URLs)

| design | views | req/s | hit ratio | clean | saved | HARMED | harmed % | p50 ms | p99 ms | origin reqs | origin verify | origin health |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vulnerable cache | 30000 | 582 | 0.906 | 28961 | 0 | 1039 | 3.46% | 2.4 | 48.9 | 2908 | 0 | 0 |
| v1 full screen + browser flag | 30000 | 573 | 0.905 | 30000 | 0 | 0 | 0.00% | 2.6 | 49.0 | 2932 | 0 | 0 |
| ghost cache (hardened) + gate | 30000 | 579 | 0.906 | 28423 | 0 | 1577 | 5.26% | 2.5 | 49.0 | 2913 | 0 | 25 |
| split: regions first + late check | 30000 | 598 | 0.906 | 29912 | 88 | 0 | 0.00% | 2.1 | 48.3 | 2905 | 0 | 0 |
| STACKED: allowlist + no-cache-errors + split regions | 30000 | 601 | 0.906 | 30000 | 0 | 0 | 0.00% | 2.2 | 48.3 | 2904 | 0 | 0 |

## CSIC 2010: 16000 real GET requests (normal: 8000, anomalous (attack payloads): 8000); 100.0% carry at least one header outside the 'standard' list (most common non-standard: pragma, accept-charset)

### CSIC normal: false alarms must be 0 (the origin never reflects these headers)

| design | views | hit ratio | false 'saved' | false 'harmed' | origin reqs | origin reqs per view | baseline fetches | origin verify | p50 ms | p99 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| vulnerable cache | 8000 | 0.821 | 0 | 0 | 1434 | 0.18 | - | 0 | 5.2 | 16.1 |
| v1 full screen + browser flag | 8000 | 0.821 | 0 | 0 | 2868 | 0.36 | 1434 | 0 | 4.7 | 29.7 |
| ghost cache (hardened) + gate | 8000 | 0.821 | 0 | 0 | 1434 | 0.18 | - | 0 | 5.3 | 13.9 |
| split: regions first + late check | 8000 | 0.821 | 0 | 0 | 1434 | 0.18 | - | 0 | 4.8 | 15.5 |
| STACKED: allowlist + no-cache-errors + split regions | 8000 | 0.821 | 0 | 0 | 1434 | 0.18 | - | 0 | 4.8 | 14.8 |

### CSIC anomalous (attack payloads): false alarms must be 0 (the origin never reflects these headers)

| design | views | hit ratio | false 'saved' | false 'harmed' | origin reqs | origin reqs per view | baseline fetches | origin verify | p50 ms | p99 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| vulnerable cache | 8000 | 0.371 | 0 | 0 | 5033 | 0.63 | - | 0 | 7.4 | 22.0 |
| v1 full screen + browser flag | 8000 | 0.371 | 0 | 0 | 10066 | 1.26 | 5033 | 0 | 10.9 | 35.0 |
| ghost cache (hardened) + gate | 8000 | 0.371 | 0 | 0 | 5033 | 0.63 | - | 0 | 6.0 | 16.5 |
| split: regions first + late check | 8000 | 0.371 | 0 | 0 | 5033 | 0.63 | - | 0 | 7.5 | 23.3 |
| STACKED: allowlist + no-cache-errors + split regions | 8000 | 0.371 | 0 | 0 | 5033 | 0.63 | - | 0 | 8.1 | 23.7 |

raw datasets deleted
done
