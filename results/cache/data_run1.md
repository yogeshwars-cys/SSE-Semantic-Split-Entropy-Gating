# Split-delivery caching on real open datasets (Docker). n=30000

## NASA-HTTP 1995: 30000 real requests, 2286 distinct URLs, top-3 URLs = 15.6% of traffic, mean size 21.4 KB

### NASA-HTTP 1995, benign replay

| design | views | req/s | hit ratio | clean | saved | HARMED | harmed % | p50 ms | p99 ms | origin reqs | origin verify | origin health |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vulnerable cache | 30000 | 613 | 0.904 | 30000 | 0 | 0 | 0.00% | 1.7 | 46.8 | 2881 | 0 | 0 |
| v1 full screen + browser flag | 30000 | 610 | 0.904 | 30000 | 0 | 0 | 0.00% | 1.9 | 47.3 | 2877 | 0 | 0 |
| ghost cache (hardened) + gate | 30000 | 614 | 0.904 | 30000 | 0 | 0 | 0.00% | 1.8 | 46.6 | 2870 | 0 | 24 |
| split: regions first + late check | 30000 | 634 | 0.905 | 30000 | 0 | 0 | 0.00% | 1.5 | 47.2 | 2855 | 0 | 0 |

### NASA-HTTP 1995, under attack (attacker poisons the 3 hottest URLs)

| design | views | req/s | hit ratio | clean | saved | HARMED | harmed % | p50 ms | p99 ms | origin reqs | origin verify | origin health |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vulnerable cache | 30000 | 608 | 0.906 | 28890 | 0 | 1110 | 3.70% | 1.8 | 47.8 | 2894 | 0 | 0 |
| v1 full screen + browser flag | 30000 | 605 | 0.906 | 30000 | 0 | 0 | 0.00% | 2.0 | 47.9 | 2909 | 0 | 0 |
| ghost cache (hardened) + gate | 30000 | 607 | 0.907 | 28237 | 0 | 1763 | 5.88% | 1.9 | 47.9 | 2890 | 0 | 24 |
| split: regions first + late check | 30000 | 630 | 0.907 | 29971 | 29 | 0 | 0.00% | 1.6 | 47.3 | 2870 | 0 | 0 |

## Wikimedia CDN 2019 (text): 30000 real requests, 26710 distinct URLs, top-3 URLs = 2.1% of traffic, mean size 33.7 KB

### Wikimedia CDN 2019 (text), benign replay

| design | views | req/s | hit ratio | clean | saved | HARMED | harmed % | p50 ms | p99 ms | origin reqs | origin verify | origin health |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vulnerable cache | 30000 | 1096 | 0.109 | 30000 | 0 | 0 | 0.00% | 6.8 | 18.1 | 26724 | 0 | 0 |
| v1 full screen + browser flag | 30000 | 935 | 0.109 | 30000 | 0 | 0 | 0.00% | 7.9 | 20.2 | 26733 | 0 | 0 |
| ghost cache (hardened) + gate | 30000 | 1188 | 0.109 | 30000 | 0 | 0 | 0.00% | 6.3 | 15.8 | 26722 | 0 | 12 |
| split: regions first + late check | 30000 | 894 | 0.108 | 30000 | 0 | 0 | 0.00% | 7.9 | 25.1 | 26754 | 0 | 0 |

### Wikimedia CDN 2019 (text), under attack (attacker poisons the 3 hottest URLs)

| design | views | req/s | hit ratio | clean | saved | HARMED | harmed % | p50 ms | p99 ms | origin reqs | origin verify | origin health |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vulnerable cache | 30000 | 1009 | 0.125 | 29491 | 0 | 509 | 1.70% | 7.3 | 18.9 | 26723 | 0 | 0 |
| v1 full screen + browser flag | 30000 | 981 | 0.125 | 30000 | 0 | 0 | 0.00% | 7.5 | 19.5 | 26745 | 0 | 0 |
| ghost cache (hardened) + gate | 30000 | 1250 | 0.122 | 29491 | 0 | 509 | 1.70% | 6.1 | 14.8 | 26724 | 0 | 12 |
| split: regions first + late check | 30000 | 935 | 0.125 | 29918 | 82 | 0 | 0.00% | 7.7 | 20.7 | 26756 | 0 | 0 |

## UA: real User-Agent distribution (intoli): 10000 records, K=764 distinct strings, entropy 5.61 bits, effective cardinality 49 (6.4% of K); top-1 18.6%, top-10 55.6%, top-100 92.2% of traffic

| cache key | UA stream | requests | overall hit ratio | steady (last 25%) | ceiling = 1 - distinct/N | req/s |
|---|---|---|---|---|---|---|
| default key (header unkeyed) | real distribution | 40000 | 1.000 | 1.000 | 0.981 | 1826 |
| keyed on User-Agent | real distribution | 40000 | 0.981 | 0.997 | 0.981 | 1576 |
| keyed on User-Agent | uniform over same K | 40000 | 0.981 | 1.000 | 0.981 | 1463 |
| keyed on User-Agent | Zipf s=1 over same K | 40000 | 0.981 | 1.000 | 0.981 | 1533 |

## CSIC 2010: 16000 real GET requests (normal: 8000, anomalous (attack payloads): 8000); 100.0% carry at least one header outside the 'standard' list (most common non-standard: pragma, accept-charset)

### CSIC normal: false alarms must be 0 (the origin never reflects these headers)

| design | views | hit ratio | false 'saved' | false 'harmed' | origin reqs | origin reqs per view | baseline fetches | origin verify | p50 ms | p99 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| vulnerable cache | 8000 | 0.821 | 0 | 0 | 1434 | 0.18 | - | 0 | 4.9 | 17.2 |
| v1 full screen + browser flag | 8000 | 0.821 | 0 | 0 | 2868 | 0.36 | 1434 | 0 | 5.8 | 32.4 |
| ghost cache (hardened) + gate | 8000 | 0.821 | 0 | 0 | 1434 | 0.18 | - | 0 | 6.7 | 17.2 |
| split: regions first + late check | 8000 | 0.821 | 0 | 0 | 1434 | 0.18 | - | 0 | 4.3 | 13.0 |

### CSIC anomalous (attack payloads): false alarms must be 0 (the origin never reflects these headers)

| design | views | hit ratio | false 'saved' | false 'harmed' | origin reqs | origin reqs per view | baseline fetches | origin verify | p50 ms | p99 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| vulnerable cache | 8000 | 0.371 | 0 | 118 | 5033 | 0.63 | - | 0 | 8.4 | 23.5 |
| v1 full screen + browser flag | 8000 | 0.371 | 0 | 118 | 10066 | 1.26 | 5033 | 0 | 13.6 | 51.3 |
| ghost cache (hardened) + gate | 8000 | 0.371 | 0 | 118 | 5033 | 0.63 | - | 0 | 5.8 | 15.7 |
| split: regions first + late check | 8000 | 0.371 | 0 | 0 | 5033 | 0.63 | - | 0 | 8.1 | 25.4 |

## HDRS: 1084 unkeyed-header candidates (SecLists Param Miner list), 152 injection payload values

### Header names, annotated origin (reflection sits inside the origin's <!--dyn--> region): 1084 of 1084 names let an attacker poison the plain cache

| design | poisonable names | stopped (blocked/held) | STILL POISONED | still poisoned % | payload values | payloads stopped | payloads still poisoned |
|---|---|---|---|---|---|---|---|
| vulnerable cache | 1084 | 0 | 1084 | 100.0% | 152 | 0 | 152 |
| v1 full screen + browser flag | 1084 | 0 | 7 | 0.6% | 152 | 0 | 0 |
| ghost cache (hardened) + gate | 1084 | 0 | 1084 | 100.0% | 152 | 0 | 152 |
| split: regions first + late check | 1084 | 1084 | 0 | 0.0% | 152 | 152 | 0 |

### Header names, unannotated origin (reflection is plain text in the page): 1084 of 1084 names let an attacker poison the plain cache

| design | poisonable names | stopped (blocked/held) | STILL POISONED | still poisoned % | payload values | payloads stopped | payloads still poisoned |
|---|---|---|---|---|---|---|---|
| vulnerable cache | 1084 | 0 | 1084 | 100.0% | 152 | 0 | 152 |
| v1 full screen + browser flag | 1084 | 0 | 7 | 0.6% | 152 | 0 | 0 |
| ghost cache (hardened) + gate | 1084 | 0 | 1084 | 100.0% | 152 | 0 | 152 |
| split: regions first + late check | 1084 | 0 | 1084 | 100.0% | 152 | 2 | 150 |

Names on the v1 'standard' list (never screened by v1): accept, accept-encoding, accept-language, cache-control, cookie, referer, user-agent

raw datasets deleted
done
