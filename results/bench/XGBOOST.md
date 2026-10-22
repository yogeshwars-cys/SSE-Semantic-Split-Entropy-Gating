# xgboost

Same test rows as SSE (results/bench/SSE_5LAPS*.md); protocol in bench/tabular_baselines.py.

| Dataset | detection | false alarms | precision | F1 | wrong | ROC-AUC | det. @1% FA | model size | us/record (batch / single) |
|---|---|---|---|---|---|---|---|---|---|
| srbh | 93.0% | 44.6% | 82.4% | 0.874 | 71,558 | 0.830 | 0.0% | 2.35 MB | 34.7 / 1444 |
| csic | 92.8% | 4.5% | 93.5% | 0.932 | 3,419 | 0.992 | 86.7% | 2.10 MB | 62.1 / 2119 |
| cic_Friday-Afternoon-DDos | 100.0% | 0.0% | 100.0% | 1.000 | 10 | 1.000 | 100.0% | 0.52 MB | 8.7 / 839 |
| cic_Friday-Afternoon-PortScan | 100.0% | 0.0% | 100.0% | 1.000 | 11 | 1.000 | 100.0% | 0.50 MB | 7.8 / 891 |
| cic_Thursday-Morning-WebAttacks | 99.3% | 0.0% | 99.8% | 0.996 | 17 | 1.000 | 100.0% | 0.47 MB | 11.0 / 1655 |
| cic_Wednesday | 100.0% | 0.0% | 99.9% | 0.999 | 49 | 1.000 | 100.0% | 0.74 MB | 16.4 / 625 |
| unsw | 98.0% | 26.5% | 81.9% | 0.892 | 10,709 | 0.982 | 85.5% | 3.34 MB | 22.1 / 1148 |
| kitsune_arp_mitm | 100.0% | 0.0% | 100.0% | 1.000 | 11 | 1.000 | 100.0% | 0.42 MB | 8.9 / 463 |
| kitsune_mirai | not run | | | | | | | | |
| kitsune_os_scan | 100.0% | 0.0% | 100.0% | 1.000 | 4 | 1.000 | 100.0% | 0.40 MB | 7.8 / 1690 |
| kitsune_ssdp_flood | 100.0% | 0.2% | 100.0% | 1.000 | 6 | 1.000 | 100.0% | 0.37 MB | 4.8 / 813 |
| kitsune_syn_dos | 100.0% | 0.0% | 100.0% | 1.000 | 4 | 1.000 | 100.0% | 0.39 MB | 7.8 / 895 |
