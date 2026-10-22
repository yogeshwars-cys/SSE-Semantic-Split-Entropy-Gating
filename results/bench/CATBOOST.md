# catboost

Same test rows as SSE (results/bench/SSE_5LAPS*.md); protocol in bench/tabular_baselines.py.

| Dataset | detection | false alarms | precision | F1 | wrong | ROC-AUC | det. @1% FA | model size | us/record (batch / single) |
|---|---|---|---|---|---|---|---|---|---|
| srbh | 92.1% | 44.0% | 82.5% | 0.870 | 73,169 | 0.816 | 0.6% | 3.38 MB | 12.2 / 330 |
| csic | 92.3% | 4.1% | 94.0% | 0.932 | 3,398 | 0.992 | 86.8% | 3.38 MB | 3.4 / 824 |
| cic_Friday-Afternoon-DDos | 100.0% | 0.0% | 100.0% | 1.000 | 15 | 1.000 | 100.0% | 3.35 MB | 7.7 / 1104 |
| cic_Friday-Afternoon-PortScan | 100.0% | 0.0% | 100.0% | 1.000 | 19 | 1.000 | 100.0% | 3.36 MB | 22.3 / 908 |
| cic_Thursday-Morning-WebAttacks | 98.9% | 0.0% | 99.9% | 0.994 | 23 | 1.000 | 100.0% | 3.38 MB | 71.7 / 686 |
| cic_Wednesday | 100.0% | 0.1% | 99.9% | 0.999 | 53 | 1.000 | 100.0% | 3.37 MB | 23.7 / 834 |
| unsw | 98.0% | 26.6% | 81.8% | 0.892 | 10,779 | 0.982 | 84.9% | 3.38 MB | 33.5 / 15707 |
| kitsune_arp_mitm | 100.0% | 0.0% | 100.0% | 1.000 | 9 | 1.000 | 100.0% | 3.38 MB | 8.9 / 775 |
| kitsune_mirai | not run | | | | | | | | |
| kitsune_os_scan | 100.0% | 0.0% | 100.0% | 1.000 | 6 | 1.000 | 100.0% | 3.38 MB | 10.5 / 734 |
| kitsune_ssdp_flood | 100.0% | 0.2% | 100.0% | 1.000 | 7 | 1.000 | 100.0% | 3.38 MB | 6.0 / 559 |
| kitsune_syn_dos | 100.0% | 0.0% | 100.0% | 1.000 | 2 | 1.000 | 100.0% | 3.37 MB | 2.0 / 336 |
