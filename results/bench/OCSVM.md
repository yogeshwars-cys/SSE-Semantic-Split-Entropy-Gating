# ocsvm

Same test rows as SSE (results/bench/SSE_5LAPS*.md); protocol in bench/tabular_baselines.py.

| Dataset | detection | false alarms | precision | F1 | wrong | ROC-AUC | det. @1% FA | model size | us/record (batch / single) |
|---|---|---|---|---|---|---|---|---|---|
| srbh | 66.9% | 20.8% | 87.9% | 0.760 | 112,768 | 0.782 | 4.2% | 0.17 MB | 54.4 / 677 |
| csic | 52.2% | 1.1% | 97.1% | 0.679 | 12,375 | 0.845 | 51.6% | 0.13 MB | 43.7 / 429 |
| cic_Friday-Afternoon-DDos | 63.6% | 2.0% | 98.5% | 0.773 | 30,203 | 0.824 | 63.6% | 0.06 MB | 27.2 / 571 |
| cic_Friday-Afternoon-PortScan | 3.2% | 1.7% | 76.9% | 0.061 | 74,445 | 0.802 | 0.8% | 0.07 MB | 26.5 / 786 |
| cic_Thursday-Morning-WebAttacks | 3.9% | 0.7% | 8.4% | 0.054 | 2,715 | 0.937 | 4.0% | 0.07 MB | 36.7 / 202 |
| cic_Wednesday | 21.7% | 1.4% | 90.8% | 0.351 | 37,144 | 0.896 | 21.6% | 0.05 MB | 14.5 / 524 |
| unsw | 45.3% | 2.6% | 95.6% | 0.614 | 25,758 | 0.759 | 28.9% | 0.08 MB | 23.1 / 482 |
| kitsune_arp_mitm | 64.2% | 86.0% | 93.5% | 0.761 | 76,561 | 0.311 | 0.0% | 0.07 MB | 18.7 / 737 |
| kitsune_mirai | not run | | | | | | | | |
| kitsune_os_scan | 100.0% | 67.5% | 23.4% | 0.380 | 111,835 | 0.897 | 0.0% | 0.06 MB | 20.9 / 422 |
| kitsune_ssdp_flood | 100.0% | 63.9% | 98.9% | 0.995 | 2,109 | 0.944 | 0.0% | 0.10 MB | 23.9 / 425 |
| kitsune_syn_dos | 24.6% | 4.3% | 33.1% | 0.282 | 8,812 | 0.661 | 24.6% | 0.10 MB | 18.8 / 752 |
