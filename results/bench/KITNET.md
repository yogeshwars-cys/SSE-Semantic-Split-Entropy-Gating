# kitnet

Same test rows as SSE (results/bench/SSE_5LAPS*.md); protocol in bench/tabular_baselines.py.

| Dataset | detection | false alarms | precision | F1 | wrong | ROC-AUC | det. @1% FA | model size | us/record (batch / single) |
|---|---|---|---|---|---|---|---|---|---|
| srbh | 67.2% | 20.8% | 87.9% | 0.762 | 111,816 | 0.752 | 1.2% | 0.65 MB | 420.4 / 5177 |
| csic | 48.4% | 1.3% | 96.4% | 0.645 | 13,380 | 0.892 | 46.9% | 0.93 MB | 711.9 / 11038 |
| cic_Friday-Afternoon-DDos | 63.6% | 2.4% | 98.2% | 0.772 | 30,424 | 0.748 | 0.0% | 0.24 MB | 71.7 / 9364 |
| cic_Friday-Afternoon-PortScan | 0.0% | 2.0% | 0.0% | 0.000 | 76,998 | 0.737 | 0.0% | 0.23 MB | 161.2 / 3392 |
| cic_Thursday-Morning-WebAttacks | 0.0% | 0.7% | 0.0% | 0.000 | 2,784 | 0.827 | 0.0% | 0.24 MB | 66.1 / 2443 |
| cic_Wednesday | 23.3% | 1.2% | 92.2% | 0.371 | 36,328 | 0.749 | 23.0% | 0.25 MB | 148.0 / 6855 |
| unsw | 13.4% | 2.6% | 86.3% | 0.232 | 40,220 | 0.756 | 13.4% | 0.42 MB | 477.0 / 4730 |
| kitsune_arp_mitm | 2.9% | 5.2% | 91.4% | 0.056 | 185,227 | 0.523 | 0.0% | 0.25 MB | 163.0 / 2289 |
| kitsune_mirai | not run | | | | | | | | |
| kitsune_os_scan | 100.0% | 21.1% | 49.4% | 0.661 | 35,027 | 0.897 | 5.0% | 0.24 MB | 31.0 / 2338 |
| kitsune_ssdp_flood | 100.0% | 65.8% | 98.9% | 0.995 | 2,173 | 0.964 | 16.8% | 0.23 MB | 43.5 / 1618 |
| kitsune_syn_dos | 24.6% | 1.2% | 64.6% | 0.356 | 6,255 | 0.852 | 24.6% | 0.26 MB | 209.1 / 5879 |
