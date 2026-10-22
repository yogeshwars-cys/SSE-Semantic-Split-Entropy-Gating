# random_forest

Same test rows as SSE (results/bench/SSE_5LAPS*.md); protocol in bench/tabular_baselines.py.

| Dataset | detection | false alarms | precision | F1 | wrong | ROC-AUC | det. @1% FA | model size | us/record (batch / single) |
|---|---|---|---|---|---|---|---|---|---|
| srbh | 92.9% | 57.0% | 78.6% | 0.851 | 86,452 | 0.838 | 18.0% | 166.23 MB | 73.9 / 49094 |
| csic | 93.1% | 4.5% | 93.5% | 0.933 | 3,366 | 0.992 | 86.4% | 84.44 MB | 111.9 / 57219 |
| cic_Friday-Afternoon-DDos | 100.0% | 0.0% | 100.0% | 1.000 | 28 | 1.000 | 100.0% | 2.79 MB | 23.4 / 51055 |
| cic_Friday-Afternoon-PortScan | 100.0% | 0.0% | 100.0% | 1.000 | 13 | 1.000 | 100.0% | 2.07 MB | 70.6 / 59252 |
| cic_Thursday-Morning-WebAttacks | 97.3% | 0.0% | 99.8% | 0.985 | 56 | 1.000 | 99.8% | 4.17 MB | 25.1 / 59096 |
| cic_Wednesday | 99.9% | 0.1% | 99.9% | 0.999 | 121 | 1.000 | 100.0% | 10.35 MB | 55.9 / 65448 |
| unsw | 98.6% | 27.0% | 81.7% | 0.894 | 10,646 | 0.980 | 78.5% | 311.49 MB | 155.3 / 59970 |
| kitsune_arp_mitm | not run | | | | | | | | |
| kitsune_mirai | not run | | | | | | | | |
| kitsune_os_scan | not run | | | | | | | | |
| kitsune_ssdp_flood | not run | | | | | | | | |
| kitsune_syn_dos | not run | | | | | | | | |
