# C++ SSE, fully binary semantic store + int8 rescore, 5 belt laps, with the exact-route threshold layer

Lap 1 is traffic never seen; laps 2-5 replay the same test stream, learning every lap (convergence, not
generalisation). Teacher = dataset label. Cells: detection / false alarms (F1), wrong verdicts below.

| Dataset | test records (attacks) | lap 1 | lap 2 | lap 3 | lap 4 | lap 5 | us/record (frozen) | store after lap 1 |
|---|---|---|---|---|---|---|---|---|
| srbh | 384,629 (266,246) | 95.7% / 8.7% (0.959)<br>21,871 wrong | 95.3% / 8.0% (0.958)<br>21,979 wrong | 98.1% / 5.3% (0.979)<br>11,341 wrong | 98.5% / 4.2% (0.983)<br>8,952 wrong | 98.8% / 3.8% (0.985)<br>7,827 wrong | 161 | 22.7 MB |
| csic | 61,065 (25,065) | 88.0% / 13.7% (0.848)<br>7,922 wrong | 90.1% / 12.8% (0.864)<br>7,082 wrong | 91.9% / 10.8% (0.886)<br>5,908 wrong | 92.6% / 9.4% (0.899)<br>5,227 wrong | 92.8% / 7.7% (0.910)<br>4,581 wrong | 297 | 18.8 MB |
| cic_Friday-Afternoon-DDos | 120,000 (80,969) | 100.0% / 0.5% (0.999)<br>236 wrong | 100.0% / 0.4% (0.999)<br>197 wrong | 100.0% / 0.3% (0.999)<br>127 wrong | 100.0% / 0.2% (0.999)<br>112 wrong | 100.0% / 0.2% (0.999)<br>114 wrong | 87 | 13.5 MB |
| cic_Friday-Afternoon-PortScan | 120,000 (76,137) | 99.8% / 1.0% (0.996)<br>570 wrong | 99.8% / 0.8% (0.997)<br>483 wrong | 99.7% / 0.8% (0.996)<br>540 wrong | 99.7% / 0.8% (0.997)<br>528 wrong | 99.8% / 0.7% (0.997)<br>517 wrong | 31 | 13.9 MB |
| cic_Thursday-Morning-WebAttacks | 120,000 (1,954) | 88.1% / 0.6% (0.784)<br>948 wrong | 88.0% / 0.5% (0.806)<br>830 wrong | 87.9% / 0.5% (0.810)<br>804 wrong | 88.1% / 0.4% (0.819)<br>762 wrong | 88.1% / 0.4% (0.823)<br>742 wrong | 61 | 14.2 MB |
| cic_Wednesday | 120,000 (46,153) | 99.4% / 1.0% (0.989)<br>1,002 wrong | 99.5% / 0.8% (0.991)<br>788 wrong | 99.6% / 0.6% (0.993)<br>668 wrong | 99.6% / 0.6% (0.993)<br>622 wrong | 99.6% / 0.5% (0.994)<br>582 wrong | 68 | 15.3 MB |
| unsw | 82,332 (45,332) | 78.3% / 29.9% (0.773)<br>20,875 wrong | 79.3% / 28.7% (0.782)<br>20,011 wrong | 79.8% / 28.1% (0.788)<br>19,528 wrong | 80.1% / 27.7% (0.790)<br>19,273 wrong | 80.3% / 27.4% (0.792)<br>19,082 wrong | 63 | 51.7 MB |
| kitsune_arp_mitm | 200,000 (190,178) | 100.0% / 1.8% (0.999)<br>243 wrong | 100.0% / 0.7% (1.000)<br>90 wrong | 100.0% / 0.3% (1.000)<br>38 wrong | 100.0% / 0.2% (1.000)<br>25 wrong | 100.0% / 0.2% (1.000)<br>19 wrong | 125 | 17.7 MB |
| kitsune_mirai | 200,000 (200,000) | 100.0% / 0.0% (1.000)<br>1 wrong | 100.0% / 0.0% (1.000)<br>0 wrong | 100.0% / 0.0% (1.000)<br>0 wrong | 100.0% / 0.0% (1.000)<br>0 wrong | 100.0% / 0.0% (1.000)<br>0 wrong | 60 | 25.4 MB |
| kitsune_os_scan | 200,000 (34,212) | 100.0% / 0.0% (0.999)<br>43 wrong | 100.0% / 0.0% (1.000)<br>7 wrong | 100.0% / 0.0% (1.000)<br>2 wrong | 100.0% / 0.0% (1.000)<br>0 wrong | 100.0% / 0.0% (1.000)<br>0 wrong | 23 | 16.6 MB |
| kitsune_ssdp_flood | 200,000 (196,700) | 100.0% / 0.6% (1.000)<br>23 wrong | 100.0% / 0.3% (1.000)<br>14 wrong | 100.0% / 0.3% (1.000)<br>12 wrong | 100.0% / 0.2% (1.000)<br>7 wrong | 100.0% / 0.1% (1.000)<br>5 wrong | 10 | 21.8 MB |
| kitsune_syn_dos | 88,927 (7,038) | 95.9% / 1.0% (0.926)<br>1,079 wrong | 97.2% / 0.7% (0.948)<br>746 wrong | 97.6% / 0.5% (0.958)<br>601 wrong | 98.0% / 0.5% (0.964)<br>518 wrong | 98.1% / 0.4% (0.967)<br>477 wrong | 37 | 21.3 MB |

Wrong verdicts, base (token thresholds only) -> this variant:

| Dataset | lap 1 | lap 5 |
|---|---|---|
| srbh | 27,936 -> 21,871 | 25,623 -> 7,827 |
| csic | 8,001 -> 7,922 | 3,905 -> 4,581 |
| cic_Friday-Afternoon-DDos | 472 -> 236 | 559 -> 114 |
| cic_Friday-Afternoon-PortScan | 584 -> 570 | 640 -> 517 |
| cic_Thursday-Morning-WebAttacks | 1,580 -> 948 | 921 -> 742 |
| cic_Wednesday | 2,789 -> 1,002 | 2,931 -> 582 |
| unsw | 18,538 -> 20,875 | 17,789 -> 19,082 |
| kitsune_arp_mitm | 366 -> 243 | 174 -> 19 |
| kitsune_mirai | 9 -> 1 | 0 -> 0 |
| kitsune_os_scan | 122 -> 43 | 0 -> 0 |
| kitsune_ssdp_flood | 46 -> 23 | 4 -> 5 |
| kitsune_syn_dos | 1,277 -> 1,079 | 400 -> 477 |
