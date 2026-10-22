# C++ SSE, fully binary semantic store + int8 rescore, 5 belt laps

Lap 1 is traffic never seen; laps 2-5 replay the same test stream, learning every lap (convergence, not
generalisation). Teacher = dataset label. Cells: detection / false alarms (F1), wrong verdicts below.

| Dataset | test records (attacks) | lap 1 | lap 2 | lap 3 | lap 4 | lap 5 | us/record (frozen) | store after lap 1 |
|---|---|---|---|---|---|---|---|---|
| srbh | 384,629 (266,246) | 94.2% / 10.7% (0.947)<br>27,936 wrong | 92.9% / 8.7% (0.944)<br>29,328 wrong | 92.8% / 7.8% (0.946)<br>28,417 wrong | 93.3% / 7.5% (0.949)<br>26,608 wrong | 93.6% / 7.2% (0.951)<br>25,623 wrong | 142 | 19.9 MB |
| csic | 61,065 (25,065) | 91.6% / 16.4% (0.852)<br>8,001 wrong | 92.7% / 13.6% (0.874)<br>6,713 wrong | 93.7% / 12.3% (0.886)<br>6,015 wrong | 94.8% / 10.3% (0.905)<br>4,984 wrong | 95.8% / 7.9% (0.925)<br>3,905 wrong | 259 | 13.7 MB |
| cic_Friday-Afternoon-DDos | 120,000 (80,969) | 99.9% / 1.0% (0.997)<br>472 wrong | 99.7% / 1.3% (0.995)<br>734 wrong | 99.7% / 1.2% (0.996)<br>658 wrong | 99.8% / 1.1% (0.996)<br>608 wrong | 99.8% / 0.9% (0.997)<br>559 wrong | 78 | 9.4 MB |
| cic_Friday-Afternoon-PortScan | 120,000 (76,137) | 99.9% / 1.1% (0.996)<br>584 wrong | 99.8% / 1.0% (0.996)<br>611 wrong | 99.8% / 1.1% (0.996)<br>627 wrong | 99.8% / 1.1% (0.996)<br>648 wrong | 99.8% / 1.1% (0.996)<br>640 wrong | 30 | 9.5 MB |
| cic_Thursday-Morning-WebAttacks | 120,000 (1,954) | 88.5% / 1.1% (0.687)<br>1,580 wrong | 81.7% / 1.2% (0.647)<br>1,742 wrong | 86.3% / 0.7% (0.762)<br>1,054 wrong | 86.9% / 0.6% (0.777)<br>973 wrong | 85.8% / 0.5% (0.785)<br>921 wrong | 63 | 10.0 MB |
| cic_Wednesday | 120,000 (46,153) | 97.5% / 2.2% (0.970)<br>2,789 wrong | 96.6% / 2.6% (0.962)<br>3,513 wrong | 96.6% / 2.4% (0.964)<br>3,299 wrong | 96.7% / 2.1% (0.967)<br>3,043 wrong | 96.7% / 1.9% (0.968)<br>2,931 wrong | 69 | 11.2 MB |
| unsw | 82,332 (45,332) | 74.2% / 18.5% (0.784)<br>18,538 wrong | 73.8% / 16.8% (0.787)<br>18,081 wrong | 73.9% / 16.8% (0.788)<br>18,068 wrong | 74.0% / 16.7% (0.789)<br>17,983 wrong | 74.2% / 16.5% (0.791)<br>17,789 wrong | 39 | 11.7 MB |
| kitsune_arp_mitm | 200,000 (190,178) | 100.0% / 3.1% (0.999)<br>366 wrong | 100.0% / 2.1% (0.999)<br>240 wrong | 100.0% / 2.0% (0.999)<br>229 wrong | 100.0% / 1.5% (1.000)<br>169 wrong | 100.0% / 1.5% (1.000)<br>174 wrong | 117 | 13.6 MB |
| kitsune_mirai | 200,000 (200,000) | 100.0% / 0.0% (1.000)<br>9 wrong | 100.0% / 0.0% (1.000)<br>0 wrong | 100.0% / 0.0% (1.000)<br>0 wrong | 100.0% / 0.0% (1.000)<br>0 wrong | 100.0% / 0.0% (1.000)<br>0 wrong | 57 | 21.2 MB |
| kitsune_os_scan | 200,000 (34,212) | 100.0% / 0.1% (0.998)<br>122 wrong | 100.0% / 0.0% (1.000)<br>25 wrong | 100.0% / 0.0% (1.000)<br>11 wrong | 100.0% / 0.0% (1.000)<br>0 wrong | 100.0% / 0.0% (1.000)<br>0 wrong | 21 | 12.5 MB |
| kitsune_ssdp_flood | 200,000 (196,700) | 100.0% / 1.4% (1.000)<br>46 wrong | 100.0% / 0.2% (1.000)<br>8 wrong | 100.0% / 0.1% (1.000)<br>4 wrong | 100.0% / 0.1% (1.000)<br>5 wrong | 100.0% / 0.1% (1.000)<br>4 wrong | 8 | 17.6 MB |
| kitsune_syn_dos | 88,927 (7,038) | 94.4% / 1.1% (0.912)<br>1,277 wrong | 96.7% / 0.7% (0.946)<br>783 wrong | 97.5% / 0.5% (0.957)<br>619 wrong | 98.0% / 0.4% (0.966)<br>491 wrong | 98.4% / 0.4% (0.972)<br>400 wrong | 35 | 17.2 MB |
