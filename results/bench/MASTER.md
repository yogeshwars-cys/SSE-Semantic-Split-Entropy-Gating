# IDS benchmark: master table

Same test rows for every method on a dataset. SSE learns online from the label of every wrong verdict (an oracle teacher: an upper bound of the loop), laps 2-5 replay the same stream. Unsupervised baselines see normal traffic only, thresholded at 1% false alarms on held-out normal traffic; supervised baselines train on labelled attacks (SR-BH: 17 Jul; UNSW: official training set; others: 2-fold cross-validation over the test stream, optimistic on flow captures). CRS rows: paranoia level with the fewest wrong verdicts, picked on the test set. *Kitsune Mirai's test stream has no normal traffic (false alarms undefined). Protocols: bench/*.py docstrings.

## F1

| Method | SR-BH | CSIC | CIC DDoS | CIC Scan | CIC Web | CIC DoS | UNSW | K ARP | K Mirai* | K Scan | K SSDP | K SYN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SSE C++ binary, exact routes, lap 1 | 0.959 | 0.848 | 0.999 | 0.996 | 0.784 | 0.989 | 0.773 | 0.999 | 1.000 | 0.999 | 1.000 | 0.926 |
| SSE C++ binary, exact routes, lap 5 | 0.985 | 0.910 | 0.999 | 0.997 | 0.823 | 0.994 | 0.792 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 |
| SSE C++ binary, token routes only, lap 1 | 0.947 | 0.852 | 0.997 | 0.996 | 0.687 | 0.970 | 0.784 | 0.999 | 1.000 | 0.998 | 1.000 | 0.912 |
| ModSecurity + CRS 3.3.5 (best PL) | 0.629 | 0.631 | - | - | - | - | - | - | - | - | - | - |
| Coraza + CRS 4.25 (best PL) | - | - | - | - | - | - | - | - | - | - | - | - |
| char-CNN (supervised) | - | - | - | - | - | - | - | - | - | - | - | - |
| char-BiLSTM (supervised) | - | - | - | - | - | - | - | - | - | - | - | - |
| Isolation Forest | - | - | - | - | - | - | - | - | - | - | - | - |
| One-Class SVM | 0.760 | 0.679 | 0.773 | 0.061 | 0.054 | 0.351 | 0.614 | 0.761 | - | 0.380 | 0.995 | 0.282 |
| KitNET (Kitsune) | 0.762 | 0.645 | 0.772 | 0.000 | 0.000 | 0.371 | 0.232 | 0.056 | - | 0.661 | 0.995 | 0.356 |
| Random Forest (supervised) | 0.851 | 0.933 | 1.000 | 1.000 | 0.985 | 0.999 | 0.894 | - | - | - | - | - |
| XGBoost (supervised) | - | - | - | - | - | - | - | - | - | - | - | - |
| CatBoost (supervised) | - | - | - | - | - | - | - | - | - | - | - | - |

## Detection / false alarms

| Method | SR-BH | CSIC | CIC DDoS | CIC Scan | CIC Web | CIC DoS | UNSW | K ARP | K Mirai* | K Scan | K SSDP | K SYN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SSE C++ binary, exact routes, lap 1 | 95.7% / 8.7% | 88.0% / 13.7% | 100.0% / 0.5% | 99.8% / 1.0% | 88.1% / 0.6% | 99.4% / 1.0% | 78.3% / 29.9% | 100.0% / 1.8% | 100.0% / 0.0% | 100.0% / 0.0% | 100.0% / 0.6% | 95.9% / 1.0% |
| SSE C++ binary, exact routes, lap 5 | 98.8% / 3.8% | 92.8% / 7.7% | 100.0% / 0.2% | 99.8% / 0.7% | 88.1% / 0.4% | 99.6% / 0.5% | 80.3% / 27.4% | 100.0% / 0.2% | 100.0% / 0.0% | 100.0% / 0.0% | 100.0% / 0.1% | 98.1% / 0.4% |
| SSE C++ binary, token routes only, lap 1 | 94.2% / 10.7% | 91.6% / 16.4% | 99.9% / 1.0% | 99.9% / 1.1% | 88.5% / 1.1% | 97.5% / 2.2% | 74.2% / 18.5% | 100.0% / 3.1% | 100.0% / 0.0% | 100.0% / 0.1% | 100.0% / 1.4% | 94.4% / 1.1% |
| ModSecurity + CRS 3.3.5 (best PL) | 49.1% / 15.5% (PL3) | 55.6% / 14.3% (PL3) | - | - | - | - | - | - | - | - | - | - |
| Coraza + CRS 4.25 (best PL) | - | - | - | - | - | - | - | - | - | - | - | - |
| char-CNN (supervised) | - | - | - | - | - | - | - | - | - | - | - | - |
| char-BiLSTM (supervised) | - | - | - | - | - | - | - | - | - | - | - | - |
| Isolation Forest | - | - | - | - | - | - | - | - | - | - | - | - |
| One-Class SVM | 66.9% / 20.8% | 52.2% / 1.1% | 63.6% / 2.0% | 3.2% / 1.7% | 3.9% / 0.7% | 21.7% / 1.4% | 45.3% / 2.6% | 64.2% / 86.0% | - | 100.0% / 67.5% | 100.0% / 63.9% | 24.6% / 4.3% |
| KitNET (Kitsune) | 67.2% / 20.8% | 48.4% / 1.3% | 63.6% / 2.4% | 0.0% / 2.0% | 0.0% / 0.7% | 23.3% / 1.2% | 13.4% / 2.6% | 2.9% / 5.2% | - | 100.0% / 21.1% | 100.0% / 65.8% | 24.6% / 1.2% |
| Random Forest (supervised) | 92.9% / 57.0% | 93.1% / 4.5% | 100.0% / 0.0% | 100.0% / 0.0% | 97.3% / 0.0% | 99.9% / 0.1% | 98.6% / 27.0% | - | - | - | - | - |
| XGBoost (supervised) | - | - | - | - | - | - | - | - | - | - | - | - |
| CatBoost (supervised) | - | - | - | - | - | - | - | - | - | - | - | - |

## Wrong verdicts

| Method | SR-BH | CSIC | CIC DDoS | CIC Scan | CIC Web | CIC DoS | UNSW | K ARP | K Mirai* | K Scan | K SSDP | K SYN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SSE C++ binary, exact routes, lap 1 | 21,871 | 7,922 | 236 | 570 | 948 | 1,002 | 20,875 | 243 | 1 | 43 | 23 | 1,079 |
| SSE C++ binary, exact routes, lap 5 | 7,827 | 4,581 | 114 | 517 | 742 | 582 | 19,082 | 19 | 0 | 0 | 5 | 477 |
| SSE C++ binary, token routes only, lap 1 | 27,936 | 8,001 | 472 | 584 | 1,580 | 2,789 | 18,538 | 366 | 9 | 122 | 46 | 1,277 |
| ModSecurity + CRS 3.3.5 (best PL) | 153,914 | 16,288 | - | - | - | - | - | - | - | - | - | - |
| Coraza + CRS 4.25 (best PL) | - | - | - | - | - | - | - | - | - | - | - | - |
| char-CNN (supervised) | - | - | - | - | - | - | - | - | - | - | - | - |
| char-BiLSTM (supervised) | - | - | - | - | - | - | - | - | - | - | - | - |
| Isolation Forest | - | - | - | - | - | - | - | - | - | - | - | - |
| One-Class SVM | 112,768 | 12,375 | 30,203 | 74,445 | 2,715 | 37,144 | 25,758 | 76,561 | - | 111,835 | 2,109 | 8,812 |
| KitNET (Kitsune) | 111,816 | 13,380 | 30,424 | 76,998 | 2,784 | 36,328 | 40,220 | 185,227 | - | 35,027 | 2,173 | 6,255 |
| Random Forest (supervised) | 86,452 | 3,366 | 28 | 13 | 56 | 121 | 10,646 | - | - | - | - | - |
| XGBoost (supervised) | - | - | - | - | - | - | - | - | - | - | - | - |
| CatBoost (supervised) | - | - | - | - | - | - | - | - | - | - | - | - |

## Size and speed (median over the datasets a method ran on)

| Method | model / state size | us per record | datasets |
|---|---|---|---|
| SSE C++ binary, exact routes, lap 1 | 18.21 MB | 62 | 12 |
| SSE C++ binary, exact routes, lap 5 | 18.21 MB | 62 | 12 |
| SSE C++ binary, token routes only, lap 1 | 13.05 MB | 60 | 12 |
| ModSecurity + CRS 3.3.5 (best PL) | 26.24 MB | 2262 | 2 |
| Coraza + CRS 4.25 (best PL) | - | - | 0 |
| char-CNN (supervised) | - | - | 0 |
| char-BiLSTM (supervised) | - | - | 0 |
| Isolation Forest | - | - | 0 |
| One-Class SVM | 0.07 MB | 524 | 11 |
| KitNET (Kitsune) | 0.25 MB | 4730 | 11 |
| Random Forest (supervised) | 10.35 MB | 59096 | 7 |
| XGBoost (supervised) | - | - | 0 |
| CatBoost (supervised) | - | - | 0 |

SSE size = semantic store after lap 1 + threshold tables; its time = inference only (frozen pass), one core. CRS size = process peak RSS, time = ms/request x 1000. Others: pickled / saved model, one record at a time. Runs shared the machine, so times carry contention.
