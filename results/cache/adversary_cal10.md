
## Adaptive adversary vs gate + semantic + split + calibration (split verdicts tune both gates), 10 min

  first confirmed bypass after 0.2 s: [referer/length-matched] 'https://szf-hwv.us/9mqdwsrzowpmk8'
  minute 1: victims harmed 0.0%  held 3.3%  | attacker trials this minute 354/1974 passed
  minute 2: victims harmed 0.0%  held 3.3%  | attacker trials this minute 108/2280 passed
  minute 3: victims harmed 0.0%  held 3.2%  | attacker trials this minute 212/3039 passed
  minute 4: victims harmed 0.0%  held 3.0%  | attacker trials this minute 264/3267 passed
  minute 5: victims harmed 0.0%  held 3.2%  | attacker trials this minute 40/3445 passed
  minute 6: victims harmed 0.0%  held 2.9%  | attacker trials this minute 106/3372 passed
  minute 7: victims harmed 0.0%  held 2.9%  | attacker trials this minute 704/5093 passed
  minute 8: victims harmed 0.0%  held 3.1%  | attacker trials this minute 102/4837 passed
  minute 9: victims harmed 0.0%  held 3.0%  | attacker trials this minute 224/4097 passed
  minute 10: victims harmed 0.0%  held 3.0%  | attacker trials this minute 437/4983 passed

victim views 11532: harmed 0 (0.0%), held 356 (3.1%), clean 11176 (96.9%)
proxy: misses 80132, hits 47492; fills blocked by gate 1660, semantic 894, calibration 40351; entries flipped by split quorum 145, confirmed poison fed back 145, blames 145 (fallback 0), threshold tightenings 290
attacker: 2551 of 36388 live trials passed; first bypass at 0.2 s; 43316 exploit requests

| header | live trials | passed | pass rate |
|---|---|---|---|
| user-agent | 280 | 8 | 2.9% |
| accept-language | 0 | 0 | 0.0% |
| referer | 1921 | 72 | 3.7% |
| cookie | 2314 | 84 | 3.6% |
| accept | 0 | 0 | 0.0% |
| x-forwarded-for | 31873 | 2387 | 7.5% |

| minute | victim views | harmed % | held % | clean % | attacker trials passed |
|---|---|---|---|---|---|
| 1 | 1160 | 0.0% | 3.3% | 96.7% | 354/1974 (17.9%) |
| 2 | 1152 | 0.0% | 3.3% | 96.7% | 108/2280 (4.7%) |
| 3 | 1153 | 0.0% | 3.2% | 96.8% | 212/3039 (7.0%) |
| 4 | 1151 | 0.0% | 3.0% | 97.0% | 264/3267 (8.1%) |
| 5 | 1152 | 0.0% | 3.2% | 96.8% | 40/3445 (1.2%) |
| 6 | 1156 | 0.0% | 2.9% | 97.1% | 106/3372 (3.1%) |
| 7 | 1152 | 0.0% | 2.9% | 97.1% | 704/5093 (13.8%) |
| 8 | 1156 | 0.0% | 3.1% | 96.9% | 102/4837 (2.1%) |
| 9 | 1147 | 0.0% | 3.0% | 97.0% | 224/4097 (5.5%) |
| 10 | 1153 | 0.0% | 3.0% | 97.0% | 437/4983 (8.8%) |
| 11 | 1 | 0.0% | 0.0% | 0.0% | 0/1 (0.0%) |

calibrated thresholds at the end (route | layer | name -> threshold; global gate and semantic thresholds unchanged):
- `/refl/cookie|gate|cookie` -> -1.76
- `/refl/cookie|sem|cookie` -> -0.22
- `/refl/referer|gate|referer` -> -1.07
- `/refl/referer|sem|referer` -> -2.81
- `/refl/user-agent|gate|user-agent` -> -1.61
- `/refl/user-agent|sem|user-agent` -> 1.14
- `/refl/x-forwarded-for|gate|x-forwarded-for` -> 0.14
- `/refl/x-forwarded-for|sem|x-forwarded-for` -> -0.25

benign traffic on other pages (20,000 requests, TTL 3 s): calibrated proxy right after the attack hit ratio 0.795, calibration bypasses 0, held 0, harmed 0 | never-attacked proxy hit ratio 0.822, held 0, harmed 0
data deleted
done
