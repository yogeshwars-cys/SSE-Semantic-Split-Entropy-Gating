
## Adaptive adversary vs gate + semantic + split (no feedback), 5 min

  first confirmed bypass after 0.1 s: [referer/length-matched] 'https://szf-hwv.us/9mqdwsrzowpmk8'
  minute 1: victims harmed 0.0%  held 14.6%  | attacker trials this minute 8595/8819 passed
  minute 2: victims harmed 0.0%  held 16.2%  | attacker trials this minute 9251/9446 passed
  minute 3: victims harmed 0.0%  held 15.3%  | attacker trials this minute 8400/8560 passed
  minute 4: victims harmed 0.0%  held 15.9%  | attacker trials this minute 8555/8695 passed
  minute 5: victims harmed 0.0%  held 14.4%  | attacker trials this minute 7906/8057 passed

victim views 5532: harmed 0 (0.0%), held 844 (15.3%), clean 4688 (84.7%)
proxy: misses 45451, hits 70455; fills blocked by gate 597, semantic 418, calibration 0; entries flipped by split quorum 368, confirmed poison fed back 0, blames 0 (fallback 0), threshold tightenings 0
attacker: 42708 of 43578 live trials passed; first bypass at 0.1 s; 23218 exploit requests

| header | live trials | passed | pass rate |
|---|---|---|---|
| user-agent | 123 | 113 | 91.9% |
| accept-language | 0 | 0 | 0.0% |
| referer | 2357 | 2271 | 96.4% |
| cookie | 11767 | 11516 | 97.9% |
| accept | 0 | 0 | 0.0% |
| x-forwarded-for | 29331 | 28808 | 98.2% |

| minute | victim views | harmed % | held % | clean % | attacker trials passed |
|---|---|---|---|---|---|
| 1 | 1132 | 0.0% | 14.6% | 85.4% | 8595/8819 (97.5%) |
| 2 | 1108 | 0.0% | 16.2% | 83.8% | 9251/9446 (97.9%) |
| 3 | 1100 | 0.0% | 15.3% | 84.7% | 8400/8560 (98.1%) |
| 4 | 1099 | 0.0% | 15.9% | 84.1% | 8555/8695 (98.4%) |
| 5 | 1093 | 0.0% | 14.4% | 85.6% | 7906/8057 (98.1%) |
| 6 | 1 | 0.0% | 0.0% | 0.0% | 1/1 (100.0%) |

## Adaptive adversary vs gate + semantic + split + calibration (split verdicts tune both gates), 5 min

  first confirmed bypass after 0.1 s: [referer/length-matched] 'https://szf-hwv.us/9mqdwsrzowpmk8'
  minute 1: victims harmed 0.0%  held 3.5%  | attacker trials this minute 357/2122 passed
  minute 2: victims harmed 0.0%  held 2.1%  | attacker trials this minute 159/2445 passed
  minute 3: victims harmed 0.0%  held 2.7%  | attacker trials this minute 554/4645 passed
  minute 4: victims harmed 0.0%  held 3.2%  | attacker trials this minute 326/4081 passed
  minute 5: victims harmed 0.0%  held 3.1%  | attacker trials this minute 45/3896 passed

victim views 5788: harmed 0 (0.0%), held 169 (2.9%), clean 5619 (97.1%)
proxy: misses 37751, hits 24484; fills blocked by gate 638, semantic 461, calibration 19049; entries flipped by split quorum 73, confirmed poison fed back 73, blames 73 (fallback 0), threshold tightenings 146
attacker: 1441 of 17189 live trials passed; first bypass at 0.1 s; 22069 exploit requests

| header | live trials | passed | pass rate |
|---|---|---|---|
| user-agent | 181 | 8 | 4.4% |
| accept-language | 0 | 0 | 0.0% |
| referer | 498 | 8 | 1.6% |
| cookie | 2731 | 183 | 6.7% |
| accept | 0 | 0 | 0.0% |
| x-forwarded-for | 13779 | 1242 | 9.0% |

| minute | victim views | harmed % | held % | clean % | attacker trials passed |
|---|---|---|---|---|---|
| 1 | 1156 | 0.0% | 3.5% | 96.5% | 357/2122 (16.8%) |
| 2 | 1160 | 0.0% | 2.1% | 97.9% | 159/2445 (6.5%) |
| 3 | 1164 | 0.0% | 2.7% | 97.3% | 554/4645 (11.9%) |
| 4 | 1152 | 0.0% | 3.2% | 96.8% | 326/4081 (8.0%) |
| 5 | 1156 | 0.0% | 3.1% | 96.9% | 45/3896 (1.2%) |

calibrated thresholds at the end (route | layer | name -> threshold; global gate and semantic thresholds unchanged):
- `/refl/cookie|gate|cookie` -> -1.74
- `/refl/cookie|sem|cookie` -> -0.25
- `/refl/referer|gate|referer` -> -1.18
- `/refl/referer|sem|referer` -> -1.72
- `/refl/user-agent|gate|user-agent` -> -1.73
- `/refl/user-agent|sem|user-agent` -> 1.42
- `/refl/x-forwarded-for|gate|x-forwarded-for` -> 0.2
- `/refl/x-forwarded-for|sem|x-forwarded-for` -> -0.19

benign traffic right after the attack (20,000 requests on other pages, same proxy): hit ratio 0.801 (0.932 before any attack in combined_semantic.md), calibration bypasses 0, harmed 20000
data deleted
done
