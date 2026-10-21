# Offline detection: statistical gate, semantic layer behind it, cascade (semantic_test.py)

| test set | n | gate flags | semantic flags (of gate-passed) | cascade flags |
|---|---|---|---|---|
| normal held-out (false alarms) | 4000 | 1.1% | 0.9% | 2.0% |
| mimic: real UA + ' evil.example' (harness payload) | 1000 | 26.8% | 100.0% | 100.0% |
| mimic: real UA + '"><script src=//evil.example/a.js></script>' | 1000 | 91.6% | 100.0% | 100.0% |
| mimic: Accept-Language + ' evil.example' | 1000 | 41.2% | 100.0% | 100.0% |
| mimic: Referer pointing at https://evil.example/x | 1000 | 100.0% | 0.0% | 100.0% |
| rare but valid real UA (legitimate, want LOW) | 1000 | 9.1% | 4.3% | 13.0% |
| oversize Cookie 'A' x 500 | 1000 | 100.0% | 0.0% | 100.0% |
| utm_content=evil (no header change) | 1000 | 1.1% | 0.9% | 2.0% |

semantic score: 0.367 ms per request (3114 baseline values over 7 headers)
