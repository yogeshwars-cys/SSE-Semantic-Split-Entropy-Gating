# Statistical front gate + split-delivery, real Docker run

## Attack: unknown header names (5 sampled from the Param Miner list)  (10s, attacker hammering, 4 victims)

| design | victim views | clean | held | HARMED | harmed % | fills blocked by gate | fills blocked by semantic |
|---|---|---|---|---|---|---|---|
| unprotected cache | 200 | 0 | 0 | 200 | 100% | 0 | 0 |
| gate only | 196 | 196 | 0 | 0 | 0% | 77 | 0 |
| split only | 181 | 98 | 83 | 0 | 0% | 0 | 0 |
| gate + split | 196 | 196 | 0 | 0 | 0% | 74 | 0 |
| gate + semantic | 196 | 196 | 0 | 0 | 0% | 77 | 0 |
| gate + semantic + split | 196 | 196 | 0 | 0 | 0% | 79 | 1 |

## Attack: X-Forwarded-Host with a hostname  (10s, attacker hammering, 4 victims)

| design | victim views | clean | held | HARMED | harmed % | fills blocked by gate | fills blocked by semantic |
|---|---|---|---|---|---|---|---|
| unprotected cache | 200 | 60 | 0 | 140 | 70% | 0 | 0 |
| gate only | 200 | 200 | 0 | 0 | 0% | 11 | 0 |
| split only | 176 | 60 | 116 | 0 | 0% | 0 | 0 |
| gate + split | 200 | 200 | 0 | 0 | 0% | 16 | 0 |
| gate + semantic | 200 | 200 | 0 | 0 | 0% | 16 | 0 |
| gate + semantic + split | 200 | 200 | 0 | 0 | 0% | 17 | 0 |

## Attack: oversize Cookie (header-oversize through a standard header)  (10s, attacker hammering, 4 victims)

| design | victim views | clean | held | HARMED | harmed % | fills blocked by gate | fills blocked by semantic |
|---|---|---|---|---|---|---|---|
| unprotected cache | 200 | 60 | 0 | 140 | 70% | 0 | 0 |
| gate only | 200 | 200 | 0 | 0 | 0% | 5 | 0 |
| split only | 186 | 118 | 68 | 0 | 0% | 0 | 0 |
| gate + split | 200 | 200 | 0 | 0 | 0% | 4 | 0 |
| gate + semantic | 200 | 200 | 0 | 0 | 0% | 5 | 0 |
| gate + semantic + split | 200 | 200 | 0 | 0 | 0% | 7 | 0 |

## Attack: mimic: real User-Agent carrying attacker content  (10s, attacker hammering, 4 victims)

| design | victim views | clean | held | HARMED | harmed % | fills blocked by gate | fills blocked by semantic |
|---|---|---|---|---|---|---|---|
| unprotected cache | 200 | 0 | 0 | 200 | 100% | 0 | 0 |
| gate only | 200 | 1 | 0 | 199 | 100% | 2 | 0 |
| split only | 176 | 59 | 117 | 0 | 0% | 0 | 0 |
| gate + split | 176 | 61 | 115 | 0 | 0% | 4 | 0 |
| gate + semantic | 200 | 200 | 0 | 0 | 0% | 12 | 10 |
| gate + semantic + split | 200 | 200 | 0 | 0 | 0% | 10 | 10 |

## Attack: unkeyed query parameter utm_content=evil (no header change)  (10s, attacker hammering, 4 victims)

| design | victim views | clean | held | HARMED | harmed % | fills blocked by gate | fills blocked by semantic |
|---|---|---|---|---|---|---|---|
| unprotected cache | 200 | 60 | 0 | 140 | 70% | 0 | 0 |
| gate only | 200 | 0 | 0 | 200 | 100% | 0 | 0 |
| split only | 188 | 131 | 57 | 0 | 0% | 0 | 0 |
| gate + split | 187 | 119 | 68 | 0 | 0% | 0 | 0 |
| gate + semantic | 200 | 0 | 0 | 200 | 100% | 0 | 0 |
| gate + semantic + split | 176 | 57 | 119 | 0 | 0% | 0 | 0 |

## Attack: rare but valid real User-Agent (legitimate variant)  (10s, attacker hammering, 4 victims)

| design | victim views | clean | held | HARMED | harmed % | fills blocked by gate | fills blocked by semantic |
|---|---|---|---|---|---|---|---|
| unprotected cache | 200 | 0 | 0 | 200 | 100% | 0 | 0 |
| gate only | 200 | 2 | 0 | 198 | 99% | 3 | 0 |
| split only | 184 | 118 | 66 | 0 | 0% | 0 | 0 |
| gate + split | 176 | 66 | 110 | 0 | 0% | 4 | 0 |
| gate + semantic | 200 | 2 | 0 | 198 | 99% | 2 | 0 |
| gate + semantic + split | 176 | 61 | 115 | 0 | 0% | 3 | 0 |

## Benign traffic (20,000 realistic requests): what each layer costs normal users

| design | requests | req/s | hit ratio | p50 ms | p99 ms | origin reqs | gate bypasses | semantic bypasses | share of misses bypassed | held (false) | harmed (false) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unprotected cache | 20000 | 1801 | 0.933 | 3.9 | 12.6 | 1345 | 0 | 0 | 0.0% | 0 | 0 |
| gate only | 20000 | 1894 | 0.932 | 3.7 | 11.9 | 1351 | 14 | 0 | 1.0% | 0 | 0 |
| split only | 20000 | 1840 | 0.933 | 3.5 | 11.3 | 1348 | 0 | 0 | 0.0% | 0 | 0 |
| gate + split | 20000 | 1716 | 0.932 | 3.8 | 13.2 | 1352 | 14 | 0 | 1.0% | 0 | 0 |
| gate + semantic | 20000 | 1581 | 0.932 | 4.1 | 19.5 | 1361 | 14 | 11 | 1.8% | 0 | 0 |
| gate + semantic + split | 20000 | 1523 | 0.932 | 4.0 | 21.5 | 1361 | 14 | 11 | 1.8% | 0 | 0 |

done
