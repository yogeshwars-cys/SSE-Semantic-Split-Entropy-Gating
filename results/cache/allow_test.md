# Split-delivery caching on real open datasets (Docker). n=30000

## ALLOW: allowlisting (strip unknown request headers) against the split-delivery proxy, in cases where it is expected to fail

### oversize Cookie (header-oversize via a standard header)

| design | victim views | held | HARMED | harmed % |
|---|---|---|---|---|
| unprotected cache | 240 | 0 | 180 | 75% |
| allowlist only | 240 | 0 | 180 | 75% |
| allowlist + no-cache-errors | 240 | 0 | 0 | 0% |
| full-screen origin verifier | 240 | 0 | 0 | 0% |
| split-delivery (annotated) | 228 | 53 | 0 | 0% |

### oversize Referer (header-oversize via a standard header)

| design | victim views | held | HARMED | harmed % |
|---|---|---|---|---|
| unprotected cache | 240 | 0 | 180 | 75% |
| allowlist only | 240 | 0 | 180 | 75% |
| allowlist + no-cache-errors | 240 | 0 | 0 | 0% |
| full-screen origin verifier | 240 | 0 | 0 | 0% |
| split-delivery (annotated) | 223 | 79 | 0 | 0% |

### reflection via User-Agent (standard header)

| design | victim views | held | HARMED | harmed % |
|---|---|---|---|---|
| unprotected cache | 240 | 0 | 180 | 75% |
| allowlist only | 240 | 0 | 180 | 75% |
| allowlist + no-cache-errors | 240 | 0 | 180 | 75% |
| full-screen origin verifier | 212 | 0 | 126 | 59% |
| split-delivery (annotated) | 217 | 100 | 0 | 0% |

### reflection via Cookie (standard header)

| design | victim views | held | HARMED | harmed % |
|---|---|---|---|---|
| unprotected cache | 240 | 0 | 180 | 75% |
| allowlist only | 240 | 0 | 240 | 100% |
| allowlist + no-cache-errors | 240 | 0 | 180 | 75% |
| full-screen origin verifier | 200 | 0 | 195 | 98% |
| split-delivery (annotated) | 217 | 102 | 0 | 0% |

### unkeyed query parameter (utm_content), no header at all

| design | victim views | held | HARMED | harmed % |
|---|---|---|---|---|
| unprotected cache | 240 | 0 | 180 | 75% |
| allowlist only | 240 | 0 | 180 | 75% |
| allowlist + no-cache-errors | 240 | 0 | 180 | 75% |
| full-screen origin verifier | n/a | n/a | n/a | n/a |
| split-delivery (annotated) | 208 | 151 | 0 | 0% |

### Headers a real application legitimately reads: wrong responses out of 20 requests that carry the header

| header | unprotected cache | allowlist (standard headers only) |
|---|---|---|
| x-tenant | 0 / 20 | 20 / 20 |
| x-forwarded-host | 0 / 20 | 20 / 20 |
| x-forwarded-proto | 0 / 20 | 20 / 20 |
| authorization | 0 / 20 | 20 / 20 |
| x-api-key | 0 / 20 | 20 / 20 |
| x-requested-with | 0 / 20 | 20 / 20 |
| x-api-version | 0 / 20 | 20 / 20 |
| traceparent | 0 / 20 | 20 / 20 |
| origin | 0 / 20 | 20 / 20 |
| x-forwarded-for | 0 / 20 | 20 / 20 |
| x-real-ip | 0 / 20 | 20 / 20 |
| accept-language | 0 / 20 | 0 / 20 |

done
