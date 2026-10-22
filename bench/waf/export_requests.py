"""Write a benchmark's test stream as length-prefixed binary records for the WAF evaluators (modsec_eval, coraza):
per request  u32 n_fields, then n_fields x (u32 length, utf-8 bytes): method, uri, body, then header name/value pairs.

usage: python -I bench/waf/export_requests.py <srbh|csic> <out.bin>
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import http_data  # noqa: E402

name, out = sys.argv[1], sys.argv[2]
d = http_data.load(name)
with open(out, "wb") as f:
    for row, _, _ in d["test"]:
        method, uri, hdrs, body = http_data.http_parts(row)
        fields = [method, uri, body] + [x for kv in hdrs for x in kv]
        f.write(struct.pack("<I", len(fields)))
        for x in fields:
            b = x.encode("utf-8", "surrogateescape")
            f.write(struct.pack("<I", len(b)) + b)
print(len(d["test"]), "requests ->", out)
