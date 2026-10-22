"""Per-row digest of what the gates read (fields, lengths, entropies, 3-gram counts, parameter names, route), the
same serialisation as `replay_srbh --features`, so the C++ request parsing can be diffed against Python's.

usage: python -I cpp/parity_features.py <data_capec_multilabel.csv> <out.txt>
"""
import os
import sys
import urllib.parse
import zlib

CSV, OUT = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], CSV, os.path.join(HERE, "..", "sse"), "/dev/null"]
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
import run_sse_srbh as base  # noqa: E402
from anomaly_model import shannon  # noqa: E402
from semantic_model import gram_counts  # noqa: E402

rows, _ = base.load()
with open(OUT, "w") as f:
    for ts, _, row in rows:
        h, params = base.to_request(row, True)
        s = ""
        for k, v in h.items():
            idx, cnt = gram_counts(v)
            s += k + "\x1e" + v + "\x1e%d\x1e%.17g\x1e" % (len(v), shannon(v))
            s += "".join(f"{i}:{c}," for i, c in zip(idx, cnt)) + "\x1d"
        s += "\x1c" + "".join(p + "\x1d" for p in params)
        s += "\x1c" + (urllib.parse.urlparse(row["request_http_request"] or "/").path or "/")
        f.write(f"{int(ts)} {zlib.crc32(s.encode())}\n")
print(len(rows), "rows")
