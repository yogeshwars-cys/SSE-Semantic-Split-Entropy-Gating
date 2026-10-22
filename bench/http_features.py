"""http_features.py: a fixed numeric vector per HTTP request, for the tabular baselines (Isolation Forest, OC-SVM,
KitNET, Random Forest, XGBoost, CatBoost) on SR-BH 2020 and CSIC 2010.

It reads the same fields the SSE gates read (11 headers, @method, @path, @query, @body; run_sse_srbh.to_request)
and follows the hand-built feature sets common in the HTTP-IDS literature (lengths, character classes, entropies,
attack-token counts, argument counts):
  per field (15 x 6)   present, log length, entropy, non-alphanumeric share, digit share, upper-case share
  request (6)          parameter count, longest parameter value, path depth, path segment max length, known static
                       extension, method is GET / POST / other
  query + body (45)    log1p count of each of 25 special characters and 20 attack tokens (SQL, XSS, traversal,
                       command / code injection), case-folded, after percent-decoding
"""
import math
import os
import sys
import urllib.parse
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from http_data import HEADER_COLS  # noqa: E402

FIELDS = list(HEADER_COLS) + ["@method", "@path", "@query", "@body"]
SPECIAL = list("<>'\"();=&%|`$*{}[]\\/:-#!@,+")
TOKENS = ["select", "union", "insert", "drop", "sleep(", "benchmark(", " or ", "--", "/*", "char(", "<script", "alert(",
          "onerror", "javascript:", "../", "etc/passwd", "cmd", "exec", "eval(", "0x"]
STATIC = {"css", "js", "png", "jpg", "jpeg", "gif", "ico", "svg", "woff", "woff2", "ttf", "map", "html", "htm", "txt"}


def entropy(s):
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def fields(row):
    p = urllib.parse.urlparse(row["request_http_request"] or "")
    body = row.get("request_body") or ""
    out = {c: row.get(c) or "" for c in HEADER_COLS}
    out["@method"] = row["request_http_method"] or ""
    out["@path"] = p.path or "/"
    out["@query"] = urllib.parse.unquote_plus(p.query) if p.query else ""
    out["@body"] = urllib.parse.unquote_plus(body) if body else ""
    return out, p, body


def vector(row):
    f, p, body = fields(row)
    v = []
    for k in FIELDS:
        s = f[k]
        n = len(s) or 1
        v += [1.0 if s else 0.0, math.log1p(len(s)), entropy(s), sum(not c.isalnum() for c in s) / n,
              sum(c.isdigit() for c in s) / n, sum(c.isupper() for c in s) / n]
    params = urllib.parse.parse_qsl(p.query, keep_blank_values=True)
    if body and "application/x-www-form-urlencoded" in f["request_content_type"]:
        params += urllib.parse.parse_qsl(body, keep_blank_values=True)
    segs = [x for x in (p.path or "/").split("/") if x]
    ext = segs[-1].rsplit(".", 1)[1].lower() if segs and "." in segs[-1] else ""
    m = f["@method"].upper()
    v += [math.log1p(len(params)), math.log1p(max([len(x) for _, x in params] + [0])), float(len(segs)),
          math.log1p(max([len(x) for x in segs] + [0])), float(ext in STATIC), float(m == "GET"), float(m == "POST"),
          float(m not in ("GET", "POST"))]
    text = (f["@query"] + "\n" + f["@body"] + "\n" + urllib.parse.unquote_plus(p.path or "")).lower()
    v += [math.log1p(text.count(c)) for c in SPECIAL]
    v += [math.log1p(text.count(t)) for t in TOKENS]
    return v


def matrix(rows):
    return np.array([vector(r) for r in rows], dtype=np.float32)


NAMES = [f"{k}:{x}" for k in FIELDS for x in ("present", "loglen", "entropy", "nonalnum", "digit", "upper")] + \
    ["n_params", "max_param_len", "path_depth", "max_seg_len", "static_ext", "get", "post", "other_method"] + \
    [f"char:{c}" for c in SPECIAL] + [f"tok:{t}" for t in TOKENS]
