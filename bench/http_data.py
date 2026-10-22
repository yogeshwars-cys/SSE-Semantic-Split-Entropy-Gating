"""http_data.py: the two HTTP (layer-7) datasets of the IDS benchmark, with exactly the splits the SSE harnesses use.

SR-BH 2020 (harness/replay_hybrid2.py): rows in time order; 10,000 normal 17 Jul requests drawn with
  random.Random(7).shuffle fit the gates ("fit_normal"); 18-19 Jul (384,629 requests, traffic never seen) is the
  test stream. Supervised baselines train on every labelled 17 Jul request ("train", 160,710), so they see attacks
  from the day before; SSE and the unsupervised baselines see the 10,000 normal requests only.
CSIC 2010 (harness/csic_lap.py): normal test + anomalous test interleaved with random.Random(11).shuffle (61,065
  requests); 10,000 requests sampled from normalTrafficTraining fit the gates. CSIC has no labelled attack training
  set, so supervised baselines run 2-fold cross-validation over the test stream (seeded halves, "fold"): every
  request is scored by a model that never saw it, and every method is scored on the same 61,065 requests.

Rows are dicts in the SR-BH column schema (the 11 header columns, method, request line, body), as the SSE harnesses
build them. load() caches both datasets as one pickle under data/ (written and read only by this file).

usage: from http_data import load; d = load("srbh")   # or "csic"
"""
import os
import pickle
import random
import re
import sys
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
HEADER_COLS = {
    "request_user_agent": "User-Agent", "request_referer": "Referer", "request_host": "Host",
    "request_origin": "Origin", "request_cookie": "Cookie", "request_content_type": "Content-Type",
    "request_accept": "Accept", "request_accept_language": "Accept-Language",
    "request_accept_encoding": "Accept-Encoding", "request_do_not_track": "DNT",
    "request_connection": "Connection",
}
SRBH_CSV = os.path.join(DATA, "sr-bh-2020", "data_capec_multilabel.csv")
CSIC_DIR = os.path.join(DATA, "csic-2010")


def _srbh():
    sys.argv = [sys.argv[0], SRBH_CSV, os.path.join(ROOT, "sse"), "/dev/null"]
    sys.path.insert(0, os.path.join(ROOT, "harness"))
    import replay_realtime as rr
    import run_sse_srbh as base
    rows, _ = base.load()
    rnd = random.Random(base.SEED)
    day1 = [r for r in rows if r[0] < rr.DAY2]
    replay = [r for r in rows if rr.DAY2 <= r[0] < base.ATTACK_END]
    b1 = [r for r in day1 if r[1] == rr.NORMAL]
    rnd.shuffle(b1)
    norm = lambda r: (r[2], r[1] != rr.NORMAL, [c for c in r[1] if c != rr.NORMAL[0]])  # noqa: E731
    return {"fit_normal": [r[2] for r in b1[:base.N_FIT]], "train": [norm(r) for r in day1],
            "test": [norm(r) for r in replay]}


def _csic():
    sys.argv = [sys.argv[0], "unused.csv", os.path.join(ROOT, "sse"), "/dev/null"]
    sys.path.insert(0, os.path.join(ROOT, "harness"))
    import run_sse_srbh as base
    cols = {v: k for k, v in base.HEADER_COLS.items()}

    def parse(path):                                             # csic_lap.parse
        text = open(path, encoding="latin-1").read()
        out = []
        for blk in re.split(r"\n(?=(?:GET|POST|PUT) http)", text):
            lines = blk.strip("\n").split("\n")
            if not lines or not re.match(r"(GET|POST|PUT) http", lines[0]):
                continue
            method, url = lines[0].split(" ")[:2]
            u = urllib.parse.urlparse(url)
            row = {"request_http_method": method, "request_http_request": u.path + ("?" + u.query if u.query else ""),
                   "request_body": "", "src_ip": "", **{c: "" for c in base.HEADER_COLS}}
            i = 1
            while i < len(lines) and lines[i].strip():
                k, _, v = lines[i].partition(":")
                col = cols.get(k.strip().lower())
                if col:
                    row[col] = v.strip()
                i += 1
            row["request_body"] = "\n".join(x for x in lines[i:] if x.strip()).strip()
            out.append(row)
        return out

    train = parse(os.path.join(CSIC_DIR, "normalTrafficTraining.txt"))
    normal = parse(os.path.join(CSIC_DIR, "normalTrafficTest.txt"))
    attack = parse(os.path.join(CSIC_DIR, "anomalousTrafficTest.txt"))
    rnd = random.Random(11)                                      # csic_lap.SEED, same draws in the same order
    stream = [(r, False) for r in normal] + [(r, True) for r in attack]
    rnd.shuffle(stream)
    fit = rnd.sample(train, base.N_FIT)
    fold = [i % 2 for i in range(len(stream))]
    random.Random(12).shuffle(fold)
    return {"fit_normal": fit, "train_normal": train, "test": [(r, a, ["csic_attack"] if a else []) for r, a in stream],
            "fold": fold}


def load(name):
    path = os.path.join(DATA, f"bench_{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    d = {"srbh": _srbh, "csic": _csic}[name]()
    with open(path, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
    return d


def http_parts(row):
    """(method, uri, [(header, value)], body) of a row, for the WAFs and the character models. Only the headers the
    dataset records (the 11 SR-BH columns) are sent, plus Content-Length when there is a body."""
    method = row["request_http_method"] or "GET"
    uri = row["request_http_request"] or "/"
    body = row.get("request_body") or ""
    hdrs = [(name, row[c]) for c, name in HEADER_COLS.items() if row.get(c)]
    if not any(n == "Host" for n, _ in hdrs):
        hdrs.insert(0, ("Host", "localhost"))
    if body:
        hdrs.append(("Content-Length", str(len(body.encode("utf-8", "surrogateescape")))))
    return method, uri, hdrs, body


def raw_text(row):
    """The request as one string: request line, headers, blank line, body (character models' input)."""
    method, uri, hdrs, body = http_parts(row)
    return f"{method} {uri} HTTP/1.1\n" + "".join(f"{k}: {v}\n" for k, v in hdrs) + "\n" + body
