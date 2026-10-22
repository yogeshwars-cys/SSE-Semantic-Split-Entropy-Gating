"""flow_prepare.py: network-flow and packet datasets as token streams for the C++ SSE engine (cpp/replay_srbh
--prepared), which reads HTTP-shaped records. Nothing in the engine changes; each record is written as one request:

  route (@path)      CIC-IDS2017: /p<destination port> for ports carrying >= 1% of the fit flows, /wk (other port
                     < 1024) or /eph; UNSW-NB15: /<service> (/<proto> when the service is "-"); Kitsune: /pkt
  @method            CIC: FLOW; UNSW: the transport protocol; Kitsune: PKT
  five feature groups in five field slots (a split like SSE's: what the sender does vs how the other side answers)
                     each value is "tag:code" tokens, one per feature: tag = two letters naming the feature,
                     code = p / m (sign) + two letters for round(2 log2(1 + |x|)), i.e. half-bit log buckets.
                     Letters only, because the semantic layer folds digits to 0.
  UNSW also gives its connection state as a word in a sixth slot.
Slots are the engine's header fields (user-agent, referer, cookie, accept, @body, accept-language); field names in
results are therefore those slot names (FIELD_MAP below says what each one holds).

Next to each stream, <name>.npz holds the same rows as raw numbers for the tabular baselines (fit, test, y, cls;
UNSW also train / ytrain, with proto / service / state one-hot).

Splits (fit = normal records the gates fit on; precal = labelled stream learned from before the test laps):
  cic      per capture file, ids/benchmark_sse_cic.load_cic_csv (the published SSE-IDS split): fit = 10,000 of its
           calm benign training flows, evenly spaced; test = its test stream (<= 120,000 flows, time order)
  unsw     official partition: fit = 10,000 normal flows of the training set (seeded); precal = the training set
           (175,341 flows, seeded order); test = the testing set (82,332 flows, seeded order: the files are grouped)
  kitsune  per capture: fit = 10,000 packets evenly spaced over the benign prefix before the first attack packet;
           test = every packet after that prefix, evenly thinned to <= 200,000 (packet order kept)

usage: python -I bench/flow_prepare.py <cic|unsw|kitsune> <out dir>
"""
import csv
import glob
import gzip
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
SLOTS = ["request_user_agent", "request_referer", "request_cookie", "request_accept", "request_body"]
COLS = ["split", "label", "class", "request_http_method", "request_http_request", "request_body",
        "request_user_agent", "request_referer", "request_host", "request_origin", "request_cookie",
        "request_content_type", "request_accept", "request_accept_language", "request_accept_encoding",
        "request_do_not_track", "request_connection"]
L2 = [chr(97 + i // 26) + chr(97 + i % 26) for i in range(676)]

# CIC-IDS2017 MachineLearningCSV columns (index after the header's stripped names)
CIC_GROUPS = {
    "request_user_agent": [2, 4, 6, 7, 8, 9, 34, 53, 62, 63, 68],                 # forward sizes
    "request_referer": [30, 32, 43, 44, 45, 46, 47, 48, 49, 50, 66, 69],          # flags, TCP stack (forward)
    "request_cookie": [3, 5, 10, 11, 12, 13, 35, 54, 64, 65, 67],                 # backward sizes and window
    "request_accept": [1, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 36, 37, 70, 74],  # timing
    "request_body": [38, 39, 40, 41, 51, 52],                                     # whole-flow shape
}
UNSW_GROUPS = {
    "request_user_agent": ["spkts", "sbytes", "smean", "sttl", "sload", "sloss", "trans_depth"],
    "request_referer": ["swin", "stcpb", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "is_sm_ips_ports"],
    "request_cookie": ["dpkts", "dbytes", "dmean", "dttl", "dload", "dloss", "dwin", "dtcpb", "response_body_len"],
    "request_accept": ["dur", "rate", "sinpkt", "dinpkt", "sjit", "djit", "tcprtt", "synack", "ackdat"],
    "request_body": ["ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
                     "ct_dst_src_ltm", "ct_src_ltm", "ct_srv_dst"],
}
# Kitsune (AfterImage) 115 features: per time window 5 x (MI_dir 3, H 3, HH 7, HH_jit 3, HpHp 7), stored by family
KIT_GROUPS = {"request_user_agent": range(0, 15), "request_referer": range(15, 30), "request_cookie": range(30, 65),
              "request_accept": range(65, 80), "request_body": range(80, 115)}
FIELD_MAP = {"cic": {"user-agent": "forward sizes", "referer": "flags + forward TCP stack",
                     "cookie": "backward sizes + window", "accept": "timing", "@body": "whole-flow shape"},
             "unsw": {"user-agent": "source sizes/ttl/load", "referer": "source TCP + app flags",
                      "cookie": "destination sizes/ttl/load", "accept": "timing", "@body": "connection counts",
                      "accept-language": "state"},
             "kitsune": {"user-agent": "MI_dir (src MAC-IP)", "referer": "H (src IP)", "cookie": "HH (channel)",
                         "accept": "HH_jit (channel jitter)", "@body": "HpHp (socket)"}}


def encode(X, groups):
    """(n, d) array -> per slot a list of token strings."""
    X = np.nan_to_num(np.asarray(X, np.float64), nan=0.0, posinf=1e12, neginf=-1e12)
    B = np.clip(np.round(2 * np.log2(1 + np.abs(X))), 0, 675).astype(np.int64)
    S = X < 0
    out = {}
    for slot, cols in groups.items():
        cols = list(cols)
        tags = [L2[c % 676] for c in cols]
        out[slot] = [" ".join(f"{t}:{'m' if S[r, c] else 'p'}{L2[B[r, c]]}" for t, c in zip(tags, cols))
                     for r in range(len(X))]
    return out


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in rows:
            w.writerow([r.get(c, "") for c in COLS])


def records(split, labels, classes, method, route, enc, extra=None):
    for i in range(len(labels)):
        r = {"split": split, "label": int(labels[i]), "class": classes[i] if labels[i] else "",
             "request_http_method": method[i], "request_http_request": route[i]}
        for slot, vals in enc.items():
            r[slot] = vals[i]
        if extra:
            for k, v in extra.items():
                r[k] = v[i]
        yield r


def cic(out):
    sys.path.insert(0, os.path.join(ROOT, "ids"))
    import pathlib
    from benchmark_sse_cic import FILES, load_cic_csv
    for fname in FILES:
        Xtr, Xte, lab = load_cic_csv(pathlib.Path(DATA) / "cic-ids2017" / fname)
        fit = Xtr[np.linspace(0, len(Xtr) - 1, min(10000, len(Xtr))).astype(int)]
        ports = fit[:, 0].astype(int)
        u, c = np.unique(ports, return_counts=True)
        major = set(u[c >= 0.01 * len(ports)].tolist())

        def route(X):
            return [f"/p{p}" if p in major else "/wk" if p < 1024 else "/eph" for p in X[:, 0].astype(int)]
        y = np.array([x != "BENIGN" for x in lab])
        rows = list(records("fit", np.zeros(len(fit), int), [""] * len(fit), ["FLOW"] * len(fit), route(fit),
                            encode(fit, CIC_GROUPS)))
        rows += list(records("test", y.astype(int), lab, ["FLOW"] * len(Xte), route(Xte), encode(Xte, CIC_GROUPS)))
        short = fname.split(".")[0].replace("-WorkingHours", "").replace("-workingHours", "")
        write(os.path.join(out, f"cic_{short}.csv"), rows)
        np.savez(os.path.join(out, f"cic_{short}.npz"), fit=fit, test=Xte, y=y, cls=np.array(lab))
        print(short, len(fit), "fit,", len(Xte), "test,", int(y.sum()), "attacks; major ports", sorted(major), flush=True)


def unsw(out):
    import pandas as pd
    tr = pd.read_csv(os.path.join(DATA, "unsw-nb15", "test.csv"))     # Mireu-Lab's test.csv = official training set
    te = pd.read_csv(os.path.join(DATA, "unsw-nb15", "train.csv"))    # and train.csv = official testing set
    rnd = random.Random(15)
    tr = tr.iloc[rnd.sample(range(len(tr)), len(tr))].reset_index(drop=True)
    te = te.iloc[rnd.sample(range(len(te)), len(te))].reset_index(drop=True)
    norm = tr[tr["label"] == 0]
    fit = norm.iloc[sorted(rnd.sample(range(len(norm)), 10000))]
    names, groups, pos = [], {}, 0                               # UNSW_GROUPS as column positions of X
    for s, cols in UNSW_GROUPS.items():
        names += cols
        groups[s] = range(pos, pos + len(cols))
        pos += len(cols)

    def recs(split, d):
        enc = encode(d[names].to_numpy(), groups)
        route = [f"/{s}" if s != "-" else f"/{p}" for s, p in zip(d["service"], d["proto"])]
        state = {"request_accept_language": [str(x).lower() for x in d["state"]]}
        return records(split, d["label"].to_numpy(), d["attack_cat"].tolist(), d["proto"].tolist(), route, enc, state)
    rows = list(recs("fit", fit)) + list(recs("precal", tr)) + list(recs("test", te))
    write(os.path.join(out, "unsw.csv"), rows)
    num = [c for c in tr.columns if c not in ("id", "proto", "service", "state", "attack_cat", "label")]
    cats = {c: tr[c].value_counts().index[:30].tolist() for c in ("proto", "service", "state")}

    def mat(d):                                                  # numeric columns + one-hot of the training set's
        oh = [(d[c] == v).to_numpy(np.float32)[:, None] for c, vs in cats.items() for v in vs]   # top categories
        return np.hstack([d[num].to_numpy(np.float32)] + oh)
    np.savez(os.path.join(out, "unsw.npz"), fit=mat(fit), train=mat(tr), ytrain=tr["label"].to_numpy().astype(bool),
             test=mat(te), y=te["label"].to_numpy().astype(bool), cls=te["attack_cat"].to_numpy().astype(str))
    print("unsw", len(fit), "fit,", len(tr), "precal,", len(te), "test,", int(te["label"].sum()), "test attacks")


def kitsune(out, max_test=200000):
    import pandas as pd
    for d in sorted(glob.glob(os.path.join(DATA, "kitsune", "*"))):
        name = os.path.basename(d)
        ds = glob.glob(os.path.join(d, "*_dataset.csv.gz"))
        lb = glob.glob(os.path.join(d, "*_labels.csv.gz"))
        if not ds or not lb:
            continue
        with gzip.open(lb[0], "rt") as f:
            first = f.readline()
        labels = pd.read_csv(lb[0], header=0 if "x" in first else None)
        y = labels.iloc[:, -1].to_numpy().astype(int)
        X = pd.read_csv(ds[0], header=None, dtype=np.float32, engine="c").to_numpy()
        if X.shape[1] == 116:
            X = X[:, 1:]                                         # leading row index
        n = min(len(X), len(y))
        X, y = X[:n], y[:n]
        a0 = int(np.argmax(y > 0)) if y.any() else n
        fit = X[np.linspace(0, a0 - 1, min(10000, a0)).astype(int)]
        idx = np.arange(a0, n)
        if len(idx) > max_test:
            idx = idx[np.linspace(0, len(idx) - 1, max_test).astype(int)]
        cls = name
        rows = list(records("fit", np.zeros(len(fit), int), [""] * len(fit), ["PKT"] * len(fit), ["/pkt"] * len(fit),
                            encode(fit, KIT_GROUPS)))
        rows += list(records("test", y[idx], [cls] * len(idx), ["PKT"] * len(idx), ["/pkt"] * len(idx),
                             encode(X[idx], KIT_GROUPS)))
        write(os.path.join(out, f"kitsune_{name}.csv"), rows)
        np.savez(os.path.join(out, f"kitsune_{name}.npz"), fit=fit, test=X[idx], y=y[idx].astype(bool),
                 cls=np.array([cls if v else "" for v in y[idx]]))
        print(name, n, "packets; benign prefix", a0, "; test", len(idx), "with", int(y[idx].sum()), "attacks", flush=True)
        del X


if __name__ == "__main__":
    which, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    {"cic": cic, "unsw": unsw, "kitsune": kitsune}[which](out)
