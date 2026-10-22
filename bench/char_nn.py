"""char_nn.py: supervised deep-learning WAFs on the raw request text (SR-BH 2020, CSIC 2010).

  char_cnn     byte embedding 32 -> 1-D convolutions (128 filters each, widths 3 / 5 / 7) -> global max pool ->
               dropout 0.3 -> dense 128 -> sigmoid; first 512 bytes; 3 epochs, Adam 1e-3, batch 256
  char_bilstm  byte embedding 32 -> bidirectional LSTM (64 per direction) -> max pool over time -> dense 64 ->
               sigmoid; first 256 bytes; 2 epochs, Adam 1e-3, batch 256
Input text: request line, body, then headers (so a truncated input keeps the parts attacks live in).
Training: SR-BH every labelled 17 Jul request (tested on 18-19 Jul); CSIC 2-fold cross-validation over the test
stream (the halves tabular_baselines.py uses). Verdict: probability >= 0.5.

usage: python -I bench/char_nn.py <srbh|csic> [char_cnn|char_bilstm ...]
"""
import io
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import http_data  # noqa: E402

NAME = sys.argv[1]
ONLY = set(sys.argv[2:])
THREADS = int(os.environ.get("BENCH_THREADS", 2))
torch.set_num_threads(THREADS)


def text(row):
    method, uri, hdrs, body = http_data.http_parts(row)
    return f"{method} {uri}\n{body}\n" + "".join(f"{k}: {v}\n" for k, v in hdrs)


def encode(rows, L):
    X = np.zeros((len(rows), L), np.uint8)                       # byte + 1, 0 = padding (bytes 255 clipped)
    for i, r in enumerate(rows):
        b = np.frombuffer(text(r).encode("utf-8", "surrogateescape")[:L], np.uint8)
        X[i, :len(b)] = np.minimum(b.astype(np.int32) + 1, 255)
    return X


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(256, 32, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(32, 128, k, padding=k // 2) for k in (3, 5, 7)])
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        return self.head(torch.cat([torch.relu(c(e)).amax(dim=2) for c in self.convs], 1)).squeeze(1)


class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(256, 32, padding_idx=0)
        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return self.head(h.amax(dim=1)).squeeze(1)


def train(make, X, y, epochs):
    torch.manual_seed(0)
    m = make()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        m.train()
        idx = rng.permutation(len(X))
        t, tot = time.perf_counter(), 0.0
        for s in range(0, len(idx), 256):
            b = idx[s:s + 256]
            opt.zero_grad()
            loss = lossf(m(torch.from_numpy(X[b]).long()), torch.from_numpy(y[b].astype(np.float32)))
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        print(f"    epoch {ep + 1}: loss {tot / len(X):.4f} ({time.perf_counter() - t:.0f}s)", flush=True)
    return m


@torch.no_grad()
def predict(m, X):
    m.eval()
    return np.concatenate([torch.sigmoid(m(torch.from_numpy(X[s:s + 2048]).long())).numpy()
                           for s in range(0, len(X), 2048)])


def main():
    d = http_data.load(NAME)
    y = np.array([a for _, a, _ in d["test"]])
    cls = [c for _, _, c in d["test"]]
    res_path = f"{NAME}/charnn.json"
    prev = os.path.join(common.RESULTS, res_path)
    out = json.load(open(prev)) if os.path.exists(prev) else {}
    out.update({"dataset": NAME, "test_records": int(len(y)), "threads": THREADS})
    for name, make, L, epochs in (("char_cnn", CNN, 512, 3), ("char_bilstm", BiLSTM, 256, 2)):
        if ONLY and name not in ONLY:
            continue
        print(f"== {NAME} {name}", flush=True)
        Xt = encode([r for r, _, _ in d["test"]], L)
        t = common.Timer()
        if "train" in d:
            m = train(make, encode([r for r, _, _ in d["train"]], L), np.array([a for _, a, _ in d["train"]]), epochs)
            p, how = predict(m, Xt), "trained on every labelled 17 Jul request"
        else:
            fold = np.array([i % 2 for i in range(len(y))])
            np.random.default_rng(12).shuffle(fold)
            p = np.zeros(len(y))
            for k in (0, 1):
                m = train(make, Xt[fold != k], y[fold != k], epochs)
                p[fold == k] = predict(m, Xt[fold == k])
            how = "2-fold cross-validation over the test stream"
        secs = t()
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        tb = time.perf_counter()
        predict(m, Xt[:20000])
        batch = (time.perf_counter() - tb) / min(20000, len(Xt)) * 1e6
        tb = time.perf_counter()
        for i in range(300):
            predict(m, Xt[i:i + 1])
        single = (time.perf_counter() - tb) / 300 * 1e6
        out[name] = common.report(y, p, p >= 0.5, cls, train_seconds=secs, protocol=how, model_bytes=buf.tell(),
                                  us_per_record_batch=batch, us_per_record_single=single)
        print(common.line(f"{NAME} {name}", out[name]), flush=True)
        common.save(res_path, out)


if __name__ == "__main__":
    main()
