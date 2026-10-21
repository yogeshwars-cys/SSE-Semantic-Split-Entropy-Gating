"""prep_nq_100k.py: build a 100k-passage NQ slice from the BEIR NQ archive (stands in for bqboost's prep script).

Keeps every passage relevant to a test query, then fills up to 100,000 with a seeded uniform sample of the other
2.68M passages. BEIR-shaped output (corpus.jsonl, queries.jsonl, qrels/test.tsv), so the SciFact loaders read it.
Not the identical slice behind sse_nq100k_results.json (that one sampled from HaDA's 1M MiniLM set), so absolute
scores differ a little; every comparison made on this slice is like for like.

usage: python -I prep_nq_100k.py <nq.zip> <out dir>
"""
import json
import os
import sys
import zipfile

import numpy as np

ZIP, OUT = sys.argv[1], sys.argv[2]
N, SEED = 100_000, 20261021


def main():
    z = zipfile.ZipFile(ZIP)
    qrels = z.read("nq/qrels/test.tsv").decode("utf-8")
    rel = {line.split("\t")[1] for line in qrels.splitlines()[1:]}
    ids = []
    with z.open("nq/corpus.jsonl") as f:                      # pass 1: ids only (cheap prefix parse)
        for line in f:
            ids.append(line[line.index(b'"_id": "') + 8:].split(b'"', 1)[0].decode())
    others = [i for i, d in enumerate(ids) if d not in rel]
    keep = {i for i, d in enumerate(ids) if d in rel}
    rng = np.random.default_rng(SEED)
    keep.update(int(i) for i in rng.choice(others, N - len(keep), replace=False))
    os.makedirs(os.path.join(OUT, "qrels"), exist_ok=True)
    with z.open("nq/corpus.jsonl") as f, open(os.path.join(OUT, "corpus.jsonl"), "wb") as g:   # pass 2: copy kept rows
        for i, line in enumerate(f):
            if i in keep:
                g.write(line)
    with open(os.path.join(OUT, "queries.jsonl"), "wb") as g:
        g.write(z.read("nq/queries.jsonl"))
    with open(os.path.join(OUT, "qrels", "test.tsv"), "w", encoding="utf-8") as g:
        g.write(qrels)
    print(f"{len(ids):,} passages in NQ, {len(rel):,} test-relevant, wrote {len(keep):,} to {OUT}")


if __name__ == "__main__":
    main()
