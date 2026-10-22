"""tabular_baselines.py: the tabular baselines on every IDS dataset, scored on the same test rows as SSE.

Inputs: HTTP datasets (srbh, csic) as http_features vectors; flow / packet datasets (cic_*, unsw, kitsune_*) as
their raw numeric columns (bench/flow_prepare.py npz, rows aligned with the SSE streams).

  unsupervised, normal traffic only (like SSE): fit on the first 5,000 of the dataset's 10,000 normal fit records,
      threshold at the 99th percentile of the other 5,000 (SSE's 1% protocol). Flow / packet columns go through
      sign(x) log(1 + |x|) first, then (Isolation Forest aside) standard scaling fitted on those 5,000.
        Isolation Forest   300 trees, 256 samples per tree (Liu et al.'s recommended setting)
        One-Class SVM      RBF, gamma "scale", nu 0.01 (the 1% target)
        KitNET             Kitsune's detector at its paper settings: <= 10 inputs per autoencoder, 75% hidden,
                           lr 0.1; feature mapping on the first 1,000 records, training on the next 4,000
  supervised, labelled attacks: probability >= 0.5.
        Random Forest 300 trees; XGBoost 500 trees, depth 8, lr 0.1, subsample 0.9, colsample 0.8; CatBoost 800
        iterations, depth 8. Training data: SR-BH every labelled 17 Jul request; UNSW-NB15 the official training
        set; CSIC, CIC-IDS2017 and Kitsune have no separate labelled training traffic, so 2-fold cross-validation
        over the test stream (seeded halves): every row is scored by a model that never saw it. Random splits of
        a flow capture put near-identical flows on both sides, so these numbers are optimistic (as on the public
        leaderboards).
Results: results/bench/<dataset>/<method>.json. Size = pickled model; latency = model inference per record, in a batch and one record at a time.

usage: python -I bench/tabular_baselines.py <dataset> [method ...]
       datasets: srbh csic unsw cic_<file> kitsune_<capture> (see data/sse_streams)
"""
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from kitnet import KitNET  # noqa: E402

NAME = sys.argv[1]
ONLY = set(sys.argv[2:])
FPR = 0.01
THREADS = int(os.environ.get("BENCH_THREADS", 2))
STREAMS = os.path.join(common.ROOT, "data", "sse_streams")


def load():
    """fit (normal), test, y, cls, and train / ytrain or fold."""
    if NAME in ("srbh", "csic"):
        import http_data
        import http_features
        path = os.path.join(common.ROOT, "data", f"bench_{NAME}_features.npz")
        d = http_data.load(NAME)
        if not os.path.exists(path):
            out = {"fit": http_features.matrix(d["fit_normal"]), "test": http_features.matrix([r for r, _, _ in d["test"]]),
                   "y": np.array([a for _, a, _ in d["test"]])}
            if "train" in d:
                out["train"] = http_features.matrix([r for r, _, _ in d["train"]])
                out["ytrain"] = np.array([a for _, a, _ in d["train"]])
            np.savez(path, **out)
        z = dict(np.load(path))
        z["cls"] = [c for _, _, c in d["test"]]
        z["flow"] = False
    else:
        z = dict(np.load(os.path.join(STREAMS, f"{NAME}.npz")))
        z["cls"] = [c if c not in ("", "BENIGN", "Normal") else "" for c in z["cls"].tolist()]
        z["flow"] = True
    if "train" not in z:
        fold = np.array([i % 2 for i in range(len(z["y"]))])
        np.random.default_rng(12).shuffle(fold)
        z["fold"] = fold
    for k in ("fit", "test", "train"):
        if k in z:
            z[k] = np.nan_to_num(z[k].astype(np.float32), nan=0.0, posinf=1e12, neginf=-1e12)
    return z


def latency(fn, X):
    X = X[:20000]
    t = time.perf_counter()
    fn(X)
    batch = (time.perf_counter() - t) / len(X) * 1e6
    t = time.perf_counter()
    for i in range(300):
        fn(X[i:i + 1])
    return {"us_per_record_batch": batch, "us_per_record_single": (time.perf_counter() - t) / 300 * 1e6}


def main():
    z = load()
    y, cls = z["y"].astype(bool), z["cls"]
    info = {"dataset": NAME, "test_records": int(len(y)), "test_attacks": int(y.sum()),
            "dims": int(z["test"].shape[1]), "threads": THREADS}
    out = {}
    tf = (lambda X: np.sign(X) * np.log1p(np.abs(X))) if z["flow"] else (lambda X: X)
    fit, hold, test = tf(z["fit"][:5000]), tf(z["fit"][5000:10000]), tf(z["test"])
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(fit)
    sc.scale_[sc.scale_ < 1e-6] = 1.0

    def unsup(name, make, score, scaled=True):
        if ONLY and name not in ONLY:
            return
        f, h, t_ = (sc.transform(fit), sc.transform(hold), sc.transform(test)) if scaled else (fit, hold, test)
        t = common.Timer()
        m = make(f)
        fit_s = t()
        thr = common.quantile_threshold(score(m, h), FPR)
        s = score(m, t_)
        out[name] = common.report(y, s, s > thr, cls, fit_seconds=fit_s, threshold=thr, model_bytes=len(pickle.dumps(m)),
                                  **latency(lambda X: score(m, X), t_))
        print(common.line(f"{NAME} {name}", out[name]), flush=True)
        common.save(f"{NAME}/{name}.json", {**info, name: out[name]})

    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.svm import OneClassSVM
    unsup("isolation_forest", lambda f: IsolationForest(n_estimators=300, max_samples=256, random_state=0,
                                                        n_jobs=THREADS).fit(f), lambda m, X: -m.score_samples(X), False)
    unsup("ocsvm", lambda f: OneClassSVM(nu=0.01, gamma="scale").fit(f), lambda m, X: -m.decision_function(X))

    def kit(f):
        k = KitNET(f.shape[1], max_autoencoder_size=10, FM_grace_period=1000, AD_grace_period=4000)
        return k.fit(np.vstack([f, f[:1]]))
    unsup("kitnet", kit, lambda m, X: m.execute_batch(X))

    def sup(name, make):
        if ONLY and name not in ONLY:
            return
        t = common.Timer()
        if "train" in z:
            m = make().fit(z["train"], z["ytrain"].astype(int))
            p = m.predict_proba(z["test"])[:, 1]
            models, how = [m], "trained on the labelled training traffic"
        else:
            p, models = np.zeros(len(y)), []
            for k in (0, 1):
                tr, te = z["fold"] != k, z["fold"] == k
                m = make().fit(z["test"][tr], y[tr].astype(int))
                p[te] = m.predict_proba(z["test"][te])[:, 1]
                models.append(m)
            how = "2-fold cross-validation over the test stream"
        out[name] = common.report(y, p, p >= 0.5, cls, train_seconds=t(), protocol=how,
                                  model_bytes=len(pickle.dumps(models[0])),
                                  **latency(lambda X: models[0].predict_proba(X), z["test"]))
        print(common.line(f"{NAME} {name}", out[name]), flush=True)
        common.save(f"{NAME}/{name}.json", {**info, name: out[name]})

    sup("random_forest", lambda: RandomForestClassifier(n_estimators=300, n_jobs=THREADS, random_state=0))
    from xgboost import XGBClassifier
    sup("xgboost", lambda: XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1, subsample=0.9,
                                         colsample_bytree=0.8, tree_method="hist", n_jobs=THREADS, random_state=0))
    from catboost import CatBoostClassifier
    sup("catboost", lambda: CatBoostClassifier(iterations=800, depth=8, thread_count=THREADS, verbose=False,
                                               random_seed=0))
    print("done", NAME, sorted(out), flush=True)


if __name__ == "__main__":
    main()
