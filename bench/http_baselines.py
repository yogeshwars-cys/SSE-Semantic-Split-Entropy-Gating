"""http_baselines.py: tabular baselines on the HTTP datasets (SR-BH 2020, CSIC 2010), on http_features vectors.

  unsupervised (normal traffic only, like SSE): Isolation Forest, One-Class SVM, KitNET (Kitsune's detector).
      Fit on the first 5,000 of the 10,000 normal requests SSE fits on, threshold at the 99th percentile of the
      other 5,000 (SSE's 1% protocol).
  supervised (labelled attacks): Random Forest, XGBoost, CatBoost. SR-BH: trained on every labelled 17 Jul request,
      tested on 18-19 Jul. CSIC: 2-fold cross-validation over the test stream (http_data.py).
Size = pickled model bytes; latency = model inference only, per request, batch and one request at a time
(feature extraction is timed separately).

usage: python -I bench/http_baselines.py <srbh|csic> [method ...]
"""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import http_data  # noqa: E402
import http_features  # noqa: E402
from kitnet import KitNET  # noqa: E402

NAME = sys.argv[1]
ONLY = set(sys.argv[2:])
FPR = 0.01


def features():
    path = os.path.join(common.ROOT, "data", f"bench_{NAME}_features.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    d = http_data.load(NAME)
    t = time.perf_counter()
    out = {"fit": http_features.matrix(d["fit_normal"]), "test": http_features.matrix([r for r, _, _ in d["test"]])}
    out["feature_us_per_request"] = np.array((time.perf_counter() - t) / (len(d["fit_normal"]) + len(d["test"])) * 1e6)
    out["y"] = np.array([a for _, a, _ in d["test"]])
    if "train" in d:
        out["train"] = http_features.matrix([r for r, _, _ in d["train"]])
        out["ytrain"] = np.array([a for _, a, _ in d["train"]])
    if "fold" in d:
        out["fold"] = np.array(d["fold"])
    np.savez(path, **out)
    return out


def latency(score_fn, X):
    t = time.perf_counter()
    score_fn(X)
    batch = (time.perf_counter() - t) / len(X) * 1e6
    one = X[:300]
    t = time.perf_counter()
    for i in range(len(one)):
        score_fn(one[i:i + 1])
    single = (time.perf_counter() - t) / len(one) * 1e6
    return {"us_per_request_batch": batch, "us_per_request_single": single}


def main():
    F = features()
    d = http_data.load(NAME)
    classes = [c for _, _, c in d["test"]]
    y = F["y"]
    out = {"dataset": NAME, "test_requests": int(len(y)), "test_attacks": int(y.sum()),
           "feature_dims": int(F["test"].shape[1]), "feature_us_per_request": float(F["feature_us_per_request"])}
    res_path = f"{NAME}/tabular.json"
    prev = os.path.join(common.RESULTS, res_path)
    if os.path.exists(prev) and ONLY:
        import json
        out = {**json.load(open(prev)), **out}
    from sklearn.preprocessing import StandardScaler
    fit, hold = F["fit"][:5000], F["fit"][5000:]
    sc = StandardScaler().fit(fit)
    Xf, Xh, Xt = sc.transform(fit), sc.transform(hold), sc.transform(F["test"])

    def unsup(name, model, score):
        if ONLY and name not in ONLY:
            return
        t = common.Timer()
        model = model()
        fit_s = t()
        thr = common.quantile_threshold(score(model, Xh), FPR)
        s = score(model, Xt)
        out[name] = common.report(y, s, s > thr, classes, fit_seconds=fit_s, threshold=thr,
                                  model_bytes=len(pickle.dumps(model)), **latency(lambda X: score(model, X), Xt[:20000]))
        print(common.line(name, out[name]), flush=True)
        common.save(res_path, out)

    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.svm import OneClassSVM
    unsup("isolation_forest", lambda: IsolationForest(n_estimators=200, random_state=0, n_jobs=4).fit(Xf),
          lambda m, X: -m.score_samples(X))
    unsup("ocsvm", lambda: OneClassSVM(nu=0.01, gamma="scale").fit(Xf), lambda m, X: -m.decision_function(X))

    def kit():
        k = KitNET(Xf.shape[1], max_autoencoder_size=10, FM_grace_period=1000, AD_grace_period=4000)
        k.fit(np.vstack([Xf, Xf[:1]]))
        return k
    unsup("kitnet", kit, lambda m, X: m.execute_batch(X))

    def sup(name, make):
        if ONLY and name not in ONLY:
            return
        t = common.Timer()
        if "train" in F:                                         # SR-BH: day before
            m = make().fit(F["train"], F["ytrain"])
            p = m.predict_proba(F["test"])[:, 1]
            models = [m]
        else:                                                    # CSIC: 2-fold cross-validation over the test stream
            p, models = np.zeros(len(y)), []
            for k in (0, 1):
                tr, te = F["fold"] != k, F["fold"] == k
                m = make().fit(F["test"][tr], y[tr])
                p[te] = m.predict_proba(F["test"][te])[:, 1]
                models.append(m)
        out[name] = common.report(y, p, p >= 0.5, classes, train_seconds=t(), model_bytes=len(pickle.dumps(models[0])),
                                  **latency(lambda X: models[0].predict_proba(X), F["test"][:20000]))
        print(common.line(name, out[name]), flush=True)
        common.save(res_path, out)

    sup("random_forest", lambda: RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=0))
    from xgboost import XGBClassifier
    sup("xgboost", lambda: XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1, tree_method="hist",
                                         n_jobs=4, random_state=0))
    from catboost import CatBoostClassifier
    sup("catboost", lambda: CatBoostClassifier(iterations=500, depth=8, thread_count=4, verbose=False, random_seed=0))
    print("wrote", common.save(res_path, out))


if __name__ == "__main__":
    main()
