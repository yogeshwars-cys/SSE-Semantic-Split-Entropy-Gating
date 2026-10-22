"""kitnet.py: KitNET, the anomaly detector of Kitsune (Mirsky et al., NDSS 2018), vendored from
https://github.com/ymirsky/Kitsune-py (KitNET/KitNET.py, dA.py, corClust.py; MIT licence, notice below).

Unchanged algorithm: an incremental correlation clustering maps the features onto an ensemble of small autoencoders
(at most `max_autoencoder_size` inputs each) during the feature-mapping grace period, the ensemble and an output
autoencoder then train one sample at a time (SGD, online 0-1 normalisation) during the anomaly-detection grace
period, and the score of a sample is the output autoencoder's RMSE. Edits: numpy 2 (inf instead of Inf), no prints,
and execute_batch(), the same arithmetic as execute() on many rows at once (execution does not change the model).

Copyright (c) 2017 Yisroel Mirsky; portions (dA) Copyright (c) 2017 Yusuke Sugomori. MIT License:
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice
and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree

np.seterr(all="ignore")


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


class dA:
    def __init__(self, n_visible, lr=0.1, hidden_ratio=0.75):
        self.n_visible, self.lr = n_visible, lr
        self.n_hidden = int(np.ceil(n_visible * hidden_ratio))
        self.norm_max = np.ones((n_visible,)) * -np.inf
        self.norm_min = np.ones((n_visible,)) * np.inf
        self.n = 0
        self.rng = np.random.RandomState(1234)
        a = 1. / n_visible
        self.W = np.array(self.rng.uniform(low=-a, high=a, size=(n_visible, self.n_hidden)))
        self.hbias = np.zeros(self.n_hidden)
        self.vbias = np.zeros(n_visible)
        self.W_prime = self.W.T

    def train(self, x):
        self.n += 1
        self.norm_max[x > self.norm_max] = x[x > self.norm_max]
        self.norm_min[x < self.norm_min] = x[x < self.norm_min]
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
        y = sigmoid(np.dot(x, self.W) + self.hbias)
        z = sigmoid(np.dot(y, self.W_prime) + self.vbias)
        L_h2 = x - z
        L_h1 = np.dot(L_h2, self.W) * y * (1 - y)
        L_W = np.outer(x.T, L_h1) + np.outer(L_h2.T, y)
        self.W += self.lr * L_W
        self.hbias += self.lr * L_h1
        self.vbias += self.lr * L_h2
        return np.sqrt(np.mean(L_h2 ** 2))

    def execute_batch(self, X):
        X = (X - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
        Z = sigmoid(np.dot(sigmoid(np.dot(X, self.W) + self.hbias), self.W_prime) + self.vbias)
        return np.sqrt(((X - Z) ** 2).mean(axis=1))


class corClust:
    def __init__(self, n):
        self.n = n
        self.c = np.zeros(n)
        self.c_r = np.zeros(n)
        self.c_rs = np.zeros(n)
        self.C = np.zeros((n, n))
        self.N = 0

    def update(self, x):
        self.N += 1
        self.c += x
        c_rt = x - self.c / self.N
        self.c_r += c_rt
        self.c_rs += c_rt ** 2
        self.C += np.outer(c_rt, c_rt)

    def cluster(self, maxClust):
        c_rs_sqrt = np.sqrt(self.c_rs)
        C_rs_sqrt = np.outer(c_rs_sqrt, c_rs_sqrt)
        C_rs_sqrt[C_rs_sqrt == 0] = 1e-100
        D = 1 - self.C / C_rs_sqrt
        D[D < 0] = 0
        Z = linkage(D[np.triu_indices(self.n, 1)])
        maxClust = min(max(maxClust, 1), self.n)
        return self._break(to_tree(Z), maxClust)

    def _break(self, dendro, maxClust):
        if dendro.count <= maxClust:
            return [dendro.pre_order()]
        return self._break(dendro.get_left(), maxClust) + self._break(dendro.get_right(), maxClust)


class KitNET:
    def __init__(self, n, max_autoencoder_size=10, FM_grace_period=None, AD_grace_period=10000, learning_rate=0.1,
                 hidden_ratio=0.75):
        self.AD_grace_period = AD_grace_period
        self.FM_grace_period = AD_grace_period if FM_grace_period is None else FM_grace_period
        self.m = max(1, max_autoencoder_size)
        self.lr, self.hr, self.n = learning_rate, hidden_ratio, n
        self.ensembleLayer, self.outputLayer, self.v = [], None, None
        self.n_trained = 0
        self.FM = corClust(n)

    def train(self, x):
        if self.n_trained <= self.FM_grace_period and self.v is None:
            self.FM.update(x)
            if self.n_trained == self.FM_grace_period:
                self.v = self.FM.cluster(self.m)
                self.ensembleLayer = [dA(len(m), self.lr, self.hr) for m in self.v]
                self.outputLayer = dA(len(self.v), self.lr, self.hr)
        else:
            S_l1 = np.array([ae.train(x[self.v[a]]) for a, ae in enumerate(self.ensembleLayer)])
            self.outputLayer.train(S_l1)
        self.n_trained += 1

    def fit(self, X):
        """Kitsune's grace periods: the first FM_grace_period rows map features, the next AD_grace_period train."""
        for x in X:
            self.train(np.asarray(x, dtype=np.float64))
        return self

    def execute_batch(self, X):
        X = np.asarray(X, dtype=np.float64)
        S = np.stack([ae.execute_batch(X[:, self.v[a]]) for a, ae in enumerate(self.ensembleLayer)], axis=1)
        return self.outputLayer.execute_batch(S)

    def nbytes(self):
        aes = self.ensembleLayer + [self.outputLayer]
        return int(sum(a.W.nbytes + a.hbias.nbytes + a.vbias.nbytes + a.norm_max.nbytes + a.norm_min.nbytes for a in aes))
