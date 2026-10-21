"""engine.py: the repo's SSE gates wired for the demo cache.

Statistical gate + semantic gate score every request; per-(route, field) thresholds from the Calibrator decide what
fires. A split mismatch (a poisoned fill) is a miss: blame the reflected header, make the gates more sensitive and
tighten its threshold on that route. An audited gate alarm whose response matched the clean copy is a false alarm:
the value joins the baseline and the threshold loosens. Same loop as harness/replay_learn.py."""
import copy
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sse"))
from anomaly_model import Gate  # noqa: E402
from calibration import Calibrator, blame  # noqa: E402
from semantic_model import Semantic  # noqa: E402

SKIP = {"host", "content-length", "x-sse-part"}                 # transport details, not request content


def _r(x):
    return None if x is None else round(float(x), 2)


class Engine:
    def __init__(self, train, fpr=0.01):
        t = time.time()
        self.gate, self.sem = Gate(fpr=fpr), Semantic(fpr=fpr)
        self.gate.fit(train)
        self.sem.fit(train)
        self.fit_s = round(time.time() - t, 2)
        self.pristine = copy.deepcopy((self.gate, self.sem))
        self.cal = Calibrator(halflife=None)
        self.lock = threading.RLock()

    def reset(self):
        """Forget everything learned during the demo; back to the fitted gates."""
        with self.lock:
            self.gate, self.sem = copy.deepcopy(self.pristine)
            self.cal = Calibrator(halflife=None)

    def inspect(self, headers, pnames, rt):
        h = {k: v for k, v in headers.items() if k not in SKIP}
        now = time.time()
        with self.lock:
            g, s, cal = self.gate, self.sem, self.cal
            zg = g.header_z(h)
            zg["_req"] = g.score(h, pnames)
            zs = s.header_z(h)
            og = cal.over(rt, "gate", zg, g.thr, now)
            os_ = cal.over(rt, "sem", zs, s.thr, now)
            rows = [{"name": k, "value": v if len(v) <= 160 else v[:157] + "...", "len": len(v),
                     "zg": _r(zg.get(k)), "tg": _r(cal.thr(rt, "gate", k, g.thr, now)),
                     "zs": _r(zs.get(k)), "ts": _r(cal.thr(rt, "sem", k, s.thr, now)) if k in zs else None,
                     "fg": k in og, "fs": k in os_} for k in h]
            req = {"z": _r(zg["_req"]), "t": _r(cal.thr(rt, "gate", "_req", g.thr, now)), "fired": "_req" in og}
        return {"h": h, "pn": pnames, "rt": rt, "zg": zg, "zs": zs, "og": og, "os": os_,
                "flagged": bool(og or os_), "view": {"rows": rows, "req": req, "params": pnames}}

    def learn_miss(self, insp, flagged_body, clean_body, qpairs):
        """A poisoned fill got past the gates. Returns (blamed, changes) for the dashboards."""
        h, pn, rt, zg, zs = insp["h"], insp["pn"], insp["rt"], insp["zg"], insp["zs"]
        now = time.time()
        found = blame(flagged_body, clean_body, h, qpairs)
        fields = {k: v for k, v in zg.items() if k != "_req"}
        if found:                                               # only headers can be gated; a blamed param stays split's job
            targets = [k for kind, k in found if kind == "header" and k in fields]
        else:                                                   # nothing reflected verbatim: blame the oddest field
            targets = [max(fields, key=fields.get)] if fields else []
        changes = []
        with self.lock:
            g, s, cal = self.gate, self.sem, self.cal
            for k in targets:
                w0 = g.w.get(k, 1.0)
                g.learn(h, pn, True, blamed=k)
                changes.append({"type": "weight", "field": k, "old": _r(w0), "new": _r(g.w.get(k, 1.0))})
                for layer, z, base in (("gate", zg.get(k), g.thr), ("sem", zs.get(k), s.thr)):
                    if z is None:
                        continue
                    old = cal.thr(rt, layer, k, base, now)
                    if cal.tighten(rt, layer, k, z, base, now):
                        changes.append({"type": "tighten", "layer": layer, "route": rt, "field": k,
                                        "old": _r(old), "new": _r(cal.thr(rt, layer, k, base, now))})
            if zs:
                s.learn(h, True, zs=zs)
                changes.append({"type": "sharpen", "field": max(zs, key=zs.get)})
        return [list(b) for b in found], changes

    def learn_false_alarm(self, insp):
        """A gate alarm whose response matched the clean copy: this request was normal."""
        h, pn, rt, zg, zs, og, os_ = (insp[k] for k in ("h", "pn", "rt", "zg", "zs", "og", "os"))
        now = time.time()
        changes = []
        with self.lock:
            g, s, cal = self.gate, self.sem, self.cal
            g.learn(h, pn, False, fired=[k for k in og if k != "_req"])
            s.learn(h, False, fired=os_)
            for layer, fired, zz, base in (("gate", og, zg, g.thr), ("sem", os_, zs, s.thr)):
                for k in fired:
                    old = cal.thr(rt, layer, k, base, now)
                    if cal.loosen(rt, layer, k, zz[k], base, now):
                        changes.append({"type": "loosen", "layer": layer, "route": rt, "field": k,
                                        "old": _r(old), "new": _r(cal.thr(rt, layer, k, base, now))})
        return changes

    def state(self):
        now = time.time()
        with self.lock:
            g, s, cal = self.gate, self.sem, self.cal
            bases = {"gate": g.thr, "sem": s.thr}
            over = [{"route": r, "layer": l, "field": n, "thr": _r(cal.thr(r, l, n, bases[l], now)),
                     "base": _r(bases[l])} for (r, l, n) in list(cal.o)]
            return {"gate_thr": _r(g.thr), "sem_thr": _r(s.thr), "overrides": over,
                    "weights": {k: _r(v) for k, v in g.w.items()},
                    "boosted": {k: int((sp.gain > 1).sum()) for k, sp in s.space.items() if (sp.gain > 1).any()},
                    "baseline_n": g.base.N, "fit_s": self.fit_s}
