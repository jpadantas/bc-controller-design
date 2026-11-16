"""Action-chunking BC on robomimic Lift (ACT-style), and the switches built on it.

Per (held-out operator, seed): BC trained to predict the next K actions from the current observation (targets
padded with the last action at the end of a demonstration); at run time every step predicts a chunk, and the
action applied is the exponentially weighted average (weight exp(-m i) for the prediction made i steps ago) of all
predictions for the current step -- the temporal ensembling of ACT. The chunked policy is queried at every step
(so its buffer stays current) and the error switch / support switch decide whether its action or the labeler's is
applied. Same networks, steps, monitors and evaluation as lift_study.py.
Writes outputs/robo/<LIFT_OUT>_chunk/p{op}_s{seed}.csv.

    MUJOCO_GL=disable ROBO_DATA=<path> LIFT_OUT=lift_big LIFT_STEPS=60000 LIFT_HIDDEN=1024 python lift_chunk.py
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lift_common as L  # noqa: E402
import lift_study as S  # noqa: E402

K = 8
M = 0.1


def chunk_targets(demos):
    Ys = []
    for d in demos:
        A = d["actions"]
        T = len(A)
        idx = np.minimum(np.arange(T)[:, None] + np.arange(K)[None, :], T - 1)
        Ys.append(A[idx].reshape(T, -1))
    return np.concatenate(Ys).astype(np.float32)


class Chunked:
    """Temporal ensembling over the chunks predicted at the last K steps."""

    def __init__(self, net):
        self.net = net
        self.reset()

    def reset(self):
        self.buf = []          # list of (age, chunk) with chunk (K, 7)

    def __call__(self, x):
        c = np.asarray(self.net(x)).reshape(K, 7)
        self.buf = [(age + 1, ch) for age, ch in self.buf if age + 1 < K] + [(0, c)]
        preds = np.array([ch[age] for age, ch in self.buf])
        w = np.exp(-M * np.array([age for age, _ in self.buf], dtype=float))
        return (w[:, None] * preds).sum(0) / w.sum()


def run(args):
    import pandas as pd
    from scipy.spatial import cKDTree
    held, seed = args
    out = S.OUT + "_chunk"
    path = os.path.join(out, f"p{held}_s{seed}.csv")
    if os.path.exists(path):
        return path
    data, kw = L.load()
    env = L.make_env(kw)
    tr_ops = [o for o in L.OPERATORS if o != held]
    demos = [d for o in tr_ops for d in data[o]]
    O = np.concatenate([d["obs"] for d in demos])
    Y = chunk_targets(demos)
    opid = np.concatenate([np.full(len(d["obs"]), i) for i, o in enumerate(tr_ops) for d in data[o]])
    net = S.train(O, Y, O, S.mseed("lift", held, seed, "BC-chunk"))
    mu, sd = O.mean(0), O.std(0) + 1e-6
    Z = (O - mu) / sd
    tree = cKDTree(Z)
    dd = np.empty(len(Z))
    for i in np.unique(opid):
        a = opid == i
        dd[a] = cKDTree(Z[~a]).query(Z[a])[0]
    thr = float(np.percentile(dd, 95))
    lo_e, hi_e = O[:, 0:3].min(0) - 0.02, O[:, 0:3].max(0) + 0.02
    lo_c, hi_c = O[:, 9:11].min(0) - 0.02, O[:, 9:11].max(0) + 0.02

    def in_workspace(x):
        return bool(np.all(x[0:3] >= lo_e) and np.all(x[0:3] <= hi_e) and np.all(x[9:11] >= lo_c) and np.all(x[9:11] <= hi_c))

    def in_support(x):
        return tree.query((x - mu) / sd)[0] <= thr

    lab = L.labeler

    class Gated:
        def __init__(self, cond):
            self.bc, self.cond = Chunked(net), cond

        def reset(self):
            self.bc.reset()

        def __call__(self, x):
            a = self.bc(x)                       # always queried, so the chunk buffer stays current
            return a if self.cond(x) else lab(x)

    pols = {"BC-chunk": Chunked(net), "Switch-chunk": Gated(in_workspace),
            "Support-chunk": Gated(lambda x: in_workspace(x) and in_support(x))}
    erng = np.random.default_rng(S.mseed("lift-eval", held))
    nom_states = []
    for _ in range(S.N_EVAL):
        env.reset()
        nom_states.append(env.sim.get_state().flatten().copy())
    off = S.offnominal_starts(env, S.N_EVAL, erng)
    rows = []
    for m, p in pols.items():
        r = dict(held_out=held, seed=seed, method=m)
        succ, kept = [], []
        for st in nom_states:
            X, U, s = L.rollout(env, p, state=st)
            succ.append(s)
            kept.append(all(in_workspace(x) for x in X))
        r.update(nom_success=np.mean(succ), retention=np.mean(kept))
        r["recovery"] = np.mean([L.rollout(env, p, state=st)[2] for _, st in off])
        dist, fs = [], []
        for d in data[held][::2]:
            X, U, s = L.rollout(env, p, state=d["states"][0], horizon=max(L.HORIZON, len(d["obs"]) + 20))
            P, R = X[:, 0:3], d["obs"][:, 0:3]
            T = max(len(P), len(R))
            P = np.concatenate([P, np.repeat(P[-1:], T - len(P), 0)])
            R = np.concatenate([R, np.repeat(R[-1:], T - len(R), 0)])
            dist.append(np.sqrt(np.mean(np.sum((P - R) ** 2, 1))))
            fs.append(s)
        r.update(fid_dist=float(np.mean(dist)), fid_success=float(np.mean(fs)))
        rows.append(r)
        print(held, seed, m, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    jobs = [(op, int(s)) for s in a.seeds.split(",") for op in L.OPERATORS]
    with ProcessPoolExecutor(a.workers) as ex:
        for p in ex.map(run, jobs):
            print("done", p, flush=True)


if __name__ == "__main__":
    main()
