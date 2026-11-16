"""Closed-loop cross-validation on the demonstrations: choose among the error switch, the residual blend
and the look-ahead switch without querying the demonstrator.

For each (domain, budget, seed), the demonstration episodes of run.py are split into K folds (K = number
of episodes, at most 5). For each fold, BC and the residual blend are trained on the other episodes with
the recipe of run.py; each candidate (error switch, look-ahead switch on that BC, residual blend) is run
from the initial states of the held-out episodes, and its trajectory is compared with the recorded one
(the domain's fidelity distance). Candidates that leave N from those starts are flagged. Writes
outputs/v3/cvselect/{domain}/eps{E}/seed{S}.csv with the cross-validated distance and exit flag of each
candidate. The choice is scored against the full-data policies of run.py by stats_v3.py.

    python cvselect.py --domain quad3d --eps 3,5,10,30 --seeds 0-14
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import domains as Dm  # noqa: E402
import run as R  # noqa: E402


def episodes(D, seed):
    """The demonstrations of run.demos, reshaped per episode: X (T, E, nx), O (T, E, nobs), A (T, E, nu)."""
    X, O, A = R.demos(D, seed)
    T, E = D.demo_T, D.demo_eps
    return X.reshape(T, E, -1), O.reshape(T, E, -1), A.reshape(T, E, -1)


def one(args):
    import pandas as pd
    dname, seed, eps = args
    D = Dm.DOMAINS[dname]()
    D.demo_eps = eps
    path = os.path.join(R.OUT, "cvselect", dname, f"eps{eps}", f"seed{seed}.csv")
    if os.path.exists(path):
        return path
    X, O, A = episodes(D, seed)
    E = X.shape[1]
    K = min(E, 5)
    folds = np.array_split(np.arange(E), K)
    r0 = D.r_nominal
    n_aug = int(R.AUG_FRAC * D.demo_T * 30)
    acc = {m: [] for m in ["Switch", "Filter", "Residual-box"]}
    left = {m: 0 for m in acc}
    for j, hold in enumerate(folds):
        tr = np.setdiff1d(np.arange(E), hold)
        Ot, At = O[:, tr].reshape(-1, O.shape[2]), A[:, tr].reshape(-1, A.shape[2])
        bc = R.train(Ot, At, Ot, R.mseed("cv", dname, seed, j, "BC"))
        rng = np.random.default_rng(R.mseed("cv", dname, seed, j, "aug"))
        S = D.obs(D.box_states(n_aug, rng, r0))
        res = At - D.labeler(Ot)
        rnet = R.train(np.concatenate([Ot, S]), np.concatenate([res, np.zeros((len(S), D.nu))]), Ot,
                       R.mseed("cv", dname, seed, j, "Residual-box"))

        def switch(o, bc=bc):
            out = bc(o)
            far = D.obs_error(o) >= r0
            out[far] = D.labeler(o[far])
            return out
        pols = {"Switch": switch, "Filter": R.Predictive(D, bc, r0),
                "Residual-box": (lambda o, rnet=rnet: np.clip(D.labeler(o) + rnet(o), -1, 1))}
        x0 = X[0, hold]
        T = min(D.horizon, X.shape[0])
        ref = X[:T, hold]
        for m, p in pols.items():
            err, _, traj, dv = R.rollout(D, p, x0)
            dpos = D.fid_distance(traj[:T], ref)
            acc[m].append(dpos ** 2)
            left[m] += int(((err[:T] >= r0).any(0) | dv).sum())
    row = dict(domain=dname, seed=seed, demo_eps=eps)
    for m in acc:
        row[f"cv_{m}"] = float(np.sqrt(np.concatenate([a.ravel() for a in acc[m]]).mean()))
        row[f"cvleft_{m}"] = left[m]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="quad3d")
    ap.add_argument("--eps", default="3")
    ap.add_argument("--seeds", default="0-14")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    jobs = [(a.domain, s, int(e)) for e in a.eps.split(",") for s in range(lo, hi + 1)]
    with ProcessPoolExecutor(a.workers) as ex:
        for p in ex.map(one, jobs):
            print("done", p, flush=True)


if __name__ == "__main__":
    main()
