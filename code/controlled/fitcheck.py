"""Residual check: does the residual blend fit held-out demonstrations at least as well as BC?

For each (domain, budget, seed): train BC and the residual blend's network (zero targets on box
states) with the recipe of run.py, and measure the action MSE of each policy on a held-out set of
demonstrations of the same size (fresh demonstrator episodes from fresh nominal starts).
Writes outputs/v3/fitcheck/{domain}/eps{E}/seed{S}.csv.

    python fitcheck.py --domain quad3d --eps 3,5,10,30 --seeds 0-14
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import domains as Dm  # noqa: E402
import run as R  # noqa: E402


def one(args):
    import pandas as pd
    dname, seed, eps = args
    D = Dm.DOMAINS[dname]()
    D.demo_eps = eps
    path = os.path.join(R.OUT, "fitcheck", dname, f"eps{eps}", f"seed{seed}.csv")
    if os.path.exists(path):
        return path
    _, O, A = R.demos(D, seed)
    _, Oh, Ah = R.demos(D, f"heldout{seed}")
    bc = R.train(O, A, O, R.mseed(dname, seed, "BC"))
    n_aug = int(R.AUG_FRAC * D.demo_T * 30)
    rng = np.random.default_rng(R.mseed("fitcheck", dname, seed))
    S = D.obs(D.box_states(n_aug, rng, D.r_nominal))
    res = A - D.labeler(O)
    rnet = R.train(np.concatenate([O, S]), np.concatenate([res, np.zeros((len(S), D.nu))]), O,
                   R.mseed(dname, seed, "Residual-box"))
    p_bc = np.clip(bc(Oh), -1, 1)
    p_res = np.clip(D.labeler(Oh) + rnet(Oh), -1, 1)
    row = dict(domain=dname, seed=seed, demo_eps=eps,
               fit_bc=float(np.mean((p_bc - Ah) ** 2)), fit_res=float(np.mean((p_res - Ah) ** 2)),
               fit_lab=float(np.mean((np.clip(D.labeler(Oh), -1, 1) - Ah) ** 2)),
               var_d=float(A.var(0).sum()), var_res=float(res.var(0).sum()))
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
