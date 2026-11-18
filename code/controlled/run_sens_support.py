"""Sensitivity of the support switch to its threshold (reviewer request).

The support switch hands control to the labeler when the observation is farther from every demonstration frame than
the q-th percentile of the demonstrations' own leave-one-episode-out nearest-neighbour distances (default q = 95, in
the main results). Here q in {80, 90, 99}, on the same BC, probes and nominal starts as run_sens.py.

    python run_sens_support.py --domain quad3d --eps 3,30 --seeds 0-9
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import domains as Dm  # noqa: E402
import run as R  # noqa: E402

QS = (80, 90, 99)


def run_one(args):
    import pandas as pd
    dname, seed, eps = args
    D = Dm.DOMAINS[dname]()
    D.demo_eps = eps
    out = os.path.join(R.OUT, "sens_support", dname, f"eps{eps}_seed{seed}.csv")
    if os.path.exists(out):
        return out
    X, O, A = R.demos(D, seed)
    r0 = D.r_nominal
    bc = R.train(O, A, O, R.mseed(dname, seed, "BC"))
    pols = {f"Switch-support@{q}": R.support_switch(D, bc, O, X, r0, q=float(q)) for q in QS}
    prng = np.random.default_rng(R.mseed("probe", dname, seed))
    probes = {r: D.probe(r, R.N_PROBE, prng) for r in D.radii}
    x_nom = D.nominal_starts(R.N_PROBE, np.random.default_rng(R.mseed("nom", dname, seed)))
    ref_err, _, ref_traj, ref_dv = R.rollout(D, D.demonstrator, x_nom)
    ref_kept = (ref_err < D.r_nominal).all(0) & ~ref_dv
    rows = []
    for m, p in pols.items():
        fid = R.fidelity(D, p, x_nom, ref_traj, ref_kept)
        for r, x0 in probes.items():
            err, U, _, dv = R.rollout(D, p, x0)
            met = R.probe_metrics(D, err, U, dv)
            rows.append(dict(domain=dname, seed=seed, demo_eps=eps, method=m, radius=r,
                             success=float(np.mean(met["success"])), du_peak=float(np.mean(met["du_peak"])),
                             **fid))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="quad3d")
    ap.add_argument("--eps", default="3,30")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    jobs = [(a.domain, s, int(e)) for e in a.eps.split(",") for s in range(lo, hi + 1)]
    with ProcessPoolExecutor(a.workers) as ex:
        for p in ex.map(run_one, jobs):
            print("done", p, flush=True)


if __name__ == "__main__":
    main()
