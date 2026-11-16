"""Sensitivity of the conclusions to the hyperparameters we fixed (reviewer request).

Per (domain, demonstration budget, seed):
  Switch@k / Simplex@k   switching radius (Simplex: its error thresholds) scaled by k in {0.7, 1.4}
  Residual-box@f          residual blend with f in {0.1, 1.0} x the 30-episode set (default 0.3)
  Blend-box@f             box blend with the same synthetic budgets
Defaults (k = 1, f = 0.3) are in the main results.

    python run_sens.py --domain quad3d --eps 3,30 --seeds 0-9
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import domains as Dm  # noqa: E402
import run as R  # noqa: E402


def scaled_simplex(D, bc, r_in, k):
    """Simplex whose error thresholds (hand-over and hand-back) are scaled by k; the
    attitude/speed limits of the envelope are kept."""
    pol = R.Stateful(None)

    def fn(o):
        bad, good = D.envelope(o, r_in * k)
        pol.mode = np.where(pol.mode, ~good, bad)
        out = bc(o)
        out[pol.mode] = D.labeler(o[pol.mode])
        return out
    pol.fn = fn
    return pol


def run_one(args):
    import pandas as pd
    dname, seed, eps = args
    D = Dm.DOMAINS[dname]()
    D.demo_eps = eps
    out = os.path.join(R.OUT, "sens", dname, f"eps{eps}_seed{seed}.csv")
    if os.path.exists(out):
        return out
    X, O, A = R.demos(D, seed)
    r0 = D.r_nominal
    n0 = int(R.AUG_FRAC * D.demo_T * 30)
    rng = np.random.default_rng(R.mseed("sens", dname, seed))
    bc = R.train(O, A, O, R.mseed(dname, seed, "BC"))
    pols = {}
    for k in (0.5, 0.7, 0.85, 1.4):
        def sw(o, k=k):
            out = bc(o)
            far = D.obs_error(o) >= k * r0
            out[far] = D.labeler(o[far])
            return out
        pols[f"Switch@{k}"] = sw
        pols[f"Simplex@{k}"] = scaled_simplex(D, bc, r0, k)
    res = A - D.labeler(O)
    for f in (0.1, 1.0):           # fraction of the 30-episode demonstration set (default 0.3)
        n = int(n0 * (f / 0.3))
        S = D.obs(D.box_states(n, rng, r0))
        rnet = R.train(np.concatenate([O, S]), np.concatenate([res, np.zeros((len(S), D.nu))]), O,
                       R.mseed(dname, seed, "Rsens", f))
        pols[f"Residual-box@{f}"] = (lambda o, rnet=rnet: np.clip(D.labeler(o) + rnet(o), -1, 1))
        if f == 1.0:   # capacity / training ablation for the learned blend: 2x width, 3x steps
            pols["Blend-box-big"] = R.train(np.concatenate([O, S]), np.concatenate([A, D.labeler(S)]), O,
                                            R.mseed(dname, seed, "Bbig"), hidden=256, steps=3 * R.STEPS)
        pols[f"Blend-box@{f}"] = R.train(np.concatenate([O, S]), np.concatenate([A, D.labeler(S)]), O,
                                         R.mseed(dname, seed, "Bsens", f))
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
