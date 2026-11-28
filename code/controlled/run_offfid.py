"""Fidelity during recovery (reviewer request): in the quadrotor and Reacher the demonstrator itself recovers from
the probes, so each method's recovery can be compared with the demonstrator's own.

The policies are built exactly as in run.py. For every off-nominal probe, the method's trajectory is compared with
the demonstrator's from the same start (RMS of the same distance as the nominal-start fidelity), on the probes from
which the demonstrator recovers. The per-seed mean is the column `offfid` (probes with radius > 0).
Predictions: paper/notes/prediction_offfid_2026-11-28.md.

    python run_offfid.py --workers 16      # -> outputs/v3/offfid/only_*/<domain>/eps*/seed*.csv
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as R  # noqa: E402

SETTINGS = [("quad3d", 3), ("quad3d", 10), ("reacher", 3), ("reacher", 10)]
METHODS = "BC,Switch,Switch-support,Simplex,Filter,Backup,Residual-box,Blend-box,DAgger-L"
SEEDS = range(10)


def install():
    if getattr(R, "_offfid", False):   # worker processes are reused across jobs
        return
    R._offfid = True
    base_rollout, base_metrics = R.rollout, R.probe_metrics
    last, cache = {}, {}

    def rollout(D, pol, x0):
        out = base_rollout(D, pol, x0)
        last["x0"], last["traj"] = x0, out[2]
        return out

    def probe_metrics(D, err, U, dv):
        m = base_metrics(D, err, U, dv)
        x0 = last["x0"]
        key = (D.name, x0.tobytes())
        if key not in cache:
            e, _, tr, d = base_rollout(D, D.demonstrator, x0)
            cache[key] = (tr, (e[-1] < D.success) & ~d)
        ref, ok = cache[key]
        dist = np.sqrt((D.fid_distance(last["traj"], ref) ** 2).mean(0))
        m["offfid"] = np.where(ok, dist, np.nan)
        return m

    R.rollout, R.probe_metrics = rollout, probe_metrics


def job(args):
    dname, eps, seed = args
    install()
    R.OUT = os.path.join(R.ROOT, "outputs", "v3", "offfid")
    os.environ["ONLY"] = METHODS
    return R.run_seed((dname, seed, eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    jobs = [(d, e, s) for d, e in SETTINGS for s in SEEDS]
    with ProcessPoolExecutor(a.workers) as ex:
        for p in ex.map(job, jobs):
            print(p, flush=True)


if __name__ == "__main__":
    main()
