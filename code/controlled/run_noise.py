"""Robustness of the controlled-domain results to an imperfect plant (reviewer request: the look-ahead switch and
backup filter are given the true, noise-free model).

The policies are built and trained exactly as in run.py (same demonstrations, seeds and steps). Only the evaluation
plant changes, while the look-ahead switch and backup filter keep predicting with the nominal noise-free model:
  noise      process noise at every step (the domain's demonstration noise) and sensor noise: the policies see
             the state plus N(0, (0.02 * std of the demonstration states)^2), from which observations are formed
  mismatch   the plant applies 0.8 times the commanded action, one step late (unknown to every policy and model)
Probes, nominal starts and metrics as in run.py; the demonstrator reference is rolled out on the same plant.

    python run_noise.py --workers 4      # -> outputs/v3/noise_<condition>/...
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as R  # noqa: E402

SETTINGS = [("quad3d", 3), ("quad3d", 10), ("reacher", 3), ("reacher", 10), ("pendulum", 30)]
METHODS = "BC,Switch,Switch-support,Simplex,Filter,Backup,Residual-box,Blend-box,DAgger-L"
SEEDS = range(10)
GAIN, SENSOR = 0.8, 0.02


def make_rollout(cond, D, sd):
    def rollout(D_, pol, x0):
        rng = np.random.default_rng(R.mseed("plant", cond, D_.name, len(x0), float(np.round(x0.sum(), 6))))
        x = x0.copy()
        if isinstance(pol, R.Stateful):
            pol.reset(len(x))
            f = pol.fn
        elif isinstance(pol, R.Predictive):
            f = None
        else:
            f = pol
        n = len(x)
        err = np.zeros((D_.horizon, n))
        U = np.zeros((D_.horizon, n, D_.nu))
        traj = np.zeros((D_.horizon, n, D_.nx))
        dv = np.zeros(n, dtype=bool)
        u_prev = np.zeros((n, D_.nu))
        for k in range(D_.horizon):
            xm = x + rng.normal(0, 1, x.shape) * sd if cond == "noise" else x
            o = D_.obs(xm)
            u = np.clip(pol.act(o, xm) if f is None else f(o), -1, 1)
            err[k] = D_.error(x)
            U[k] = u
            traj[k] = x
            if cond == "mismatch":
                x = D_.step(x, GAIN * u_prev)
                u_prev = u
            else:
                x = D_.process_noise(D_.step(x, u), rng)
            dv |= D_.diverged(x)
        return err, U, traj, dv
    return rollout


def job(args):
    cond, dname, eps, seed = args
    import domains as Dm
    D = Dm.DOMAINS[dname]()
    D.demo_eps = eps
    X, _, _ = R.demos(D, seed)
    sd = SENSOR * X.std(0)
    R.OUT = os.path.join(R.ROOT, "outputs", "v3", f"noise_{cond}")
    R.rollout = make_rollout(cond, D, sd)
    os.environ["ONLY"] = METHODS
    return R.run_seed((dname, seed, eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [(c, d, e, s) for c in ("noise", "mismatch") for d, e in SETTINGS for s in SEEDS]
    with ProcessPoolExecutor(a.workers) as ex:
        for p in ex.map(job, jobs):
            print(p, flush=True)


if __name__ == "__main__":
    main()
