"""A controller fitted to the demonstrations instead of an engineered one (reviewer request).

In each seed the labeler is replaced by an affine state-feedback law u = clip(W o + b), fitted by ridge regression to
that seed's demonstrations, and the combinations are rebuilt around it exactly as in run.py. Predictions:
paper/notes/prediction_fitted_2026-11-28.md.

    python run_fitted.py --workers 16      # -> outputs/v3/fitted/only_*/<domain>/eps*/seed*.csv, fitted_err.csv
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import domains as Dm  # noqa: E402
import run as R  # noqa: E402

SETTINGS = [("quad3d", 3), ("quad3d", 10), ("reacher", 3), ("reacher", 10), ("pendulum", 30)]
METHODS = "BC,Switch,Switch-support,Residual-box,Blend-box,DAgger-L"
SEEDS = range(10)
RIDGE = 1e-3


def fit_affine(O, A):
    F = np.hstack([O, np.ones((len(O), 1))])
    W = np.linalg.solve(F.T @ F + RIDGE * len(F) * np.eye(F.shape[1]), F.T @ A)
    return lambda o: np.clip(np.hstack([o, np.ones((len(o), 1))]) @ W, -1, 1)


def job(args):
    dname, eps, seed = args
    base = Dm.DOMAINS[dname]
    D0 = base()
    D0.demo_eps = eps
    _, O, A = R.demos(D0, seed)
    law = fit_affine(O, A)

    class Fitted(base):
        def labeler(self, o):
            return law(o)
    Fitted.__name__ = base.__name__ + "Fitted"
    _, Oh, Ah = R.demos(D0, seed + 1000)
    err = dict(domain=dname, demo_eps=eps, seed=seed,
               fitted=float(np.sqrt(((law(Oh) - Ah) ** 2).sum(1)).mean()),
               engineered=float(np.sqrt(((np.clip(D0.labeler(Oh), -1, 1) - Ah) ** 2).sum(1)).mean()))
    Dm.DOMAINS[dname] = Fitted
    R.Dm.DOMAINS[dname] = Fitted
    R.OUT = os.path.join(R.ROOT, "outputs", "v3", "fitted")
    os.environ["ONLY"] = METHODS
    try:
        path = R.run_seed((dname, seed, eps))
    finally:
        Dm.DOMAINS[dname] = base
        R.Dm.DOMAINS[dname] = base
    return path, err


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = [(d, e, s) for d, e in SETTINGS for s in SEEDS]
    errs = []
    with ProcessPoolExecutor(a.workers) as ex:
        for p, e in ex.map(job, jobs):
            errs.append(e)
            print(p, {k: round(v, 3) for k, v in e.items() if isinstance(v, float)}, flush=True)
    pd.DataFrame(errs).to_csv(os.path.join(R.ROOT, "outputs", "v3", "fitted", "fitted_err.csv"), index=False)


if __name__ == "__main__":
    main()
