"""The ensemble switch of the robomimic studies, scored in the controlled domains through bcbench.

The support switch, plus a second monitor: the labeler also acts when five cloned networks (the paper's BC and four
more seeds, same data and steps) disagree by more than the 95th percentile of their disagreement on the training
frames (sum over action dimensions of the standard deviation across networks). The paper's BC acts otherwise.
Predictions: paper/notes/prediction_bench_ensemble_2026-11-28.md.

    python bench_ensemble.py --workers 16      # -> outputs/v3/bench_ensemble_<split>.csv
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcbench  # noqa: E402
import run as R  # noqa: E402

N_NETS, Q = 5, 95


def ensemble_switch(task):
    D, O, A, X = task.D, task.O, task.A, task.X
    bc = task.bc()
    nets = [bc] + [R.train(O, A, O, R.mseed(task.setting, task.seed, "ens", k)) for k in range(1, N_NETS)]

    def spread(o):
        return np.stack([n(o) for n in nets]).std(0).sum(1)

    thr = float(np.percentile(spread(O), Q))
    sup = R.support_switch(D, bc, O, X, D.r_nominal)

    def pol(o):
        out = sup(o)
        far = spread(o) > thr
        out[far] = D.labeler(o[far])
        return out
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    df = bcbench.evaluate("bench_ensemble:ensemble_switch", a.split, bcbench.SEEDS, a.workers)
    out = os.path.join(R.ROOT, "outputs", "v3", f"bench_ensemble_{a.split}.csv")
    df.to_csv(out, index=False)
    table, tests = bcbench.compare(df, "ensemble switch")
    import pandas as pd
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table)
        print(tests)


if __name__ == "__main__":
    main()
