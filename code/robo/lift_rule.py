"""The frozen three-outcome rule on robomimic Lift (paper/notes/prediction_lift_2026-11-13.md): per (held-out
operator, seed), the error switch if BC never leaves the demonstrated workspace from the nominal starts (as the
labeler does not), else the residual blend if it never leaves, else the support switch. A choice is correct if
its distance to the held-out operator is within 5% of the labeler's distance of the closest of the three.

    python lift_rule.py            (reads outputs/robo/<LIFT_OUT>/*.csv)
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOL = 0.05


def main():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, "outputs", "robo",
                                                                    os.environ.get("LIFT_OUT", "lift_big"), "*.csv"))])
    t = d.pivot_table(index=["held_out", "seed"], columns="method", values=["retention", "fid_dist"])
    ret, f = t["retention"], t["fid_dist"]
    lab_ret = ret["Labeler"] - 1e-9
    pick = np.where(ret["BC"] >= lab_ret, "Switch", np.where(ret["Residual-box"] >= lab_ret, "Residual-box", "Support"))
    cands = ["Switch", "Residual-box", "Support"]
    best = f[cands].min(1).values
    tol = TOL * f["Labeler"].values
    chosen = np.array([f.loc[i, p] for i, p in zip(f.index, pick)])
    print("units", len(f), "picks", dict(zip(*np.unique(pick, return_counts=True))))
    print("correct: rule", int((chosen <= best + tol).sum()),
          {m: int((f[m].values <= best + tol).sum()) for m in cands})
    print("mean excess / labeler's distance: rule", round(float(np.mean((chosen - best) / f["Labeler"].values)), 4))


if __name__ == "__main__":
    main()
