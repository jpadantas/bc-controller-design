"""Does the measured condition of Theorem 3.1(b) predict retention? (quadrotor)

Joins the boundary-band exit rates of theory_check.py (outputs/v3/theory_check_quad.csv) with the nominal retention
of the main experiment (outputs/v3/quad3d/eps*/seed*.csv) per (episodes, seed, method), and reports Spearman's rank
correlation between the exit rate and the loss of retention. The error switch and Simplex apply BC on the band and
have the same exit rate, so only BC is kept of the three, and the labeler and demonstrator are left out.

    python theory_corr.py
"""
import glob
import os

import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    t = pd.read_csv(os.path.join(ROOT, "outputs", "v3", "theory_check_quad.csv"))
    rows = []
    for f in glob.glob(os.path.join(ROOT, "outputs", "v3", "quad3d", "eps*", "seed*.csv")):
        d = pd.read_csv(f)
        seed = int(os.path.basename(f)[4:-4])
        for m, v in d.groupby("method").kept_inner.first().items():
            rows.append(dict(demo_eps=int(d.demo_eps.iloc[0]), seed=seed, method=m, ret=v))
    m = t.merge(pd.DataFrame(rows), on=["demo_eps", "seed", "method"])
    m = m[~m.method.isin(["Switch", "Simplex", "Labeler", "Demonstrator"])]
    r = spearmanr(m.exit, 1 - m.ret)
    print(f"all {len(m)} units: rho = {r.correlation:.2f} (p = {r.pvalue:.1e})")
    for e, g in m.groupby("demo_eps"):
        r = spearmanr(g.exit, 1 - g.ret)
        print(f"{e} episodes, {len(g)} units: rho = {r.correlation:.2f} (p = {r.pvalue:.1e}), "
              f"mean retention {100 * g.ret.mean():.1f}%")


if __name__ == "__main__":
    main()
