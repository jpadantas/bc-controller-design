"""Aggregate v3 controlled results: per (domain, demonstration budget, method),
mean over seeds of: recovery success over off-nominal radii, staying in the
nominal set from nominal starts, final error, peak control step, and
closed-loop fidelity to the demonstrator from nominal starts.

    python analyze.py [domain]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "outputs", "v3")
ORDER = ["BC", "DART", "Switch", "Switch-DART", "Switch-support", "SoftGate", "Simplex", "Filter", "Backup", "Residual", "Residual-box", "Blend-part", "Blend-box", "Blend-tube", "DAgger-L",
         "Blend-box@0.5", "Blend-box@2.0", "Blend-box@3.0", "Labeler", "Demonstrator"]


def load(domain):
    fs = glob.glob(os.path.join(OUT, domain, "eps*", "seed*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs]) if fs else pd.DataFrame()


def per_seed(d):
    off = d[d.radius > 0].groupby(["demo_eps", "method", "seed"]).agg(
        recover=("success", "mean"), final=("final", "mean"), du=("du_peak", "mean"),
        diverged=("diverged", "mean"), time_in=("time_in", "mean"))
    nom = d[d.radius == 0].groupby(["demo_eps", "method", "seed"]).agg(nominal_ok=("success", "mean"),
                                                                       stay=("stay", "mean"))
    fid = d.groupby(["demo_eps", "method", "seed"]).agg(fid=("fid_state", "first"), fid_a=("fid_action", "first"),
                                                        fid_kept=("fid_kept", "first"), kept=("kept_frac", "first"),
                                                        fid_common=("fid_common", "first"), kept_inner=("kept_inner", "first"))
    return off.join(nom).join(fid).reset_index()


def main():
    domain = sys.argv[1] if len(sys.argv) > 1 else "quad3d"
    d = load(domain)
    s = per_seed(d)
    pd.set_option("display.width", 250)
    for eps, g in s.groupby("demo_eps"):
        print(f"\n=== {domain}  demonstrations: {eps} episodes  (seeds: {g.seed.nunique()})")
        t = g.groupby("method")[["recover", "nominal_ok", "time_in", "du", "fid", "fid_kept", "kept", "fid_a"]].mean()
        t = t.reindex([m for m in ORDER if m in t.index])
        ref = g[g.method == "Switch"].set_index("seed")
        pv = {}
        for m in t.index:
            x = g[g.method == m].set_index("seed")
            i = x.index.intersection(ref.index)
            dd = (x.loc[i, "recover"] - ref.loc[i, "recover"]).values
            pv[m] = stats.wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
        t["p_recover_vs_switch"] = pd.Series(pv)
        print(t.round(4).to_string())


if __name__ == "__main__":
    main()
