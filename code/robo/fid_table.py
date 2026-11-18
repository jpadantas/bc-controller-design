"""Table of fidelity per rollout and the support switch's threshold on Lift and Can (-> paper/tables/fid.tex).

Inputs: outputs/robo/fid/fid_rollouts.csv (fid_study.py) and, for the off-nominal recovery of the policies that
fid_study.py evaluates only from the operators' starts, outputs/robo/ens/ens_results.csv (same networks and starts).
Means over six held-out operators and three seeds (the labeler: one evaluation per operator).

    python fid_table.py
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BS = "\\"
ROWS = [("BC", "BC"), ("Switch", "Error switch"), ("Support@80", "Support switch, 80th pct."),
        ("Support@90", "\\quad 90th"), ("Support@95", "\\quad 95th (default)"), ("Support@99", "\\quad 99th"),
        ("Ensemble", "Ensemble switch"), ("Labeler", "Labeler alone")]
ENS = {"BC": "BC", "Support@95": "Support", "Ensemble": "Ensemble", "Labeler": "Labeler"}


def main():
    d = pd.read_csv(os.path.join(ROOT, "outputs", "robo", "fid", "fid_rollouts.csv"))
    e = pd.read_csv(os.path.join(ROOT, "outputs", "robo", "ens", "ens_results.csv"))
    lines = [BS + "begin{tabular}{llcccccc}", BS + "toprule",
             "& & Succ. & Rec. & Path & Path, succ. & DTW & DTW, succ. " + BS + BS, BS + "midrule"]
    for t, tname in [("lift", "Lift"), ("can", "Can")]:
        f = d[(d.task == t) & (d.kind == "fid")]
        off = d[(d.task == t) & (d.kind == "off")]
        for i, (m, name) in enumerate(ROWS):
            g = f[f.method == m]
            per = g.groupby("held_out").agg(s=("success", "mean"), r=("rms", "mean"), w=("dtw", "mean"))
            gs = g[g.success == 1].groupby("held_out").agg(r=("rms", "mean"), w=("dtw", "mean"))
            if m in ENS:
                rec = e[(e.task == t) & (e.method == ENS[m])].groupby("held_out").recovery.mean().mean()
            elif (off.method == m).any():
                rec = off[off.method == m].groupby("held_out").success.mean().mean()
            else:
                rec = np.nan
            cells = [f"{100 * per.s.mean():.0f}", "--" if np.isnan(rec) else f"{100 * rec:.0f}",
                     f"{100 * per.r.mean():.1f}", f"{100 * gs.r.mean():.1f}", f"{100 * per.w.mean():.1f}",
                     f"{100 * gs.w.mean():.1f}"]
            lines.append((tname if i == 0 else "") + f" & {name} & " + " & ".join(cells) + " " + BS + BS)
        lines.append(BS + "midrule" if t == "lift" else BS + "bottomrule")
    lines.append(BS + "end{tabular}")
    out = os.path.join(ROOT, "paper", "tables", "fid.tex")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
