"""Table of the ensemble-monitor study (outputs/robo/ens/ens_results.csv -> paper/tables/ens.tex).

Means over six held-out operators and three seeds (the labeler: one evaluation per operator); in brackets, the
number of operators for which the ensemble switch is better than the support switch on that measure.

    python ens_table.py
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BS = "\\"
NAMES = {"BC": "BC", "Support": "Support switch", "Ensemble": "Ensemble switch", "Labeler": "Labeler alone"}


def main():
    d = pd.read_csv(os.path.join(ROOT, "outputs", "robo", "ens", "ens_results.csv"))
    lines = [BS + "begin{tabular}{llccccc}", BS + "toprule",
             "Task & Method & Succ. & Rec. & Ret. & Dist. & BC acts " + BS + BS, BS + "midrule"]
    for t, tname in [("lift", "Lift"), ("can", "Can")]:
        x = d[d.task == t]
        op = x.groupby(["held_out", "method"]).mean(numeric_only=True)
        for i, m in enumerate(["BC", "Support", "Ensemble", "Labeler"]):
            r = op.xs(m, level="method").mean()
            cells = [f"{100 * r.nom_success:.0f}", f"{100 * r.recovery:.0f}", f"{100 * r.retention:.0f}",
                     f"{100 * r.fid_dist:.1f}", "--" if m in ("BC", "Labeler") else f"{100 * r.bc_frac:.0f}"]
            if m == "Ensemble":
                e, s = op.xs("Ensemble", level="method"), op.xs("Support", level="method")
                wins = [(e.nom_success > s.nom_success).sum(), (e.recovery > s.recovery).sum(),
                        (e.retention > s.retention).sum(), (e.fid_dist < s.fid_dist).sum()]
                cells[:4] = [f"{c} ({w})" for c, w in zip(cells[:4], wins)]
            lines.append((tname if i == 0 else "") + f" & {NAMES[m]} & " + " & ".join(cells) + " " + BS + BS)
        lines.append(BS + "midrule" if t == "lift" else BS + "bottomrule")
    lines.append(BS + "end{tabular}")
    out = os.path.join(ROOT, "paper", "tables", "ens.tex")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
