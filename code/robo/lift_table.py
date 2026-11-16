"""Writes paper/tables/lift.tex from outputs/robo/lift/*.csv (lift_study.py).

Six held-out operators x three training seeds; values are means over units. Bold / dagger: better / worse
than the error switch for every one of the six held-out operators (sign test, p = 1/64).

    python lift_table.py
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAB = os.path.join(ROOT, "paper", "tables")
ROWS = [("BC", "BC"), ("Switch", "Error switch"), ("Support", "Support switch"), ("Blend-box", "Blend, box"),
        ("Residual-box", "Residual blend"), ("Labeler", "Labeler alone")]
COLS = [("recovery", "Rec.", 100, "{:.0f}", "higher"), ("retention", "Ret.", 100, "{:.0f}", "higher"),
        ("nom_success", "Succ.", 100, "{:.0f}", "higher"), ("fid_dist", "Dist.", 100, "{:.1f}", "lower"),
        ("act_mse", "Act.", 100, "{:.1f}", "lower")]
BS = "\\"


def main():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, "outputs", "robo", os.environ.get("LIFT_OUT", "lift_big"), "*.csv"))])
    u = d.groupby(["method", "held_out"])[[c for c, *_ in COLS]].mean()
    agg = d.groupby("method")[[c for c, *_ in COLS]].mean()
    lines = [BS + "begin{tabular}{lccccc}", BS + "toprule",
             "Method & " + " & ".join(h for _, h, *_ in COLS) + " " + BS + BS, BS + "midrule"]
    for m, name in ROWS:
        if m == "Labeler":
            lines.append(BS + "midrule")
        cells = []
        for c, _, sc, fmt, better in COLS:
            t = fmt.format(sc * agg.loc[m, c])
            if m not in ("Switch", "Labeler"):
                diff = (u.loc[m, c] - u.loc["Switch", c]).values
                good = diff > 0 if better == "higher" else diff < 0
                bad = diff < 0 if better == "higher" else diff > 0
                if good.all():
                    t = BS + "textbf{" + t + "}"
                elif bad.all():
                    t = t + "$^" + BS + "dagger$"
            cells.append(t)
        lines.append(name + " & " + " & ".join(cells) + " " + BS + BS)
    ref = os.path.join(ROOT, "outputs", "robo", "between_operators.csv")
    if os.path.exists(ref):
        b = pd.read_csv(ref)
        lines.append("Between operators & & & & " + "{:.1f}".format(100 * b.between.mean()) + " & " + BS + BS)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    with open(os.path.join(TAB, "lift.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
