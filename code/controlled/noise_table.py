"""Table of the imperfect-plant runs (run_noise.py -> paper/tables/noise.tex).

For each method and plant (nominal, noise, mismatch): recovery and retention in the worst of the five settings
(the lowest per-setting mean over ten seeds), and the mean over settings.

    python noise_table.py
"""
import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETTINGS = [("quad3d", 3), ("quad3d", 10), ("reacher", 3), ("reacher", 10), ("pendulum", 30)]
ROWS = [("Switch", "Error switch"), ("Simplex", "Simplex"), ("Switch-support", "Support switch"),
        ("Filter", "Look-ahead switch"), ("Backup", "Backup filter"), ("Blend-box", "Box blend"),
        ("DAgger-L", "DAgger-L"), ("Residual-box", "Residual blend"), ("Labeler", "Labeler alone")]
BS = "\\"


def rows():
    out = []
    for cond in ("nominal", "noise", "mismatch"):
        if cond == "nominal":
            files = [os.path.join(ROOT, "outputs", "v3", d, f"eps{e}", f"seed{s}.csv") for d, e in SETTINGS
                     for s in range(10)]
        else:
            files = glob.glob(os.path.join(ROOT, "outputs", "v3", f"noise_{cond}", "only_*", "*", "eps*", "seed*.csv"))
        for f in files:
            d = pd.read_csv(f)
            rec = d[d.radius > 0].groupby("method").success.mean()
            one = d.groupby("method").kept_inner.first()
            for m in one.index:
                out.append(dict(cond=cond, setting=(d.domain.iloc[0], int(d.demo_eps.iloc[0])), method=m,
                                rec=rec[m], ret=one[m]))
    return pd.DataFrame(out)


def main():
    r = rows()
    per = r.groupby(["cond", "setting", "method"])[["rec", "ret"]].mean()
    worst = per.groupby(["cond", "method"]).min()
    lines = [BS + "begin{tabular}{lcccccc}", BS + "toprule",
             "& " + " & ".join(BS + "multicolumn{2}{c}{" + c + "}" for c in ("Nominal plant", "Noise", "Mismatch"))
             + " " + BS + BS,
             "& Rec. & Ret. & Rec. & Ret. & Rec. & Ret. " + BS + BS, BS + "midrule"]
    for m, name in ROWS:
        cells = []
        for c in ("nominal", "noise", "mismatch"):
            w = worst.loc[(c, m)]
            cells += [f"{100 * w.rec:.0f}", f"{100 * w.ret:.0f}"]
        lines.append(f"{name} & " + " & ".join(cells) + " " + BS + BS)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    out = os.path.join(ROOT, "paper", "tables", "noise.tex")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(per.unstack("cond").round(2).to_string())


if __name__ == "__main__":
    main()
