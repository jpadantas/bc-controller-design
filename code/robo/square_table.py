"""Table of the Square confirmatory study (outputs/robo/square/square_rollouts.csv -> paper/tables/square.tex),
and the scoring of its predictions S1-S6 (paper/notes/prediction_square_2026-11-18.md).

Means over six held-out operators (per operator: mean over three seeds; the labeler: one evaluation per operator).

    python square_table.py
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BS = "\\"
ROWS = [("BC", "BC"), ("Switch", "Error switch"), ("Support", "Support switch"), ("Ensemble", "Ensemble switch"),
        ("Residual-box", "Residual blend"), ("Labeler", "Labeler alone")]


def per_operator(d):
    """Per (held-out operator, method): nominal success and retention, recovery, success from the operator's starts,
    RMS and DTW gaps over all and over successful rollouts, share of steps in which BC acts (nominal starts)."""
    out = {}
    for (h, m), g in d.groupby(["held_out", "method"]):
        nom, off, fid = g[g.kind == "nom"], g[g.kind == "off"], g[g.kind == "fid"]
        fs = fid[fid.success == 1]
        out[(h, m)] = dict(succ=nom.success.mean(), ret=nom.kept.mean(), rec=off.success.mean(),
                           fsucc=fid.success.mean(), path=fid.rms.mean(), path_s=fs.rms.mean(),
                           dtw=fid.dtw.mean(), dtw_s=fs.dtw.mean(), bc=nom.bc_frac.mean())
    return pd.DataFrame(out).T.rename_axis(["held_out", "method"]).astype(float)


def score(op, d):
    m = lambda k: op.xs(k, level="method")  # noqa: E731
    bc, sup, ens, res, lab = m("BC"), m("Support"), m("Ensemble"), m("Residual-box"), m("Labeler")
    print("S1 BC nominal success per operator:", (100 * bc.succ).round(0).to_dict())
    print("   holds" if (bc.succ < 0.5).all() else "   fails", f"(mean {100 * bc.succ.mean():.0f}%)")
    gain = 100 * (sup.succ - bc.succ)
    print("S2 support - BC (points):", gain.round(1).to_dict(), "holds" if (gain < 10).all() else "fails")
    g3 = 100 * (ens.succ - bc.succ)
    n3 = int((g3 >= 20).sum())
    print(f"S3 ensemble - BC >= 20 points for {n3}/6; recovery ensemble {100 * ens.rec.mean():.0f}% vs support "
          f"{100 * sup.rec.mean():.0f}% ->", "holds" if n3 >= 5 and ens.rec.mean() >= 2 * sup.rec.mean() else "fails")
    n4 = int((ens.path_s < lab.path_s).sum())
    r4 = ens.dtw_s.mean() / lab.dtw_s.mean()
    print(f"S4 ensemble closer than labeler (RMS, successful) for {n4}/6 "
          f"({100 * ens.path_s.mean():.1f} vs {100 * lab.path_s.mean():.1f} cm); DTW ratio {r4:.3f} ->",
          "holds" if n4 >= 4 and abs(r4 - 1) <= 0.10 else "fails")
    n5 = int((res.rec < ens.rec).sum())
    print(f"S5 residual recovers less than ensemble for {n5}/6 ->", "holds" if n5 >= 5 else "fails")
    # the frozen rule, per unit (held-out operator, seed), on the nominal resets
    nom = d[d.kind == "nom"]
    kept = nom.groupby(["held_out", "seed", "method"]).kept.min()
    picks = {}
    for (h, s) in nom[["held_out", "seed"]].drop_duplicates().itertuples(index=False):
        if kept.get((h, s, "BC"), 0) == 1:
            picks[(h, s)] = "error switch"
        elif kept.get((h, s, "Residual-box"), 0) == 1:
            picks[(h, s)] = "residual blend"
        else:
            picks[(h, s)] = "full-state monitor"
    n6 = sum(v == "full-state monitor" for v in picks.values())
    print(f"S6 frozen rule picks the full-state monitor for {n6}/{len(picks)} ->", "holds" if n6 >= 15 else "fails")


def main():
    d = pd.read_csv(os.path.join(ROOT, "outputs", "robo", "square", "square_rollouts.csv"))
    op = per_operator(d)
    lines = [BS + "begin{tabular}{lcccccccc}", BS + "toprule",
             "& Succ. & Rec. & Ret. & Succ., op. & Path & Path, succ. & DTW, succ. & BC acts " + BS + BS,
             BS + "midrule"]
    for m, name in ROWS:
        r = op.xs(m, level="method").mean()
        cells = [f"{100 * r.succ:.0f}", f"{100 * r.rec:.0f}", f"{100 * r.ret:.0f}", f"{100 * r.fsucc:.0f}",
                 f"{100 * r.path:.1f}", f"{100 * r.path_s:.1f}", f"{100 * r.dtw_s:.1f}",
                 "--" if np.isnan(r.bc) else f"{100 * r.bc:.0f}"]
        lines.append(f"{name} & " + " & ".join(cells) + " " + BS + BS)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    out = os.path.join(ROOT, "paper", "tables", "square.tex")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print((100 * op).round(1).to_string())
    score(op, d)


if __name__ == "__main__":
    main()
