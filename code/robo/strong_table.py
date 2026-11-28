"""Table of the stronger-policy study (outputs/robo/strong/strong_rollouts.csv -> paper/tables/strong.tex),
and the scoring of its predictions G1-G5 (paper/notes/prediction_strong_2026-11-23.md).

Means over six held-out operators (per operator: mean over three seeds). The MLP and labeler rows come from the
ensemble study (Lift, Can: outputs/robo/ens/ens_results.csv) and the Square study (outputs/robo/square).

    python strong_table.py
"""
import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BS = "\\"
TASKS = ["lift", "can", "square"]


def load():
    """Registered run (strong) and the revised diffusion recipe (strong_v2, class renamed DiffusionR)."""
    out = []
    for run in ("strong", "strong_v2"):
        f = os.path.join(ROOT, "outputs", "robo", run, "strong_rollouts.csv")
        d = pd.read_csv(f) if os.path.exists(f) else pd.concat(
            pd.read_csv(p) for p in glob.glob(os.path.join(ROOT, "outputs", "robo", run + "_parts", "*.csv")))
        if run == "strong_v2":
            d["cls"] = "DiffusionR"
        out.append(d)
    return pd.concat(out)


def per_operator(d):
    """Per (task, held-out operator, policy): nominal success, recovery, RMS gap (all and successful rollouts from
    the operator's starts), share of steps in which the learned policy acts."""
    out = {}
    for (t, h, c, g), x in d.groupby(["task", "held_out", "cls", "gate"]):
        nom, off, fid = x[x.kind == "nom"], x[x.kind == "off"], x[x.kind == "fid"]
        out[(t, h, f"{c}/{g}")] = dict(succ=nom.success.mean(), rec=off.success.mean(), path=fid.rms.mean(),
                                       path_s=fid[fid.success == 1].rms.mean(), bc=nom.bc_frac.mean())
    return pd.DataFrame(out).T.rename_axis(["task", "held_out", "pol"]).astype(float)


def references():
    """Main MLP BC and the labeler alone, same per-operator quantities."""
    out = {}
    e = pd.read_csv(os.path.join(ROOT, "outputs", "robo", "ens", "ens_results.csv"))
    for (t, h, m), x in e[e.method.isin(["BC", "Labeler", "Ensemble"])].groupby(["task", "held_out", "method"]):
        out[(t, h, m)] = dict(succ=x.nom_success.mean(), rec=x.recovery.mean(), path=x.fid_dist.mean(),
                              path_s=float("nan"), bc=float("nan"))
    s = pd.read_csv(os.path.join(ROOT, "outputs", "robo", "square", "square_rollouts.csv"))
    for (h, m), x in s[s.method.isin(["BC", "Labeler", "Ensemble"])].groupby(["held_out", "method"]):
        fid = x[x.kind == "fid"]
        out[("square", h, m)] = dict(succ=x[x.kind == "nom"].success.mean(), rec=x[x.kind == "off"].success.mean(),
                                     path=fid.rms.mean(), path_s=fid[fid.success == 1].rms.mean(), bc=float("nan"))
    return pd.DataFrame(out).T.rename_axis(["task", "held_out", "pol"]).astype(float)


def score(op):
    v = lambda t, p, k: op.xs((t, p), level=["task", "pol"])[k]  # noqa: E731
    print("G1 diffusion alone vs MLP BC, nominal success:")
    ok1 = True
    for t in ["can", "square"]:
        a, b = v(t, "Diffusion/alone", "succ").mean(), v(t, "BC", "succ").mean()
        ok1 &= a > b
        print(f"   {t}: {100 * a:.0f} vs {100 * b:.0f}")
    print("   holds" if ok1 else "   fails")
    print("G2 support - alone, nominal success (points):")
    ok2 = True
    for t in ["can", "square"]:
        for c in ["GMM", "Diffusion"]:
            gain = 100 * (v(t, f"{c}/Support", "succ").mean() - v(t, f"{c}/alone", "succ").mean())
            ok2 &= gain < 10
            print(f"   {t} {c}: {gain:+.1f}")
    print("   holds" if ok2 else "   fails")
    print("G3 uncertainty switch recovery vs class alone:")
    ok3 = True
    for t in TASKS:
        for c in ["GMM", "Diffusion"]:
            u, a = v(t, f"{c}/Uncertainty", "rec").mean(), v(t, f"{c}/alone", "rec").mean()
            ok3 &= u >= 0.5 and u >= 2 * a
            print(f"   {t} {c}: {100 * u:.0f} vs {100 * a:.0f}")
    print("   holds" if ok3 else "   fails")
    lab = v("lift", "Labeler", "path")
    ok4 = True
    for c in ["GMM", "Diffusion"]:
        n = int((v("lift", f"{c}/Support", "path") < lab).sum())
        ok4 &= n >= 5
        print(f"G4 lift {c}+support closer than labeler for {n}/6 "
              f"({100 * v('lift', f'{c}/Support', 'path').mean():.1f} vs {100 * lab.mean():.1f} cm)")
    print("   holds" if ok4 else "   fails")
    lab = v("square", "Labeler", "path_s").mean()
    ok5 = True
    for c in ["GMM", "Diffusion"]:
        for g in ["Support", "Uncertainty"]:
            r = v("square", f"{c}/{g}", "path_s").mean() / lab
            ok5 &= r >= 0.9
            print(f"G5 square {c}+{g} RMS on successes / labeler: {r:.3f}")
    print("   holds" if ok5 else "   fails")
    print("--- revised diffusion (prediction_strong_v2_2026-11-27.md)")
    ok = all(v(t, "DiffusionR/alone", "succ").mean() > v(t, "BC", "succ").mean() for t in ["can", "square"])
    print("D1", {t: (round(100 * v(t, "DiffusionR/alone", "succ").mean()), round(100 * v(t, "BC", "succ").mean()))
                 for t in ["can", "square"]}, "holds" if ok else "fails")
    g = {t: round(100 * (v(t, "DiffusionR/Support", "succ").mean() - v(t, "DiffusionR/alone", "succ").mean()), 1)
         for t in ["can", "square"]}
    print("D2", g, "holds" if all(x < 10 for x in g.values()) else "fails")
    g = {t: (round(100 * v(t, "DiffusionR/Uncertainty", "rec").mean()), round(100 * v(t, "Ensemble", "rec").mean()))
         for t in ["can", "square"]}
    print("D3", g, "holds" if all(a < b for a, b in g.values()) else "fails")
    n = int((v("lift", "DiffusionR/Support", "path") < v("lift", "Labeler", "path")).sum())
    print(f"D4 {n}/6", "holds" if n >= 5 else "fails")


ROWS = [("BC", "MLP (main studies)"), ("GMM/alone", "GMM"), ("GMM/Support", "\\quad + support switch"),
        ("GMM/Uncertainty", "\\quad + uncertainty switch"), ("Diffusion/alone", "Diffusion"),
        ("Diffusion/Support", "\\quad + support switch"), ("Diffusion/Uncertainty", "\\quad + uncertainty switch"),
        ("DiffusionR/alone", "Diffusion, revised"), ("DiffusionR/Support", "\\quad + support switch"),
        ("DiffusionR/Uncertainty", "\\quad + uncertainty switch"), ("Ensemble", "MLP + ensemble switch"),
        ("Labeler", "Labeler alone")]


def main():
    op = pd.concat([per_operator(load()), references()])
    score(op)
    mean = op.groupby(["task", "pol"]).mean()
    lines = [BS + "begin{tabular}{lccccccccc}", BS + "toprule",
             "& " + " & ".join(BS + "multicolumn{3}{c}{" + t + "}" for t in ("Lift", "Can", "Square")) + " " + BS + BS,
             "& " + " & ".join(["Succ. & Rec. & Path"] * 3) + " " + BS + BS, BS + "midrule"]
    for p, name in ROWS:
        cells = []
        for t in TASKS:
            r = mean.loc[(t, p)]
            cells += [f"{100 * r.succ:.0f}", f"{100 * r.rec:.0f}", f"{100 * r.path:.1f}"]
        lines.append(f"{name} & " + " & ".join(cells) + " " + BS + BS)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    out = os.path.join(ROOT, "paper", "tables", "strong.tex")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print((100 * mean[["bc", "path_s"]]).round(1).to_string())


if __name__ == "__main__":
    main()
