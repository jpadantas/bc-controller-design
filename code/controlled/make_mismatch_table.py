"""Tables for the mismatched-labeler experiment (labeler replaced by tanh(G * labeler)).

    python make_mismatch_table.py
Writes paper/tables/v3_mismatch.tex (main text: retention / distance) and
paper/tables/v3_mismatch_app.tex (appendix: recovery / retention / distance, every setting).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402

TAB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "paper", "tables")
ROWS = [("BC", "BC"), ("Switch", "Error switch"), ("Switch-support", "Support switch"), ("Simplex", "Simplex"),
        ("Filter", "Look-ahead switch"), ("Backup", "Backup filter"), ("Blend-box", "Blend, box"),
        ("DAgger-L", "Blend, DAgger-L"), ("Residual-box", "Residual blend"), ("Labeler", "Labeler alone")]
BS = "\\"
NL = BS + BS


def stats(dn, e):
    d = A.load(dn)
    s = A.per_seed(d)
    t = s[s.demo_eps == e]
    rec = d[(d.demo_eps == e) & (d.radius > 0)].groupby("method").success.mean() * 100
    g = t.groupby("method")[["kept_inner", "fid_common"]].mean()
    scale = 100.0 if dn.startswith("reacher") else 1.0
    return {m: (rec.get(m, np.nan), 100 * g.loc[m, "kept_inner"], scale * g.loc[m, "fid_common"]) for m in g.index}


def table(settings, cols, name, heads):
    S = [stats(dn, e) for dn, e in settings]
    nc = len(cols)
    lines = [BS + "begin{tabular}{l" + "c" * (nc * len(S)) + "}", BS + "toprule",
             " & " + " & ".join(BS + f"multicolumn{{{nc}}}{{c}}{{{h}}}" for h in heads) + " " + NL,
             "".join(BS + f"cmidrule(lr){{{2 + i * nc}-{1 + (i + 1) * nc}}}" for i in range(len(S))),
             "Method & " + " & ".join(["Rec.", "Ret.", "Dist."][c] for _ in S for c in cols) + " " + NL, BS + "midrule"]
    for m, lab in ROWS:
        cells = []
        for st in S:
            v = st.get(m)
            for c in cols:
                cells.append("--" if v is None else (f"{v[c]:.0f}" if c < 2 else f"{v[c]:.2f}"))
        if m == "Labeler":
            lines.append(BS + "midrule")
        lines.append(lab + " & " + " & ".join(cells) + " " + NL)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    with open(os.path.join(TAB, name), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    table([("quad3d_hg4", 3), ("quad3d_hg16", 3), ("reacher_hg4", 3), ("reacher_hg4", 30)], [1, 2], "v3_mismatch.tex",
          ["Quad., $G{=}4$", "Quad., $G{=}16$", "Reacher, 3 ep.", "Reacher, 30 ep."])
    table([("quad3d_hg4", 3), ("quad3d_hg4", 5), ("quad3d_hg4", 10), ("quad3d_hg16", 3), ("quad3d_hg16", 5),
           ("quad3d_hg16", 10)], [0, 1, 2], "v3_mismatch_app_quad.tex",
          ["$G{=}4$, 3 ep.", "$G{=}4$, 5 ep.", "$G{=}4$, 10 ep.", "$G{=}16$, 3 ep.", "$G{=}16$, 5 ep.", "$G{=}16$, 10 ep."])
    table([("reacher_hg4", 3), ("reacher_hg4", 10), ("reacher_hg4", 30), ("reacher_hg16", 3), ("reacher_hg16", 10),
           ("reacher_hg16", 30)], [0, 1, 2], "v3_mismatch_app_reacher.tex",
          ["$G{=}4$, 3 ep.", "$G{=}4$, 10 ep.", "$G{=}4$, 30 ep.", "$G{=}16$, 3 ep.", "$G{=}16$, 10 ep.", "$G{=}16$, 30 ep."])
