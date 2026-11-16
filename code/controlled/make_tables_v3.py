"""LaTeX tables for the v3 paper (generated from outputs/v3).

tables/v3_main.tex -- quadrotor at 3 and 30 demonstration episodes and pendulum (30):
recovery, nominal retention, peak action step, closed-loop distance to the demonstrator.
Bold / dagger: significantly better / worse than the Switch (paired Wilcoxon over
seeds, Holm across methods within each column).
    python make_tables_v3.py
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402

TAB = os.path.join(A.ROOT, "paper", "tables")
BS = "\\"
NL = BS + BS
ROWS = [("BC", "BC"), ("Switch", "Error switch"), ("Switch-DART", "DART switch"), ("Switch-support", "Support switch"),
        ("Simplex", "Simplex"), ("Filter", "Look-ahead switch"), ("Backup", "Backup filter"), ("SoftGate", "Soft gate"),
        ("Blend-part", "Blend, box-partial"), ("Blend-box", "Blend, box"), ("Blend-tube", "Blend, tube"),
        ("DAgger-L", "Blend, DAgger-L"), ("Residual", "Residual, no zero targets"), ("Residual-box", "Residual blend"),
        ("Labeler", "Labeler alone"), ("Demonstrator", "Demonstrator")]
ROWS_APP = [("DART", "DART")] + ROWS
COLS = [("recover", 100, "higher", "{:.0f}"), ("kept_inner", 100, "higher", "{:.0f}"), ("fid_common", 1, "lower", "{:.2f}")]
COLS_APP = [("kept", 100, "higher", "{:.0f}"), ("fid_kept", 1, "lower", "{:.2f}"), ("fid_a", 1000, "lower", "{:.3g}")]
COLS_DU = [("du", 1, "lower", "{:.2f}")]


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    out, run = np.empty_like(p), 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        out[j] = run
    return out


def block(s, fid_scale=1.0, fid_fmt="{:.2f}", cols=None):
    """cells[method][col] for one (domain, budget) frame of per-seed results."""
    ref = s[s.method == "Switch"].set_index("seed")
    cells = {}
    for k, sc, better, fmt in (cols or COLS):
        ms = [m for m, _ in ROWS_APP if m != "Switch" and m in set(s.method)]
        ps, sg = [], []
        for m in ms:
            x = s[s.method == m].set_index("seed")
            i = x.index.intersection(ref.index)
            d = (x.loc[i, k] - ref.loc[i, k]).dropna().values
            ps.append(stats.wilcoxon(d).pvalue if len(d) > 5 and np.any(d != 0) else 1.0)
            sg.append(np.median(d) if np.any(d != 0) else 0.0)
        adj = dict(zip(ms, holm(ps)))
        med = dict(zip(ms, sg))
        for m, _ in ROWS_APP:
            g = s[s.method == m][k] * (fid_scale if k in ("fid", "fid_kept", "fid_common") else sc)
            if g.empty:
                continue
            g = g.dropna()
            if g.empty:
                cells.setdefault(m, {})[k] = "--"
                continue
            txt = (fid_fmt if k in ("fid", "fid_kept", "fid_common") else fmt).format(g.mean())
            if m in adj and adj[m] < 0.05 and m not in ("Labeler", "Demonstrator"):
                good = med[m] > 0 if better == "higher" else med[m] < 0
                txt = (BS + "textbf{" + txt + "}") if good else (txt + "$^" + BS + "dagger$")
            cells.setdefault(m, {})[k] = txt
    return cells


def write(name, blocks, cols, rows=None):
    nc = len(cols)
    head1 = "".join(" & " + BS + f"multicolumn{{{nc}}}{{c}}{{{t}}}" for t, _ in blocks)
    sub = {"recover": "Rec.", "nominal_ok": "Ret.", "kept": "Ret.$_{" + BS + "mathrm{all}}$", "fid": "Dist.", "fid_kept": "Dist.$_{" + BS + "mathrm{ret}}$",
           "du": "$|" + BS + "Delta a|$", "time_in": "In $" + BS + "mathcal{N}$", "fid_a": "Act.",
           "fid_common": "Dist.", "kept_inner": "Ret."}
    head2 = "Method" + "".join(" & " + sub[k] for k, *_ in cols) * len(blocks)
    lines = [BS + "begin{tabular}{l" + "c" * nc * len(blocks) + "}", BS + "toprule", head1 + " " + NL,
             "".join(BS + f"cmidrule(lr){{{2 + nc * i}-{1 + nc * (i + 1)}}}" for i in range(len(blocks))),
             head2 + " " + NL, BS + "midrule"]
    for m, mname in (rows or ROWS):
        if m == "Labeler":
            lines.append(BS + "midrule")
        row = [mname]
        for _, c in blocks:
            row += [c.get(m, {}).get(k, "--") for k, *_ in cols]
        lines.append(" & ".join(row) + " " + NL)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    with open(os.path.join(TAB, name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    q = A.per_seed(A.load("quad3d"))
    r = A.per_seed(A.load("reacher"))
    p = A.per_seed(A.load("pendulum"))
    spec = [("Quadrotor, 3 ep.", q[q.demo_eps == 3], 1.0, "{:.2f}"),
            ("Reacher, 3 ep.", r[r.demo_eps == 3], 100.0, "{:.1f}"),
            ("Quadrotor, 30 ep.", q[q.demo_eps == 30], 1.0, "{:.2f}"),
            ("Reacher, 30 ep.", r[r.demo_eps == 30], 100.0, "{:.1f}"),
            ("Pendulum, 30 ep.", p[p.demo_eps == 30], float(np.degrees(1.0)), "{:.1f}")]
    write("v3_main.tex", [(t, block(d, sc, fm)) for t, d, sc, fm in spec], COLS)
    write("v3_smooth.tex", [(t, block(d, sc, fm, COLS_DU)) for t, d, sc, fm in spec], COLS_DU, ROWS_APP)
    write("v3_app.tex", [(t, block(d, sc, fm, COLS_APP)) for t, d, sc, fm in spec], COLS_APP, ROWS_APP)


if __name__ == "__main__":
    main()
