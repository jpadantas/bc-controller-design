"""Writes paper/tables/v3_median.tex: median and interquartile range over seeds of the distance to the
demonstrator (common starts) and of retention, for the main methods, so heavy tails behind the means of Table 1
are visible.

    python make_median_table.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402

ROWS = [("BC", "BC"), ("Switch", "Error switch"), ("Switch-support", "Support switch"), ("Filter", "Look-ahead switch"),
        ("Backup", "Backup filter"), ("Blend-box", "Blend, box"), ("DAgger-L", "Blend, DAgger-L"),
        ("Residual-box", "Residual blend"), ("Labeler", "Labeler alone")]
COLS = [("quad3d", 3, 1.0), ("quad3d", 30, 1.0), ("reacher", 3, 100.0), ("reacher", 30, 100.0), ("pendulum", 30, 180 / np.pi)]
BS = "\\"


def cell(v, fmt):
    q1, m, q3 = np.percentile(v, [25, 50, 75])
    return f"{fmt.format(m)} [{fmt.format(q1)}, {fmt.format(q3)}]"


def main():
    data = {}
    for dn in sorted({c[0] for c in COLS}):
        data[dn] = A.per_seed(A.load(dn))
    head = " & ".join({"quad3d": "Quadrotor", "reacher": "Reacher", "pendulum": "Pendulum"}[d] + f", {e} ep." for d, e, _ in COLS)
    lines = [BS + "begin{tabular}{l" + "c" * len(COLS) + "}", BS + "toprule", "Method & " + head + " " + BS + BS, BS + "midrule"]
    for m, name in ROWS:
        cells = []
        for dn, e, sc in COLS:
            t = data[dn]
            v = t[(t.method == m) & (t.demo_eps == e)].fid_common.dropna().values * sc
            fmt = "{:.2f}" if dn != "pendulum" else "{:.1f}"
            cells.append(cell(v, fmt) if len(v) else "--")
        lines.append(name + " & " + " & ".join(cells) + " " + BS + BS)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    out = os.path.join(A.ROOT, "paper", "tables", "v3_median.tex")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
