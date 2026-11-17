"""Writes paper/tables/lift.tex from outputs/robo/<LIFT_OUT>{,_la,_chunk}/*.csv (lift_study.py,
lift_lookahead.py, lift_chunk.py) and outputs/robo/between_operators.csv (lift_between.py).

Six held-out operators x training seeds; values are means over units. Bold / dagger: better / worse than the
error switch built on the same BC for every one of the six held-out operators (two-sided sign test, p = 1/32).
Rows whose result files are missing are left out.

    python lift_table.py
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAB = os.path.join(ROOT, "paper", "tables")
OUT = os.environ.get("LIFT_OUT", "lift_big")
# (block, method, row name, reference for the marks); a block of None draws a rule before the row
ROWS = [("", "BC", "BC", "Switch"), ("", "Switch", "Error switch", None), ("", "Support", "Support switch", "Switch"),
        ("_la", "Look-ahead", "Look-ahead switch", "Switch"), ("", "Blend-box", "Blend, box", "Switch"),
        ("", "Residual-box", "Residual blend", "Switch"), (None,),
        ("_chunk", "BC-chunk", "BC, chunked", "Switch-chunk"), ("_chunk", "Switch-chunk", r"\quad + error switch", None),
        ("_chunk", "Support-chunk", r"\quad + support switch", "Switch-chunk"), (None,),
        ("", "Labeler", "Labeler alone", None)]
COLS = [("recovery", "Rec.", 100, "{:.0f}", "higher"), ("retention", "Ret.", 100, "{:.0f}", "higher"),
        ("nom_success", "Succ.", 100, "{:.0f}", "higher"), ("fid_dist", "Dist.", 100, "{:.1f}", "lower"),
        ("act_mse", "Act.", 100, "{:.1f}", "lower")]
BS = "\\"


def load(suffix):
    fs = glob.glob(os.path.join(ROOT, "outputs", "robo", OUT + suffix, "*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs]) if fs else None


def main():
    blocks = {s: load(s) for s in ("", "_la", "_chunk")}
    base = blocks[""]
    seeds = sorted(base.seed.unique())
    if os.environ.get("LIFT_SEEDS"):   # e.g. LIFT_SEEDS=0,1,2 while later seeds are still running
        seeds = [int(x) for x in os.environ["LIFT_SEEDS"].split(",")]
        base = blocks[""] = base[base.seed.isin(seeds)]
    for s in ("_la", "_chunk"):        # same units as the base study, so every row averages the same units
        if blocks[s] is not None:
            blocks[s] = blocks[s][blocks[s].seed.isin(seeds)]
            if blocks[s].groupby("method").size().min() < len(seeds) * 6:
                print("incomplete block", s, "- left out")
                blocks[s] = None
    blocks["_la"] = None if blocks["_la"] is None else pd.concat([blocks["_la"], base])
    lines = [BS + "begin{tabular}{lccccc}", BS + "toprule",
             "Method & " + " & ".join(h for _, h, *_ in COLS) + " " + BS + BS, BS + "midrule"]
    for row in ROWS:
        if row[0] is None:
            lines.append(BS + "midrule")
            continue
        suf, m, name, ref = row
        d = blocks[suf]
        if d is None or m not in set(d.method):
            continue
        u = d.groupby(["method", "held_out"]).mean(numeric_only=True)
        cells = []
        for c, _, sc, fmt, better in COLS:
            if c not in d.columns or d.loc[d.method == m, c].isna().all():
                cells.append("--")
                continue
            t = fmt.format(sc * d.loc[d.method == m, c].mean())
            if ref is not None:
                diff = (u.loc[m, c] - u.loc[ref, c]).values
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
        lines.append("Between operators & & & & " + "{:.1f} ({:.1f})".format(100 * b.between.mean(), 100 * b.within.mean()) + " & " + BS + BS)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    with open(os.path.join(TAB, "lift.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines), "\nseeds", seeds)


if __name__ == "__main__":
    main()
