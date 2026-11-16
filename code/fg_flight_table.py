"""Writes paper/tables/v3_flight.tex from the v3 FlightGear rollouts (uses fg_analyze_v3).

Standard starts: 5 held-out pilots x 5 starts, learned policies averaged over 3 seeds
(25 paired units); bold / dagger: significantly better / worse than the Switch
(paired Wilcoxon, Holm across policies). Style columns: 1-Wasserstein distance to
the held-out pilot's first 120 s (nominal start only), with the between-pilot value.
Probe starts (seed 0): attitude upset at the slot, and a 6,000 ft displacement.

    python fg_flight_table.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fg_analyze_v3 as F  # noqa: E402

BS = "\\"
NL = BS + BS


def main():
    log = F.pilot_log()
    rall = F.per_rollout(log)
    r = rall[rall.kind == "std"].copy()
    ex = rall[rall.kind != "std"].copy()
    cols = ["crash", "e60", "e120", "w2", "w5", "w10", "max_bank", "alt_loss", "du_peak"]
    # pilot-level units (the five starts of a held-out pilot share its trained policies): a method is
    # marked when it is better (bold) or worse (dagger) than the error switch for every one of the five
    # held-out pilots (one-sided sign test p = 1/32); Wilcoxon over 25 (pilot, start) units printed too
    u = r.groupby(["method", "pilot", "start"])[cols].mean().reset_index()
    up = r.groupby(["method", "pilot"])[cols].mean()
    sig, pvals = {}, {}
    for k, better in [("crash", "lower"), ("w5", "higher"), ("e120", "lower")]:
        for m in [m for m, _ in F.METHODS if m not in ("Switch",) and m in set(u.method)]:
            d = (up.loc[m, k] - up.loc["Switch", k]).values
            nb = int(np.sum(d < 0)) if better == "lower" else int(np.sum(d > 0))
            nw = int(np.sum(d > 0)) if better == "lower" else int(np.sum(d < 0))
            pvals[(m, k)] = (nb, nw)
            if m != "Labeler":
                if nb == 5:
                    sig[(m, k)] = True
                elif nw == 5:
                    sig[(m, k)] = False
    nom = r[r.start == "0,0"].groupby("method")[["w_bank", "w_g", "w_ailrate", "dist"]].mean()
    import glob
    fid = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(F.EXP, "fidelity_*.csv"))])
    fid = fid.groupby("method")["mse6" if "mse6" in fid and fid.mse6.notna().any() else "mse4"].mean()
    agg = r.groupby("method")[cols].mean()
    n = r.groupby("method").size()
    spread = F.inter_pilot_spread(log)
    exg = ex.groupby(["kind", "method"]).agg(crash=("crash", "sum"), n=("crash", "size"), w5=("w5", "mean")) \
        if len(ex) else None

    pd.set_option("display.width", 250)
    print("rollouts:", n.to_dict())
    print(agg.round(3).to_string())
    print(nom.round(3).to_string())
    print("between pilots:", spread.round(3).to_dict())
    print("action mse6:", fid.round(4).to_dict())
    print("pilots better/worse than switch:", pvals)
    if exg is not None:
        print(exg.to_string())
    folds = r.groupby(["method", "pilot"])[["w5", "crash"]].mean().unstack(0)
    print("fold-level w5:\n", folds["w5"].round(2).to_string())
    print("fold-level crash:\n", folds["crash"].round(2).to_string())

    def cell(m, k, v, fmt):
        t = fmt.format(v)
        if (m, k) in sig:
            t = (BS + "textbf{" + t + "}") if sig[(m, k)] else (t + "$^" + BS + "dagger$")
        return t

    lines = [BS + "begin{tabular}{lccccccccc}", BS + "toprule",
             " & " + BS + "multicolumn{3}{c}{All five starts} & " + BS + "multicolumn{4}{c}{Fidelity to the held-out pilot} & "
             + BS + "multicolumn{2}{c}{Probes: crashes} " + NL,
             BS + "cmidrule(lr){2-4}" + BS + "cmidrule(lr){5-8}" + BS + "cmidrule(lr){9-10}",
             "Method & Crash & In 5 kft & Err. 120 s & Track & Action & Bank & Ail. rate & Upset & 6 kft " + NL, BS + "midrule"]
    for m, name in F.METHODS:
        if m not in agg.index:
            continue
        a = agg.loc[m]
        if m == "Labeler":
            lines.append(BS + "midrule")
        row = [name, cell(m, "crash", 100 * a.crash, "{:.0f}"), cell(m, "w5", 100 * a.w5, "{:.0f}"),
               cell(m, "e120", a.e120 / 1000, "{:.1f}"),
               f"{nom.loc[m, 'dist'] / 1000:.1f}" if m in nom.index else "--",
               f"{100 * fid[m]:.1f}" if m in fid.index else "--",
               f"{nom.loc[m, 'w_bank']:.0f}" if m in nom.index else "--",
               f"{1000 * nom.loc[m, 'w_ailrate']:.0f}" if m in nom.index else "--"]
        for kind in ["upset", "near"]:
            row.append(f"{int(exg.loc[(kind, m), 'crash'])}/{int(exg.loc[(kind, m), 'n'])}"
                       if exg is not None and (kind, m) in exg.index else "--")
        lines.append(" & ".join(row) + " " + NL)
    lines.append("Between pilots & & & & " + f"{spread.dist / 1000:.1f} & & " + f"{spread.w_bank:.0f} & {1000 * spread.w_ailrate:.0f} & & " + NL)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    with open(os.path.join(F.TAB, os.environ.get("FLIGHT_TAB", "v3_flight.tex")), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
