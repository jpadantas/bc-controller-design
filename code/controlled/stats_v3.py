"""Every in-text statistic of the v3 paper that is not a table cell.

    python stats_v3.py            -> prints and writes paper/tables/v3_stats.txt
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402

OUT_TXT = os.path.join(A.ROOT, "paper", "tables", "v3_stats.txt")
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def p1():
    say("== P1: rank correlation with the cloned policy's final error, within each radius (variant means)")
    rng = np.random.default_rng(0)
    for dn in ["quad3d", "reacher", "pendulum"]:
        fs = glob.glob(os.path.join(A.OUT, "inherit", dn, "seed*.csv"))
        if not fs:
            continue
        d = pd.concat([pd.read_csv(f) for f in fs])

        def corr(df, col):
            g = df.groupby(["variant", "radius"]).mean(numeric_only=True)
            return np.array([stats.spearmanr(gg[col], gg.policy_final)[0] for _, gg in g.groupby("radius")])
        res = {c: corr(d, c) for c in ["glued_final", "labeler_final", "label_bias"]}
        seeds = d.seed.unique()
        boot = []
        for _ in range(500):   # seed bootstrap of the mean within-radius difference glued - label accuracy
            b = d[d.seed.isin(rng.choice(seeds, len(seeds)))]
            boot.append(np.nanmean(corr(b, "glued_final") - corr(b, "label_bias")))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        say(f"{dn}: glued {res['glued_final'].min():.2f}-{res['glued_final'].max():.2f}; "
            f"labeler alone {res['labeler_final'].min():.2f}-{res['labeler_final'].max():.2f}; "
            f"label MSE {res['label_bias'].min():.2f}-{res['label_bias'].max():.2f}; "
            f"mean(glued - label MSE) 95% bootstrap CI [{lo:.2f}, {hi:.2f}]")
        g = d.groupby("variant")[["labeler_success", "glued_success", "policy_success", "label_bias"]].mean()
        say(g.round(3).to_string())


def p1_box():
    say("== P1 without interaction (box blend / residual blend cloned from each variant's labels)")
    for dn in ["quad3d", "reacher", "pendulum"]:
        fs = glob.glob(os.path.join(A.OUT, "inherit_box", dn, "seed*.csv"))
        if not fs:
            continue
        d = pd.concat([pd.read_csv(f) for f in fs])
        g = d.groupby(["variant", "radius"]).mean(numeric_only=True)
        for pol in ["box", "resid"]:
            out = {}
            for ref in ["glued_final", "labeler_final", "label_bias"]:
                r = [stats.spearmanr(gg[ref], gg[f"{pol}_final"])[0] for _, gg in g.groupby("radius")]
                out[ref] = f"{np.nanmin(r):.2f}-{np.nanmax(r):.2f}"
            say(dn, pol, out)
        say(d.groupby("variant")[["labeler_success", "glued_success", "box_success", "resid_success", "label_bias"]]
            .mean().round(3).to_string())


SETTINGS = [("quad3d", 3), ("quad3d", 5), ("quad3d", 10), ("quad3d", 30), ("reacher", 3), ("reacher", 10),
            ("reacher", 30), ("pendulum", 30)]
COMPETITORS = ["BC", "DART", "Switch", "Switch-DART", "Switch-support", "Simplex", "Filter", "Backup", "SoftGate", "Blend-part", "Blend-box",
               "Blend-tube", "DAgger-L", "Residual", "Labeler"]


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    out, run = np.empty_like(p), 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        out[j] = run
    return out


def pairs():
    """Residual blend against every competitor: paired Wilcoxon over seeds, Holm within a setting and metric."""
    say("== Residual blend vs each competitor (mean difference residual - competitor; Holm-adjusted p)")
    cache = {}
    rows = []
    for dn, e in SETTINGS:
        if dn not in cache:
            cache[dn] = A.per_seed(A.load(dn))
        t = cache[dn][cache[dn].demo_eps == e]
        if t.empty:
            continue
        ref = t[t.method == "Residual-box"].set_index("seed")
        for k in ["kept_inner", "fid_common"]:
            ms, ps, ds = [], [], []
            for m in COMPETITORS:
                x = t[t.method == m].set_index("seed")
                if x.empty:
                    continue
                i = x.index.intersection(ref.index)
                d = (ref.loc[i, k] - x.loc[i, k]).dropna().values
                ms.append(m)
                ds.append(d.mean())
                ps.append(stats.wilcoxon(d).pvalue if len(d) > 5 and np.any(d != 0) else 1.0)
            for m, d, p in zip(ms, ds, holm(ps)):
                rows.append(dict(domain=dn, eps=e, metric=k, method=m, diff=d, p=p))
    df = pd.DataFrame(rows)
    write_pairs(df)
    for (dn, e), g in df.groupby(["domain", "eps"], sort=False):
        say(dn, e, {f"{r.method}:{r.metric[:4]}": (round(r.diff, 4), round(r.p, 3)) for r in g.itertuples()})
    return df


NAMES = {"BC": "BC", "DART": "DART", "Switch": "Error switch", "Switch-DART": "DART switch",
         "Switch-support": "Support switch", "Simplex": "Simplex", "Filter": "Look-ahead switch",
         "Backup": "Backup filter", "SoftGate": "Soft gate", "Blend-part": "Blend, box-partial",
         "Blend-box": "Blend, box", "Blend-tube": "Blend, tube", "DAgger-L": "Blend, DAgger-L",
         "Residual": "Residual, no zero targets", "Labeler": "Labeler alone"}


def write_pairs(df):
    """tables/v3_pairs.tex: the residual blend against each alternative; per setting, retention / distance:
    + residual significantly better, - significantly worse, = no significant difference (Holm, 0.05)."""
    BS = "\\"
    sets = [(dn, e) for dn, e in SETTINGS if ((df.domain == dn) & (df.eps == e)).any()]
    short = {"quad3d": "Q", "reacher": "R", "pendulum": "P"}
    lines = [BS + "begin{tabular}{l" + "c" * len(sets) + "}", BS + "toprule",
             "Alternative & " + " & ".join(f"{short[dn]}{e}" for dn, e in sets) + " " + BS + BS, BS + "midrule"]
    for m in COMPETITORS:
        cells = []
        for dn, e in sets:
            c = []
            for k, better in [("kept_inner", 1), ("fid_common", -1)]:
                r = df[(df.domain == dn) & (df.eps == e) & (df.metric == k) & (df.method == m)]
                if r.empty:
                    c.append("?")
                    continue
                r = r.iloc[0]
                c.append("=" if r.p >= 0.05 else ("+" if np.sign(r["diff"]) == better else "$-$"))
            cells.append("/".join(c))
        lines.append(NAMES[m] + " & " + " & ".join(cells) + " " + BS + BS)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    with open(os.path.join(A.ROOT, "paper", "tables", "v3_pairs.tex"), "w", encoding="utf-8") as fh:
        fh.write(chr(10).join(lines) + chr(10))


def rule():
    """The decision rule applied per seed: use the error switch if the cloned policy never leaves N from
    the nominal starts well inside it (as the labeler does; no demonstrator needed), else the residual
    blend. Scored by the distance to the demonstrator on common starts, as excess over the better of
    the two on that seed, normalized by the labeler's distance in that setting."""
    say("== Decision rule per seed (error switch vs residual blend)")
    for key in ["kept_inner", "kept"]:
        for tau in [0.0, 0.05]:
            hits, n, ex, er, es, per, hr, hs = 0, 0, [], [], [], {}, 0, 0
            for dn, e in SETTINGS:
                s = A.per_seed(A.load(dn))
                t = s[s.demo_eps == e].pivot_table(index="seed", columns="method", values=[key, "fid_common"])
                ok = t[key]["BC"] >= t[key]["Labeler"] - tau - 1e-9
                fs, fr = t["fid_common"]["Switch"], t["fid_common"]["Residual-box"]
                lab = t["fid_common"]["Labeler"].mean()
                ch, b = np.where(ok, fs, fr), np.minimum(fs, fr)
                hits += int((ch <= b + 1e-12).sum()); n += len(t)
                hr += int((fr <= b + 1e-12).sum()); hs += int((fs <= b + 1e-12).sum())
                ex += list((ch - b) / lab); er += list((fr - b) / lab); es += list((fs - b) / lab)
                per[f"{dn}{e}"] = f"switch {int(ok.sum())}/{len(t)}"
            say(f"{key} tau={tau}: rule picks the closer of the two on {hits}/{n} seeds (always residual {hr}, always switch {hs}); mean excess distance "
                f"(/labeler's) rule {np.mean(ex):.3f}, always residual {np.mean(er):.3f}, always switch {np.mean(es):.3f}", per)


MISMATCH = [("quad3d_hg4", 3), ("quad3d_hg4", 5), ("quad3d_hg4", 10), ("quad3d_hg16", 3), ("quad3d_hg16", 5),
            ("quad3d_hg16", 10), ("reacher_hg4", 3), ("reacher_hg4", 10), ("reacher_hg4", 30)]
# reacher_hg16 excluded: tanh(16 k) destabilizes the Reacher labeler itself (46% recovery); see the appendix table.


HELDOUT = [("pendulum", 10), ("pendulum_hg4", 10), ("pendulum_hg4", 30), ("pendulum_q10", 10), ("pendulum_q10", 30),
           ("quad3d_q05", 3), ("quad3d_q05", 5), ("quad3d_q05", 10), ("quad3d_hg4", 30),
           ("reacher_hg2", 3), ("reacher_hg2", 10), ("reacher_hg2", 30)]
ALL16 = ["BC", "DART", "Switch", "Switch-DART", "Switch-support", "SoftGate", "Simplex", "Filter", "Backup",
         "Residual", "Residual-box", "Blend-part", "Blend-box", "Blend-tube", "DAgger-L", "Labeler"]
TOL = 0.05   # a choice is within tolerance if within 5% of the labeler's distance of the closest of the three


RULE_PER = {}   # (decide, group) -> {choice: per-seed excess over the closest of three, / labeler's distance}


def rule_extra(decide="dec", group="held-out", n_boot=10000):
    """Paired comparison of the rule with always using the look-ahead switch (per-seed excess, seed bootstrap),
    and the number of correct choices at other tolerances."""
    if (decide, group) not in RULE_PER:
        rule3(decide)
    e = RULE_PER[(decide, group)]
    rng = np.random.default_rng(0)
    d = e["rule3"] - e["look-ahead"]
    bs = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
    w = stats.wilcoxon(e["rule3"], e["look-ahead"])
    say(f"== Rule vs always look-ahead [{group}, decided on {decide}], n={len(d)}: mean excess "
        f"{e['rule3'].mean():.3f} vs {e['look-ahead'].mean():.3f}; difference {d.mean():.3f} "
        f"(95% bootstrap {np.percentile(bs, 2.5):.3f} to {np.percentile(bs, 97.5):.3f}); Wilcoxon p={w.pvalue:.3g}")
    for tol in (0.0, 0.02, 0.05, 0.10, 0.20):
        say(f"  within {tol:.0%}:", {k: int((v <= tol + 1e-12).sum()) for k, v in e.items()})


def rule3(decide="inner"):
    """The three-outcome rule, frozen (paper/notes/prediction_heldout_2026-11-13.md): error switch if BC never
    leaves N from the inner nominal starts (as the labeler does); else the residual blend if it never leaves;
    else the look-ahead switch. Per seed, scored by distance on common starts against the closest of the three
    (exact and within TOL of the labeler's distance) and against the closest of all 16 policies, with fixed
    choices and the two-outcome rule for comparison; reported separately for the settings that shaped the
    rule, the Reacher settings registered before running, and the held-out settings."""
    say(f"== Three-outcome rule per seed (error switch / residual blend / look-ahead switch); decided on {decide}")
    C = {"Switch": "switch", "Residual-box": "residual", "Filter": "look-ahead"}
    keys = ["ruleCV", "rule3", "rule2", "switch", "residual", "look-ahead"]
    groups = {"design": SETTINGS + [m for m in MISMATCH if not m[0].startswith("reacher")],
              "registered": [m for m in MISMATCH if m[0].startswith("reacher")], "held-out": HELDOUT}
    for gname, settings in groups.items():
        hit = {k: 0 for k in keys}
        tol = {k: 0 for k in keys}
        exc = {k: [] for k in keys}
        ex16 = {k: [] for k in keys}
        n = 0
        for dn, e in settings:
            try:
                s = A.per_seed(A.load(dn))
            except Exception:
                continue
            t = s[s.demo_eps == e]
            if t.empty:
                continue
            t = t.pivot_table(index="seed", columns="method", values=["kept_inner", "fid_common"])
            if decide == "inner":
                lab_k = t["kept_inner"]["Labeler"] - 1e-9
                bc_ok = t["kept_inner"]["BC"] >= lab_k
                res_ok = t["kept_inner"]["Residual-box"] >= lab_k
            elif decide == "dec":
                # decided on a fresh, independent set of inner nominal starts (run.py with DECISION=1)
                dd = os.path.join(A.OUT, "dec_BC_Residual-box", dn, f"eps{e}")
                fs = [os.path.join(dd, f"seed{sd}.csv") for sd in t.index]
                if not all(os.path.exists(f) for f in fs):
                    continue
                z = pd.concat([pd.read_csv(f) for f in fs]).pivot_table(index="seed", columns="method", values="kept_inner")
                z = z.loc[t.index]
                bc_ok = z["BC"] >= z["Labeler"] - 1e-9
                res_ok = z["Residual-box"] >= z["Labeler"] - 1e-9
            else:
                # decided on the radius-0 probe starts, drawn independently of the nominal starts that score
                # distance: a policy "never leaves" if, once inside N, it stays as the labeler does
                raw = A.load(dn)
                z = raw[(raw.radius == 0) & (raw.demo_eps == e)].pivot_table(index="seed", columns="method", values="stay")
                z = z.loc[t.index]
                bc_ok = z["BC"] >= z["Labeler"] - 1e-9
                res_ok = z["Residual-box"] >= z["Labeler"] - 1e-9
            f = {m: t["fid_common"][m].values for m in C}
            best = np.minimum.reduce([f[m] for m in C])
            best16 = np.minimum.reduce([t["fid_common"][m].values for m in ALL16 if m in t["fid_common"]])
            lab = t["fid_common"]["Labeler"].mean()
            cvf = os.path.join(A.OUT, "cvselect", dn, f"eps{e}")
            cvrows = [pd.read_csv(os.path.join(cvf, f"seed{sd}.csv")) for sd in t.index
                      if os.path.exists(os.path.join(cvf, f"seed{sd}.csv"))]
            if len(cvrows) == len(t):
                cv = pd.concat(cvrows).set_index("seed").loc[t.index]
                cands = ["Switch", "Filter", "Residual-box"]
                score = np.stack([np.where(cv[f"cvleft_{m}"] == 0, cv[f"cv_{m}"], np.inf) for m in cands], 1)
                pick = np.where(np.isfinite(score).any(1), score.argmin(1), 1)
                cvpick = np.choose(pick, [f[m] for m in cands])
            else:
                cvpick = np.full(len(t), np.nan)
            picks = {"ruleCV": cvpick,
                     "rule3": np.where(bc_ok, f["Switch"], np.where(res_ok, f["Residual-box"], f["Filter"])),
                     "rule2": np.where(bc_ok, f["Switch"], f["Residual-box"]),
                     "switch": f["Switch"], "residual": f["Residual-box"], "look-ahead": f["Filter"]}
            row = {}
            for k, v in picks.items():
                hit[k] += int((v <= best + 1e-12).sum())
                tol[k] += int((v <= best + TOL * lab).sum())
                exc[k] += list((v - best) / lab)
                ex16[k] += list((v - best16) / lab)
                row[k] = int((v <= best + TOL * lab).sum())
            n += len(t)
            br = f"switch {int(bc_ok.sum())}, residual {int((~bc_ok & res_ok).sum())}, look-ahead {int((~bc_ok & ~res_ok).sum())}"
            say(f"  [{gname}] {dn} {e}: n={len(t)} within-tolerance {row}; branches: {br}")
        if n == 0:
            continue
        RULE_PER[(decide, gname)] = {k: np.array(v) for k, v in exc.items()}
        say(f"  [{gname}] seeds {n}; closest of three (exact) {hit}")
        say(f"  [{gname}] within {TOL:.0%} of the labeler's distance: {tol}")
        say(f"  [{gname}] mean excess over closest of three (/labeler's):", {k: round(float(np.mean(v)), 3) for k, v in exc.items()})
        say(f"  [{gname}] mean excess over closest of all 16 policies (/labeler's):", {k: round(float(np.mean(v)), 3) for k, v in ex16.items()})


def sens():
    say("== Sensitivity (seeds 0-9): recovery / retention (inner) / distance (common)")
    for dn in ["quad3d", "reacher"]:
        main = A.load(dn)
        for e in [3, 30]:
            fs = glob.glob(os.path.join(A.OUT, "sens", dn, f"eps{e}_seed*.csv"))
            if not fs:
                continue
            m = main[(main.demo_eps == e) & (main.seed <= 9)]
            d = pd.concat([pd.concat([pd.read_csv(f) for f in fs]), m])
            out = {}
            for k, g in d.groupby("method"):
                out[k] = (round(100 * g[g.radius > 0].success.mean()), round(100 * g.groupby("seed").kept_inner.first().mean()),
                          round(g.groupby("seed").fid_common.first().mean(), 4))
            say(dn, e, out)


def invariance():
    f = os.path.join(A.OUT, "theory_check_quad.csv")
    if not os.path.exists(f):
        return
    say("== Invariance check (quadrotor): boundary exit rate vs nominal retention")
    t = pd.read_csv(f)
    say(t.groupby(["demo_eps", "method"]).exit.mean().unstack().round(3).to_string())
    s = A.per_seed(A.load("quad3d"))
    keep = ["BC", "Switch-support", "Blend-box", "DAgger-L", "Residual-box"]   # Switch/Simplex equal BC there
    m = t[t.method.isin(keep)].merge(s[["demo_eps", "method", "seed", "kept"]], on=["demo_eps", "method", "seed"])
    agg = m.groupby(["demo_eps", "method"])[["exit", "kept"]].mean()
    r1 = stats.spearmanr(agg.exit, agg.kept)
    r2 = stats.spearmanr(m.exit, m.kept)
    say(f"method x budget means (n={len(agg)}): Spearman {r1[0]:.2f} (p={r1[1]:.3g}); "
        f"per seed (n={len(m)}): {r2[0]:.2f} (p={r2[1]:.3g})")


def gap():
    say("== Gap sweep (box blend recovery %, 30 episodes; 3 episodes for the quadrotor too)")
    for dn in ["pendulum", "reacher", "quad3d"]:
        s = A.per_seed(A.load(dn))
        for e in sorted(set(s.demo_eps) & {3, 30}):
            t = s[s.demo_eps == e]
            say(dn, e, {k: round(100 * t[t.method == f"Blend-box@{k}"].recover.mean(), 1) for k in [0.5, 1.0, 2.0, 3.0]},
                "dist(kept)", {k: round(t[t.method == f"Blend-box@{k}"].fid_kept.mean(), 4) for k in [0.5, 1.0, 2.0, 3.0]})


def headline():
    say("== Per budget: recovery / retention / distance (common starts) / retention (inner starts) / distance (own retained)")
    for dn in ["quad3d", "reacher", "pendulum"]:
        s = A.per_seed(A.load(dn))
        for e in sorted(s.demo_eps.unique()):
            t = s[s.demo_eps == e]
            row = {}
            for m in ["BC", "DART", "Switch", "Switch-DART", "Switch-support", "Simplex", "Filter", "Backup", "SoftGate",
                      "Blend-box", "Blend-tube", "DAgger-L", "Residual", "Residual-box", "Labeler", "Demonstrator"]:
                g = t[t.method == m]
                row[m] = (round(100 * g.recover.mean()), round(100 * g.kept.mean()), round(g.fid_common.mean(), 4),
                          round(100 * g.kept_inner.mean()), round(g.fid_kept.mean(), 4))
            say(dn, e, row)


if __name__ == "__main__":
    p1()
    p1_box()
    invariance()
    gap()
    headline()
    pairs()
    rule()
    rule3()
    rule3('probe0')
    rule3('dec')
    sens()
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
