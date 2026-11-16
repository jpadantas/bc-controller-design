"""Analysis of the v3 FlightGear experiment (fg_experiment_v3.py).

Per rollout: crash (ground contact before 120 s), slot error at 60/120 s (a crashed
rollout keeps its last value), fraction of the 120 s within 2/5/10 kft of the slot
(time after a crash counts as outside), maximum |bank|, altitude lost, g range,
peak control step after the first second, and, from the nominal start only,
distribution distances (1-Wasserstein) to the held-out pilot's own session for
bank angle, normal load factor and aileron rate.

Statistics: rollouts are grouped by (held-out pilot, start); learned policies
are averaged over their 3 training seeds within a group, giving 25 paired units;
paired Wilcoxon against the Switch with Holm across policies. We also count in
how many of the 5 folds a policy beats the Switch.

    python fg_analyze_v3.py
Writes paper/tables/v3_flight.tex and paper/figures/fig_flight_v3.pdf.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.argv = sys.argv[:1]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "outputs", "flightgear", os.environ.get("FLIGHT_EXP", "experiment_v3"))
TAB = os.path.join(ROOT, "paper", "tables")
FIGD = os.path.join(ROOT, "paper", "figures")
HORIZON = 120.0
G_FT = 32.174
METHODS = [("BC", "BC"), ("Switch", "Switch"), ("SoftGate", "Soft gate"), ("Simplex", "Simplex"), ("Support", "Support switch"),
           ("BlendPos", "Blend, box-partial"), ("Blend", "Blend, box"), ("DAgger-L", "Blend, DAgger-L"),
           ("Residual-box", "Residual blend"), ("Labeler", "Labeler alone")]
BS = "\\"
NL = BS + BS


def pilot_log():
    import flight_unified as F
    return F.prepare(pd.read_csv(F.DATA))


def first_segment(log, pilot, seconds=HORIZON):
    """The pilot's own first 120 s from the first cleaned frame: the segment the nominal-start
    rollouts reproduce."""
    g = log[log.pilot_id == pilot]
    return g[g.time_s - g.time_s.iloc[0] <= seconds]


def style_distances(roll, g_load, ail, roll_p, g_p, ail_p):
    return dict(w_bank=stats.wasserstein_distance(np.abs(roll), np.abs(roll_p)),
                w_g=stats.wasserstein_distance(g_load, g_p),
                w_ailrate=stats.wasserstein_distance(np.abs(np.diff(ail)), np.abs(np.diff(ail_p))))


def inter_pilot_spread(log):
    """Reference scale: the same distances between one pilot's first 120 s and another's."""
    rows = []
    for a in range(1, 6):
        for b in range(1, 6):
            if a == b:
                continue
            ga, gb = first_segment(log, a), first_segment(log, b)
            rows.append(style_distances(ga.roll_deg.values, -ga.pilot_z_accel_fps_sec.values / G_FT, ga.aileron.values,
                                        gb.roll_deg.values, -gb.pilot_z_accel_fps_sec.values / G_FT, gb.aileron.values))
    out = pd.DataFrame(rows).mean()
    tt = np.arange(0.0, HORIZON, 0.5)

    def track(k):
        g = first_segment(log, k)
        tp = (g.time_s - g.time_s.iloc[0]).values
        return np.stack([np.interp(tt, tp, g[c].values) for c in ("dx_body", "dy_body", "dz_leader")], 1)
    out["dist"] = np.mean([np.sqrt(np.mean(np.sum((track(a) - track(b)) ** 2, 1)))
                           for a in range(1, 6) for b in range(1, 6) if a != b])
    return out


def per_rollout(log):
    rows = []
    files = [(f, "std") for f in glob.glob(os.path.join(EXP, "p*_s*_*.csv"))]
    files += [(f, os.path.splitext(f)[0].rsplit("_", 1)[1]) for f in glob.glob(os.path.join(EXP, "extra", "p*_s*_*.csv"))]
    for f, kind in files:
        d = pd.read_csv(f)
        if d.empty:
            continue
        crashed = d.t.iloc[-1] < HORIZON - 1.0

        def at(t):
            return float(d.slot_err.iloc[(d.t - t).abs().argmin()]) if d.t.iloc[-1] >= t else float(d.slot_err.iloc[-1])
        U = d[["ail", "ele", "thr"]].values
        du = np.linalg.norm(np.diff(U, axis=0), axis=1)
        du_late = du[d.t.values[1:] > 1.0]
        n = HORIZON / 0.1
        r = dict(pilot=int(d.held_out.iloc[0]), seed=int(d.seed.iloc[0]), method=d.method.iloc[0],
                 start=(f"{int(d.off_lat.iloc[0])},{int(d.off_lon.iloc[0])}" if "off_lat" in d else kind), kind=kind, crash=float(crashed),
                 e60=at(60), e120=at(120),
                 w2=(d.slot_err < 2000).sum() / n, w5=(d.slot_err < 5000).sum() / n, w10=(d.slot_err < 10000).sum() / n,
                 max_bank=float(d.roll.abs().max()), alt_loss=float(d.alt.iloc[0] - d.alt.min()),
                 g_max=float((-d.nz / G_FT).max()), g_min=float((-d.nz / G_FT).min()),
                 du_peak=float(du_late.max()) if len(du_late) else np.nan, e0=float(d.slot_err.iloc[0]))
        if r["start"] == "0,0" and kind == "std":
            g = first_segment(log, r["pilot"])
            r.update(**style_distances(d.roll.values, -d.nz.values / G_FT, d.ail.values,
                                       g.roll_deg.values, -g.pilot_z_accel_fps_sec.values / G_FT, g.aileron.values))
            # RMS distance to the held-out pilot's own track (leader frame), aligned in time from the start;
            # after a crash the last position is held
            tp = (g.time_s - g.time_s.iloc[0]).values
            tt = np.arange(0.0, HORIZON, 0.5)
            P = np.stack([np.interp(tt, tp, g[c].values) for c in ("dx_body", "dy_body", "dz_leader")], 1)
            Q = np.stack([np.interp(tt, d.t.values, d[c].values) for c in ("xb", "yb", "dz")], 1)
            r["dist"] = float(np.sqrt(np.mean(np.sum((P - Q) ** 2, 1))))
        rows.append(r)
    return pd.DataFrame(rows)


def pilots_reference(log):
    """The pilots' own values over their sessions (formation already established)."""
    out = []
    for k, g in log.groupby("pilot_id"):
        out.append(dict(pilot=k, w2=(g.err < 2000).mean(), w5=(g.err < 5000).mean(), w10=(g.err < 10000).mean(),
                        max_bank=g.roll_deg.abs().max(), med_err=g.err.median()))
    return pd.DataFrame(out)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    out, run = np.empty_like(p), 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        out[j] = run
    return out


def mci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def main():
    log = pilot_log()
    rall = per_rollout(log)
    r = rall[rall.kind == "std"].copy()
    pd.set_option("display.width", 250)
    print("rollouts per method:", r.groupby("method").size().to_dict())
    cols = ["crash", "e60", "e120", "w2", "w5", "w10", "max_bank", "alt_loss", "g_max", "g_min", "du_peak"]
    print(r.groupby("method")[cols].mean().round(3).to_string())
    print(r[r.start == "0,0"].groupby("method")[["w_bank", "w_g", "w_ailrate"]].mean().round(3).to_string())
    print("pilots:\n", pilots_reference(log).round(3).to_string())
    # paired units: (pilot, start), learned policies averaged over seeds
    u = r.groupby(["method", "pilot", "start"])[cols].mean().reset_index()
    ref = u[u.method == "Switch"].set_index(["pilot", "start"])
    for k in ["crash", "e120", "w2", "w5", "w10", "du_peak", "max_bank"]:
        ms = [m for m, _ in METHODS if m not in ("Switch",) and m in set(u.method)]
        ps, med = [], []
        for m in ms:
            x = u[u.method == m].set_index(["pilot", "start"])
            i = x.index.intersection(ref.index)
            d = (x.loc[i, k] - ref.loc[i, k]).values
            ps.append(stats.wilcoxon(d).pvalue if np.any(d != 0) and len(d) > 5 else np.nan)
            med.append(np.median(d))
        print(k, {m: (round(md, 3), round(p, 4)) for m, md, p in zip(ms, med, holm(np.nan_to_num(ps, nan=1.0)))})
    folds = r.groupby(["method", "pilot"])[["w5", "crash"]].mean().unstack(0)
    print("fold-level w5:\n", folds["w5"].round(2).to_string())


if __name__ == "__main__":
    main()
