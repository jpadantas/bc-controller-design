"""Folding the aileron trim into the stick.

Two pilots flew with large aileron trim offset by stick. A policy that outputs stick and trim separately
must learn their split, which only their combined effect constrains. Here we map (stick, trim) to a single
equivalent stick s_eff with zero trim, through the F-16 FCS roll-command shaping (stick -> stick force
-> commanded roll rate, ada189675 branch off: force = 17 sign(s)|s|^1.3, piecewise-linear rate table):
    rate(s_eff) = rate(s) + K * trim,
and choose K by open-loop replay of the pilots' recorded controls for 30 s (as fg_replay.py), with the
trims held at zero and the stick replaced by s_eff.

    FG_INSTANCE=5 python fg_trimfold.py            # replay sweep over K
Writes outputs/flightgear/replay/trimfold_p{k}_K{K}.csv and trimfold_summary.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.argv = sys.argv[:1]

F_TAB = np.array([-17.0, -11.0, -6.0, -1.0, 0.0, 1.0, 6.0, 11.0, 17.0])
R_TAB = np.array([-308.0, -80.0, -20.0, 0.0, 0.0, 0.0, 20.0, 80.0, 308.0])
KS = tuple(float(v) for v in os.environ.get('KS', '0,12,19,26,40').split(','))
PILOTS_T = [int(v) for v in os.environ.get('PILOTS', '2,3,4').split(',')]


def rate(s):
    f = 17.0 * np.sign(s) * np.abs(np.clip(s, -1, 1)) ** 1.3
    return np.interp(f, F_TAB, R_TAB)


def inv_rate(r, s_hint):
    """Stick giving commanded rate r; inside the dead band (|rate| ~ 0) keep the pilot's own stick, clipped
    to the band, so that a zero command maps to the same small stick."""
    r = np.clip(r, -308.0, 308.0)
    grid = np.linspace(-1, 1, 4001)
    rg = rate(grid)
    out = np.empty_like(np.asarray(r, float))
    for i, (ri, si) in enumerate(zip(np.atleast_1d(r), np.atleast_1d(s_hint))):
        cand = grid[np.abs(rg - ri) <= 1e-9 + 0.5 * np.min(np.abs(rg - ri)) + 1e-6] if False else None
        j = np.argmin(np.abs(rg - ri) + 1e-4 * np.abs(grid - si))
        out[i] = grid[j]
    return out


def fold(stick, trim, K):
    return inv_rate(rate(stick) + K * trim, stick)


def main():
    import fg_bridge as B
    import fg_eval as E
    import fg_replay as RP
    import flight_unified as FU
    import time
    df = FU.prepare(pd.read_csv(FU.DATA))
    summ = []
    for k in PILOTS_T:
        g = df[df.pilot_id == k].reset_index(drop=True)
        g30 = g[g.time_s - g.time_s.iloc[0] <= 31.0].copy()
        for K in KS:
            path = os.path.join(RP.OUT, f"trimfold_p{k}_K{int(K)}.csv")
            if os.path.exists(path):
                d = pd.read_csv(path)
            else:
                gg = g.copy()
                gg.loc[g30.index, "aileron"] = fold(g30.aileron.values, g30.aileron_trim.values, K)
                gg["aileron_trim"] = 0.0
                d = RP.replay(gg, with_trim=False)
                d.to_csv(path, index=False)
            s = dict(pilot=k, K=K, t_end=float(d.t.iloc[-1]), **RP.summarize(d))
            summ.append(s)
            print({a: (round(v, 2) if isinstance(v, float) else v) for a, v in s.items()}, flush=True)
    pd.DataFrame(summ).to_csv(os.path.join(RP.OUT, "trimfold_summary.csv"), index=False)


if __name__ == "__main__":
    main()
