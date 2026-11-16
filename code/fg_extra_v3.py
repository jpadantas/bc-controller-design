"""Extra flight starts (reviewer request), training seeds 0-2, every held-out pilot:
  upset -- the held-out pilot's first frame with bank +60 deg, pitch -10 deg and 80 kt less airspeed
           (inside the switch radius, outside the pilots' usual attitude/speed)
  near  -- a 6,000 ft lateral displacement (inside the 10,000 ft switch radius)
Methods: BC, Switch, SoftGate, Simplex, BlendPos, Blend, DAgger-L (rebuilt deterministically from
the cached fg_dagger_v3 collection rollouts), Residual-box, Labeler.

    python fg_extra_v3.py --workers 6      (after fg_dagger_v3.py)
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import fg_experiment_v3 as X

STARTS = {"upset": ((0.0, 0.0), dict(roll=60.0, pitch=-10.0, kias=-80.0)), "near": ((6000.0, 0.0), None)}
SEEDS = (0, 1, 2)
COLS9 = ["xb", "yb", "dz", "roll", "pitch", "hdg_rel", "kias", "vs", "nz"]


def policies(k, SEED):
    import fg_bridge as B
    df = X.load()
    makers, lab, _, g = X.policies(df, k, SEED)
    tr = df[df.pilot_id != k]
    Xd, Yd = tr[X.INPUTS].values, tr[X.OUTPUTS].values
    rng = np.random.default_rng(X.mseed("fg3", k, SEED))
    n = int(X.AUG_FRACTION * len(Xd))
    S = X.synthetic_box(Xd, n, rng, B.SLOT)
    cache = os.path.join(X.out_dir(), "dagger")
    for it in range(2):
        fs = sorted(glob.glob(os.path.join(cache, f"p{k}_s{SEED}_r{it}_*.csv")))
        if len(fs) != 4:
            raise SystemExit(f"missing DAgger collection rollouts for pilot {k} round {it}")
        V = np.concatenate([pd.read_csv(f).query("slot_err >= 10000.0")[COLS9].values for f in fs])
        if len(V):
            S = np.concatenate([S, V])
        pi = X.train(np.concatenate([Xd, S]), np.concatenate([Yd, lab(S)]), Xd, X.mseed("fg3", k, SEED, "DAgger", it))
    S0 = X.synthetic_box(Xd, n, np.random.default_rng(X.mseed("fg3", k, SEED)), B.SLOT)
    res = Yd - lab(Xd)
    rnet = X.train(np.concatenate([Xd, S0]), np.concatenate([res, np.zeros((len(S0), res.shape[1]))]), Xd,
                   X.mseed("fg3", k, SEED, "Residual-box"))
    makers["DAgger-L"] = lambda: pi
    makers["Residual-box"] = lambda: (lambda x: lab(np.atleast_2d(x)) + rnet(np.atleast_2d(x)))
    makers["Labeler"] = lambda: lab
    return makers, g


def run_group(k, SEED):
    import fg_eval as E
    makers, g = policies(k, SEED)
    od = os.path.join(X.out_dir(), "extra")
    os.makedirs(od, exist_ok=True)
    for name, (off, pert) in STARTS.items():
        for m in ["BC", "Switch", "SoftGate", "Simplex", "BlendPos", "Blend", "DAgger-L", "Residual-box", "Labeler"]:
            if m == "Labeler" and SEED != 0:      # the labeler does not depend on the training seed
                continue
            path = os.path.join(od, f"p{k}_s{SEED}_{m}_{name}.csv")
            if os.path.exists(path):
                continue
            act = makers[m]()
            log = E.rollout(lambda x, act=act: np.atleast_2d(act(x))[0], g, off, 120.0, tag=m, perturb=pert)
            log["held_out"], log["seed"], log["method"], log["start_kind"] = k, SEED, m, name
            log.to_csv(path, index=False)
            last = log.iloc[-1]
            print(f"[inst {os.environ.get('FG_INSTANCE')}] pilot {k} {m:12s} {name}: t_end={last.t:5.1f}s "
                  f"err={last.slot_err:8.0f} ft", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--group":
        k, sd = int(sys.argv[2]), int(sys.argv[3])
        sys.argv = sys.argv[:1]
        run_group(k, sd)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    sys.argv = sys.argv[:1]
    groups = [(k, sd) for sd in SEEDS for k in X.PILOTS]
    free, running = list(range(a.workers)), []
    while groups or running:
        while groups and free:
            inst = free.pop(0)
            k = groups.pop(0)
            p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--group", str(k[0]), str(k[1])],
                                 env=dict(os.environ, FG_INSTANCE=str(inst)))
            running.append((p, inst, k))
        time.sleep(5)
        for item in list(running):
            p, inst, k = item
            if p.poll() is not None:
                running.remove(item)
                free.append(inst)
                print(f"group pilot {k} finished (exit {p.returncode})", flush=True)


if __name__ == "__main__":
    main()
