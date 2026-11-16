"""DAgger-L in FlightGear: the blend trained on labeler labels for the off-nominal states
the policy itself visits (the method the controlled domains recommend).

Per (held-out pilot, training seed), starting from the v3 'Blend' (same data and seed):
  2 rounds: fly the current policy from 4 COLLECTION starts (15,000 ft diagonal
  displacements of the held-out pilot's first frame -- different from the 5
  evaluation starts), keep the visited states with slot error >= 10,000 ft, label
  them with the formation labeler (trims 0), add them to the synthetic set, retrain.
Then evaluate on the same 5 starts as fg_experiment_v3.py (method 'DAgger-L'), and also the
residual policy 'Residual-box' (labeler + residual cloned from the pilots, zero off nominal).

    python fg_dagger_v3.py --workers 6
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import fg_experiment_v3 as X

COLLECT = [(10607.0, 10607.0), (-10607.0, 10607.0), (10607.0, -10607.0), (-10607.0, -10607.0)]
ROUNDS = 2
COLS9 = ["xb", "yb", "dz", "roll", "pitch", "hdg_rel", "kias", "vs", "nz"]


def run_group(k, seed):
    import fg_bridge as B
    import fg_eval as E
    df = X.load()
    od = X.out_dir()
    cache = os.path.join(od, "dagger")
    os.makedirs(cache, exist_ok=True)
    tr = df[df.pilot_id != k]
    Xd, Yd = tr[X.INPUTS].values, tr[X.OUTPUTS].values
    lab = X.labeler6(B.FormationController())
    rng = np.random.default_rng(X.mseed("fg3", k, seed))
    n = int(X.AUG_FRACTION * len(Xd))
    S = X.synthetic_box(Xd, n, rng, B.SLOT)          # identical to the v3 Blend's synthetic set
    _ = X.synthetic_pos(Xd, n, rng, B.SLOT)
    pi = X.train(np.concatenate([Xd, S]), np.concatenate([Yd, lab(S)]), Xd, X.mseed("fg3", k, seed, "Blend"))
    g = df[df.pilot_id == k].reset_index(drop=True)
    for it in range(ROUNDS):
        vis = []
        for off in COLLECT:
            path = os.path.join(cache, f"p{k}_s{seed}_r{it}_lat{int(off[0])}_lon{int(off[1])}.csv")
            if os.path.exists(path):
                log = pd.read_csv(path)
            else:
                log = E.rollout(lambda x, p=pi: np.atleast_2d(p(x))[0], g, off, 120.0, tag=f"dagger{it}")
                log.to_csv(path, index=False)
            far = log[log.slot_err >= 10000.0]
            vis.append(far[COLS9].values)
        V = np.concatenate(vis) if vis else np.zeros((0, 9))
        if len(V):
            S = np.concatenate([S, V])
        pi = X.train(np.concatenate([Xd, S]), np.concatenate([Yd, lab(S)]), Xd, X.mseed("fg3", k, seed, "DAgger", it))
        print(f"[inst {os.environ.get('FG_INSTANCE')}] pilot {k} seed {seed} round {it}: +{len(V)} states", flush=True)
    for off in X.OFFSETS:
        path = os.path.join(od, f"p{k}_s{seed}_DAgger-L_lat{int(off[0])}_lon{int(off[1])}.csv")
        if os.path.exists(path):
            continue
        log = E.rollout(lambda x, p=pi: np.atleast_2d(p(x))[0], g, off, 120.0, tag="DAgger-L")
        log["held_out"], log["seed"], log["method"], log["off_lat"], log["off_lon"] = k, seed, "DAgger-L", off[0], off[1]
        log.to_csv(path, index=False)
        last = log.iloc[-1]
        print(f"[inst {os.environ.get('FG_INSTANCE')}] pilot {k} seed {seed} DAgger-L off={off}: "
              f"t_end={last.t:5.1f}s err={last.slot_err:8.0f} ft", flush=True)
    # Residual-box: labeler + residual fitted to (pilot - labeler) on pilot frames and to 0 on the
    # same full-state synthetic states as the Blend (off nominal the policy reduces to the labeler)
    rng2 = np.random.default_rng(X.mseed("fg3", k, seed))
    S0 = X.synthetic_box(Xd, n, rng2, B.SLOT)
    res = Yd - lab(Xd)
    rnet = X.train(np.concatenate([Xd, S0]), np.concatenate([res, np.zeros((len(S0), res.shape[1]))]), Xd,
                   X.mseed("fg3", k, seed, "Residual-box"))

    def resid(x):
        Z = np.atleast_2d(x)
        return lab(Z) + rnet(Z)
    for off in X.OFFSETS:
        path = os.path.join(od, f"p{k}_s{seed}_Residual-box_lat{int(off[0])}_lon{int(off[1])}.csv")
        if os.path.exists(path):
            continue
        log = E.rollout(lambda x: resid(x)[0], g, off, 120.0, tag="Residual-box")
        log["held_out"], log["seed"], log["method"], log["off_lat"], log["off_lon"] = k, seed, "Residual-box", off[0], off[1]
        log.to_csv(path, index=False)
        last = log.iloc[-1]
        print(f"[inst {os.environ.get('FG_INSTANCE')}] pilot {k} seed {seed} Residual-box off={off}: "
              f"t_end={last.t:5.1f}s err={last.slot_err:8.0f} ft", flush=True)
    va = df[df.pilot_id == k]
    Pr = np.atleast_2d(resid(va[X.INPUTS].values))
    pd.DataFrame([dict(held_out=k, seed=seed, method="Residual-box", mse4=float(np.mean((Pr[:, :4] - va[X.OUTPUTS].values[:, :4]) ** 2)),
                       mse6=float(np.mean((Pr - va[X.OUTPUTS].values) ** 2)))]).to_csv(
        os.path.join(od, f"fidelity_resid_p{k}_s{seed}.csv"), index=False)
    P = np.atleast_2d(pi(va[X.INPUTS].values))
    Yv = va[X.OUTPUTS].values
    pd.DataFrame([dict(held_out=k, seed=seed, method="DAgger-L", mse4=float(np.mean((P[:, :4] - Yv[:, :4]) ** 2)),
                       mse6=float(np.mean((P - Yv) ** 2)))]).to_csv(
        os.path.join(od, f"fidelity_dagger_p{k}_s{seed}.csv"), index=False)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--group":
        k, seed = int(sys.argv[2]), int(sys.argv[3])
        sys.argv = sys.argv[:1]
        run_group(k, seed)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    sys.argv = sys.argv[:1]
    groups = [(k, s) for s in X.SEEDS for k in X.PILOTS]
    free, running = list(range(a.workers)), []
    while groups or running:
        while groups and free:
            inst = free.pop(0)
            k, s = groups.pop(0)
            env = dict(os.environ, FG_INSTANCE=str(inst))
            p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--group", str(k), str(s)], env=env)
            running.append((p, inst, k, s))
        time.sleep(5)
        for item in list(running):
            p, inst, k, s = item
            if p.poll() is not None:
                running.remove(item)
                free.append(inst)
                print(f"group pilot {k} seed {s} finished (exit {p.returncode})", flush=True)


if __name__ == "__main__":
    main()
