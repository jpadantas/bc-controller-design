"""Probe starts for the v4 flight policies (fg_experiment_v4.py): an attitude upset at the slot and a
6,000 ft displacement, every held-out pilot and training seed, every v4 method; the labeler once.

    python fg_extra_v4.py --workers 5
Writes outputs/flightgear/experiment_v4/extra/p{k}_s{seed}_{method}_{upset|near}.csv.
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

import fg_experiment_v4 as X4
from fg_extra_v3 import STARTS

METHODS = ["BC", "Switch", "Simplex", "Support", "Blend", "Residual-box"]


def run_group(k, seed):
    import fg_eval as E
    df = X4.load()
    od = os.path.join(X4.out_dir(), "extra")
    os.makedirs(od, exist_ok=True)
    if seed < 0:
        import fg_bridge as B
        import fg_experiment_v3 as X3
        lab = X3.labeler6(B.FormationController())
        makers, ms = {"Labeler": lambda: lab}, ["Labeler"]
        g = df[df.pilot_id == k].reset_index(drop=True)
    else:
        makers, _, _, g = X4.policies(df, k, seed)
        ms = METHODS
    for name, (off, pert) in STARTS.items():
        for m in ms:
            path = os.path.join(od, f"p{k}_s{seed}_{m}_{name}.csv")
            if os.path.exists(path):
                continue
            log = None
            for attempt in range(4):
                act = makers[m]()
                try:
                    log = E.rollout(lambda x, act=act: np.atleast_2d(act(x))[0], g, off, 120.0, tag=m, perturb=pert)
                    break
                except (TimeoutError, ConnectionError, OSError) as ex:
                    print(f"[inst {os.environ.get('FG_INSTANCE')}] retry {attempt + 1} ({type(ex).__name__})", flush=True)
                    time.sleep(20)
            if log is None:
                continue
            log["held_out"], log["seed"], log["method"], log["start_kind"] = k, seed, m, name
            log.to_csv(path, index=False)
            last = log.iloc[-1]
            print(f"[inst {os.environ.get('FG_INSTANCE')}] pilot {k} seed {seed} {m:12s} {name}: t_end={last.t:5.1f}s "
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
    groups = [(k, s) for s in X4.SEEDS for k in X4.PILOTS] + [(k, -1) for k in X4.PILOTS]
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
