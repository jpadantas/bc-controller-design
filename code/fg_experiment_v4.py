"""Flight case study, version 4.

Changes from fg_experiment_v3.py:
  * (folding the aileron trim into an equivalent stick was tried, fg_trimfold.py, and does not reproduce
    the pilots' replay, so the policies keep the trims as outputs, as in v3);
  * every network gets the same number of gradient steps (STEPS), as in the controlled domains;
  * a support switch (no model needed): BC unless the input is farther from every training frame than
    the training frames' own leave-one-pilot-out nearest-neighbor distances (95th percentile), or the slot
    error exceeds 10,000 ft;
  * the residual blend and DAgger-L are trained here with the same recipe.

    python fg_experiment_v4.py --workers 6
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import fg_experiment_v3 as X3

STEPS = 12000
OFFSETS = X3.OFFSETS
SEEDS = [int(s) for s in os.environ.get("FG_SEEDS", "0,1,2").split(",")]
PILOTS = [1, 2, 3, 4, 5]
INPUTS = X3.INPUTS
OUTPUTS = X3.OUTPUTS
LEARNED = ["BC", "Switch", "SoftGate", "Simplex", "Support", "Blend", "Residual-box", "DAgger-L"]
COLLECT = [(10607.0, 10607.0), (-10607.0, 10607.0), (10607.0, -10607.0), (-10607.0, -10607.0)]
COLS9 = ["xb", "yb", "dz", "roll", "pitch", "hdg_rel", "kias", "vs", "nz"]


def out_dir():
    import flight_unified as F
    return os.path.join(F.ROOT, "outputs", "flightgear", "experiment_v4")


def load():
    return X3.load()


def train(X, Y, Xnorm, seed, steps=STEPS, batch=128):
    import torch
    import flight_unified as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m, s = F._norm(Xnorm)
    Xt = torch.tensor((X - m) / s, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    net = F.MLP(X.shape[1], Y.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    done = 0
    while done < steps:
        order = torch.tensor(rng.permutation(len(Xt)))
        for st in range(0, len(Xt), batch):
            b = order[st:st + batch]
            loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            done += 1
            if done >= steps:
                break
    return F.wrap(net, m, s)


def support_threshold(Xz, pilot_ids, q=95.0):
    from scipy.spatial import cKDTree
    d = np.empty(len(Xz))
    for p in np.unique(pilot_ids):
        a, b = pilot_ids == p, pilot_ids != p
        d[a] = cKDTree(Xz[b]).query(Xz[a])[0]
    return float(np.percentile(d, q))


def policies(df, k, seed, with_dagger=None):
    import fg_bridge as B
    from scipy.spatial import cKDTree
    tr = df[df.pilot_id != k]
    X, Y = tr[INPUTS].values, tr[OUTPUTS].values
    lab = X3.labeler6(B.FormationController())
    rng = np.random.default_rng(X3.mseed("fg4", k, seed))
    n = int(X3.AUG_FRACTION * len(X))
    Sb = X3.synthetic_box(X, n, rng, B.SLOT)
    bc = train(X, Y, X, X3.mseed("fg4", k, seed, "BC"))
    blend = train(np.concatenate([X, Sb]), np.concatenate([Y, lab(Sb)]), X, X3.mseed("fg4", k, seed, "Blend"))
    res = Y - lab(X)
    rnet = train(np.concatenate([X, Sb]), np.concatenate([res, np.zeros((len(Sb), res.shape[1]))]), X,
                 X3.mseed("fg4", k, seed, "Residual-box"))
    mu, sd = X.mean(0), X.std(0) + 1e-9
    tree = cKDTree((X - mu) / sd)
    thr = support_threshold((X - mu) / sd, tr.pilot_id.values)

    def dist(Z):
        return np.linalg.norm(np.atleast_2d(Z)[:, :3] - B.SLOT, axis=1)

    def switch(x):
        Z = np.atleast_2d(x)
        out = bc(Z)
        far = dist(Z) > 10000.0
        out[far] = lab(Z[far])
        return out

    def support(x):
        Z = np.atleast_2d(x)
        out = bc(Z)
        far = (dist(Z) > 10000.0) | (tree.query((Z - mu) / sd)[0] > thr)
        out[far] = lab(Z[far])
        return out

    def softgate(x):
        Z = np.atleast_2d(x)
        w = np.clip((dist(Z) - 8000.0) / 4000.0, 0.0, 1.0)[:, None]
        return (1 - w) * bc(Z) + w * lab(Z)

    def resid(x):
        Z = np.atleast_2d(x)
        return lab(Z) + rnet(Z)

    makers = {"BC": lambda: bc, "Blend": lambda: blend, "Switch": lambda: switch, "SoftGate": lambda: softgate,
              "Simplex": lambda: X3.Simplex(bc, lab, B.SLOT), "Support": lambda: support,
              "Residual-box": lambda: resid}
    if with_dagger is not None:
        makers["DAgger-L"] = with_dagger(X, Y, Sb, lab, blend)
    return makers, lab, tr, df[df.pilot_id == k].reset_index(drop=True)


def dagger_builder(k, seed, g):
    """DAgger-L: from the blend, two rounds of labeler labels on the far states it visits from four
    diagonal displacements not used for evaluation."""
    import fg_eval as E
    cache = os.path.join(out_dir(), "dagger")
    os.makedirs(cache, exist_ok=True)

    def build(X, Y, S, lab, blend):
        pi = blend
        for it in range(2):
            vis = []
            for off in COLLECT:
                path = os.path.join(cache, f"p{k}_s{seed}_r{it}_lat{int(off[0])}_lon{int(off[1])}.csv")
                if os.path.exists(path):
                    log = pd.read_csv(path)
                else:
                    log = E.rollout(lambda x, p=pi: np.atleast_2d(p(x))[0], g, off, 120.0, tag=f"dagger{it}")
                    log.to_csv(path, index=False)
                vis.append(log[log.slot_err >= 10000.0][COLS9].values)
            V = np.concatenate(vis)
            if len(V):
                S = np.concatenate([S, V])
            pi = train(np.concatenate([X, S]), np.concatenate([Y, lab(S)]), X, X3.mseed("fg4", k, seed, "DAgger", it))
        return lambda: pi
    return build


def run_group(k, seed, methods):
    import fg_eval as E
    df = load()
    od = out_dir()
    g = df[df.pilot_id == k].reset_index(drop=True)
    if seed < 0:
        import fg_bridge as B
        lab = X3.labeler6(B.FormationController())
        jobs = [("Labeler", lambda: lab)]
    else:
        db = dagger_builder(k, seed, g) if "DAgger-L" in methods else None
        makers, lab, tr, va = policies(df, k, seed, with_dagger=db)
        Xv, Yv = va[INPUTS].values, va[OUTPUTS].values
        rows = []
        for m in ["BC", "Blend", "Residual-box", "Support", "Switch", "SoftGate"] + (["DAgger-L"] if db else []):
            P = np.atleast_2d(makers[m]()(Xv))
            rows.append(dict(held_out=k, seed=seed, method=m, mse4=float(np.mean((P - Yv) ** 2))))
        P = lab(Xv)
        rows.append(dict(held_out=k, seed=seed, method="Labeler", mse4=float(np.mean((P - Yv) ** 2))))
        pd.DataFrame(rows).to_csv(os.path.join(od, f"fidelity_p{k}_s{seed}.csv"), index=False)
        jobs = [(m, makers[m]) for m in methods]
    for off in OFFSETS:
        for m, mk in jobs:
            path = os.path.join(od, f"p{k}_s{seed}_{m}_lat{int(off[0])}_lon{int(off[1])}.csv")
            if os.path.exists(path):
                continue
            log = None
            for attempt in range(4):   # FlightGear can time out while loading on a saturated CPU: retry
                act = mk()
                try:
                    log = E.rollout(lambda x, act=act: np.atleast_2d(act(x))[0], g, off, 120.0, tag=m)
                    break
                except (TimeoutError, ConnectionError, OSError) as ex:
                    print(f"[inst {os.environ.get('FG_INSTANCE')}] pilot {k} seed {seed} {m} off={off}: "
                          f"attempt {attempt + 1} failed ({type(ex).__name__}); retrying", flush=True)
                    time.sleep(20)
            if log is None:
                continue
            log["held_out"], log["seed"], log["method"], log["off_lat"], log["off_lon"] = k, seed, m, off[0], off[1]
            log.to_csv(path, index=False)
            last = log.iloc[-1]
            print(f"[inst {os.environ.get('FG_INSTANCE')}] pilot {k} seed {seed} {m:12s} off={off}: "
                  f"t_end={last.t:5.1f}s err={last.slot_err:8.0f} ft", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--group":
        k, seed, methods = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split(",")
        sys.argv = sys.argv[:1]
        run_group(k, seed, methods)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--methods", default=",".join(LEARNED))
    ap.add_argument("--first-instance", type=int, default=0)
    a = ap.parse_args()
    sys.argv = sys.argv[:1]
    os.makedirs(out_dir(), exist_ok=True)
    groups = [(k, s) for s in SEEDS for k in PILOTS] + [(k, -1) for k in PILOTS]
    free = list(range(a.first_instance, a.first_instance + a.workers))
    running = []
    while groups or running:
        while groups and free:
            inst = free.pop(0)
            k, s = groups.pop(0)
            env = dict(os.environ, FG_INSTANCE=str(inst))
            p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--group", str(k), str(s), a.methods],
                                 env=env)
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
