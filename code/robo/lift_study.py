"""Human-demonstration study on robomimic Lift (multi-human): six operators, leave one out.

Per (held-out operator, training seed): train on the other five operators' demonstrations
  BC            demonstrations only
  Blend-box     demonstrations + labeler labels on synthetic off-nominal states
  Residual-box  labeler + network fitted to (demonstrated action - labeler) on demonstrations and to 0 on
                the synthetic states
and combine with the scripted labeler
  Switch        BC unless the cube or end effector is outside the demonstrated workspace (a monitor on
                positions only: the task-space analogue of the error switch)
  Support       BC unless the full observation is farther from every training frame than the training
                operators' frames are from each other's (95th percentile), or the Switch condition holds
  Labeler       the scripted law alone
Every network gets STEPS gradient steps. Synthetic states: the arm's joints perturbed (sd 0.5 rad) and the
cube placed anywhere within 0.2 m, read from the simulator (so they are physically consistent).
Evaluation:
  nominal   20 resets from the task's own distribution: success, and retention (never leaves the demonstrated
            workspace, the task-space nominal set; also never leaving the full-state support)
  offnom    20 off-nominal starts (joints perturbed, sd 0.4 rad; cube within 0.15 m): recovery (success)
  fidelity  from the initial state of every other one of the held-out operator's 50 demonstrations: RMS distance of the
            end-effector path to the recorded one (time-aligned, the shorter padded with its last point),
            success, and the difference in completion time.
Writes outputs/robo/lift/p{op}_s{seed}.csv.

    MUJOCO_GL=disable ROBO_DATA=<path> <robosuite-venv>/python lift_study.py --workers 6
"""
import argparse
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lift_common as L  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "outputs", "robo", os.environ.get("LIFT_OUT", "lift"))
STEPS = int(os.environ.get("LIFT_STEPS", "20000"))
N_EVAL = 20
AUG_FRAC = 0.3
SEEDS = (0, 1, 2)


def mseed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31)


def train(X, Y, Xnorm, seed, steps=None, batch=256, hidden=None, cache=None):
    """MLP regression; with `cache` (a .pt path) the trained weights are saved there, or loaded if present."""
    steps = STEPS if steps is None else steps
    hidden = int(os.environ.get("LIFT_HIDDEN", "256")) if hidden is None else hidden
    import torch
    import torch.nn as nn
    torch.set_num_threads(int(os.environ.get("LIFT_THREADS", "2")))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m, s = Xnorm.mean(0), Xnorm.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.tensor((X - m) / s, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    done = steps if cache and os.path.exists(cache) else 0
    if done:
        net.load_state_dict(torch.load(cache))
    while done < steps:
        idx = torch.tensor(rng.permutation(len(Xt)))
        for b in range(0, len(Xt), batch):
            j = idx[b:b + batch]
            loss = ((net(Xt[j]) - Yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            done += 1
            if done >= steps:
                break
    if cache and not os.path.exists(cache):
        torch.save(net.state_dict(), cache)
    net.eval()

    def act(x):
        with torch.no_grad():
            z = torch.tensor((np.atleast_2d(x) - m) / s, dtype=torch.float32)
            out = net(z).numpy()
        return out[0] if np.ndim(x) == 1 else out
    return act


def synthetic_states(env, n, rng):
    S = []
    while len(S) < n:
        env.reset()
        S.append(L.perturb(env, rng, 0.5, rng.uniform(-0.2, 0.2, 2)))
    return np.array(S, dtype=np.float32)


def offnominal_starts(env, n, rng):
    out = []
    for _ in range(n):
        env.reset()
        out.append((L.perturb(env, rng, 0.4, rng.uniform(-0.15, 0.15, 2)), env.sim.get_state().flatten().copy()))
    return out


def run(args):
    import pandas as pd
    from scipy.spatial import cKDTree
    held, seed = args
    path = os.path.join(OUT, f"p{held}_s{seed}.csv")
    if os.path.exists(path):
        return path
    data, kw = L.load()
    env = L.make_env(kw)
    tr_ops = [o for o in L.OPERATORS if o != held]
    O = np.concatenate([d["obs"] for o in tr_ops for d in data[o]])
    A = np.concatenate([d["actions"] for o in tr_ops for d in data[o]])
    opid = np.concatenate([np.full(len(d["obs"]), i) for i, o in enumerate(tr_ops) for d in data[o]])
    rng = np.random.default_rng(mseed("lift", held, seed))
    S = synthetic_states(env, int(AUG_FRAC * len(O)), rng)
    lab = L.labeler
    bc = train(O, A, O, mseed("lift", held, seed, "BC"))
    blend = train(np.concatenate([O, S]), np.concatenate([A, lab(S)]), O, mseed("lift", held, seed, "Blend"))
    rnet = train(np.concatenate([O, S]), np.concatenate([A - lab(O), np.zeros((len(S), 7), np.float32)]), O,
                 mseed("lift", held, seed, "Residual"))
    mu, sd = O.mean(0), O.std(0) + 1e-6
    Z = (O - mu) / sd
    tree = cKDTree(Z)
    dd = np.empty(len(Z))
    for i in np.unique(opid):
        a = opid == i
        dd[a] = cKDTree(Z[~a]).query(Z[a])[0]
    thr = float(np.percentile(dd, 95))
    lo_e, hi_e = O[:, 0:3].min(0) - 0.02, O[:, 0:3].max(0) + 0.02
    lo_c, hi_c = O[:, 9:11].min(0) - 0.02, O[:, 9:11].max(0) + 0.02

    def in_workspace(x):
        return bool(np.all(x[0:3] >= lo_e) and np.all(x[0:3] <= hi_e) and np.all(x[9:11] >= lo_c) and np.all(x[9:11] <= hi_c))

    def in_support(x):
        return tree.query((x - mu) / sd)[0] <= thr

    pols = {"BC": bc, "Blend-box": blend,
            "Residual-box": lambda x: lab(x) + rnet(x),
            "Switch": lambda x: bc(x) if in_workspace(x) else lab(x),
            "Support": lambda x: bc(x) if (in_workspace(x) and in_support(x)) else lab(x),
            "Labeler": lab}
    erng = np.random.default_rng(mseed("lift-eval", held))     # same starts for every method and seed
    nom_states = []
    for _ in range(N_EVAL):
        env.reset()
        nom_states.append(env.sim.get_state().flatten().copy())
    off = offnominal_starts(env, N_EVAL, erng)
    rows = []
    for m, p in pols.items():
        r = dict(held_out=held, seed=seed, method=m)
        succ, kept, kept_sup, bcfrac = [], [], [], []
        for st in nom_states:
            X, U, s = L.rollout(env, p, state=st)
            succ.append(s)
            kept.append(all(in_workspace(x) for x in X))
            kept_sup.append(all(in_support(x) for x in X))
            if m == "Support":
                bcfrac.append(np.mean([in_workspace(x) and in_support(x) for x in X]))
        r.update(nom_success=np.mean(succ), retention=np.mean(kept), retention_support=np.mean(kept_sup), support_bc_frac=np.mean(bcfrac) if bcfrac else np.nan)
        rec = []
        for _, st in off:
            X, U, s = L.rollout(env, p, state=st)
            rec.append(s)
        r["recovery"] = np.mean(rec)
        dist, fs, dt = [], [], []
        for d in data[held][::2]:
            X, U, s = L.rollout(env, p, state=d["states"][0], horizon=max(L.HORIZON, len(d["obs"]) + 20))
            P, R = X[:, 0:3], d["obs"][:, 0:3]
            T = max(len(P), len(R))
            P = np.concatenate([P, np.repeat(P[-1:], T - len(P), 0)])
            R = np.concatenate([R, np.repeat(R[-1:], T - len(R), 0)])
            dist.append(np.sqrt(np.mean(np.sum((P - R) ** 2, 1))))
            fs.append(s)
            dt.append(len(X) - len(d["obs"]))
        r.update(fid_dist=float(np.mean(dist)), fid_success=float(np.mean(fs)), fid_dtime=float(np.mean(np.abs(dt))))
        # open-loop action error on the held-out operator's frames
        Oh = np.concatenate([d["obs"] for d in data[held]])
        Ah = np.concatenate([d["actions"] for d in data[held]])
        Ph = np.array([p(x) for x in Oh[::5]])
        r["act_mse"] = float(np.mean((np.clip(Ph, -1, 1) - Ah[::5]) ** 2))
        rows.append(r)
        print(held, seed, m, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    jobs = [(op, int(s)) for s in a.seeds.split(",") for op in L.OPERATORS]
    with ProcessPoolExecutor(a.workers) as ex:
        for p in ex.map(run, jobs):
            print("done", p, flush=True)


if __name__ == "__main__":
    main()
