"""An uncertainty monitor on robomimic Lift and Can: does a better monitor help where the cloned policy fails?

Per (task, held-out operator, training seed): BC as in lift_study.py / can_study.py (same network, steps and seed)
plus four more networks trained on the same data with other seeds. The ensemble switch lets BC act unless the
state leaves the demonstrated workspace or the ensemble's disagreement (sum over action dimensions of the
variance of the five predicted actions) exceeds its 95th percentile on the training frames (as ThriftyDAgger
flags novel states); otherwise the labeler acts. Compared, on the same networks and starts, with BC, the support
switch (as in the main studies) and the labeler alone (evaluated once per held-out operator).
Metrics as in the main studies: nominal success and retention, off-nominal recovery, distance to the held-out
operator's end-effector path, and the fraction of steps in which BC acts.

Two phases: the networks are trained one per process and kept in outputs/robo/ens_nets, then each
(unit, policy) is evaluated in its own process; results go to outputs/robo/ens/{task}_p{op}_s{seed}.csv.

    MUJOCO_GL=disable ROBO_DATA=<lift path> ROBO_CAN=<can path> LIFT_STEPS=60000 LIFT_HIDDEN=1024 LIFT_THREADS=2 \
        python ens_study.py --workers 16 --seeds 0,1,2
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import can_common as C  # noqa: E402
import can_study as CS  # noqa: E402
import lift_common as L  # noqa: E402
import lift_study as S  # noqa: E402

OUT = os.path.join(S.ROOT, "outputs", "robo", os.environ.get("ENS_OUT", "ens"))
NETS = OUT + "_nets"
K = 5                       # ensemble size (member 0 is BC itself)
Q = 95                      # disagreement percentile on the training frames
N_EVAL = int(os.environ.get("ENS_NEVAL", "20"))
METHODS = ["BC", "Support", "Ensemble", "Labeler"]


def task_data(task, held):
    data, kw = (L.load() if task == "lift" else C.load())
    tr_ops = [o for o in L.OPERATORS if o != held]
    O = np.concatenate([d["obs"] for o in tr_ops for d in data[o]])
    A = np.concatenate([d["actions"] for o in tr_ops for d in data[o]])
    opid = np.concatenate([np.full(len(d["obs"]), i) for i, o in enumerate(tr_ops) for d in data[o]])
    return data, kw, O, A, opid


def net(task, held, seed, k, O, A):
    name = "BC" if k == 0 else f"Ens{k}"
    return S.train(O, A, O, S.mseed(task, held, seed, name), cache=os.path.join(NETS, f"{task}_p{held}_s{seed}_{k}.pt"))


def train_job(args):
    task, held, seed, k = args
    os.makedirs(NETS, exist_ok=True)
    _, _, O, A, _ = task_data(task, held)
    net(task, held, seed, k, O, A)
    return args


def eval_job(args):
    import pandas as pd
    from scipy.spatial import cKDTree
    task, held, seed, m = args
    path = os.path.join(OUT + "_parts", f"{task}_p{held}_s{seed}_{m}.csv")
    if os.path.exists(path):
        return path
    data, kw, O, A, opid = task_data(task, held)
    lab = L.labeler if task == "lift" else C.labeler
    nets = [net(task, held, seed, k, O, A) for k in range(K)] if m != "Labeler" else []
    bc = nets[0] if nets else None
    mu, sd = O.mean(0), O.std(0) + 1e-6
    Z = (O - mu) / sd
    tree = cKDTree(Z)
    dd = np.empty(len(Z))
    for i in np.unique(opid):
        a = opid == i
        dd[a] = cKDTree(Z[~a]).query(Z[a])[0]
    thr = float(np.percentile(dd, 95))
    cdim = slice(9, 11) if task == "lift" else slice(9, 12)
    lo_e, hi_e = O[:, 0:3].min(0) - 0.02, O[:, 0:3].max(0) + 0.02
    lo_c, hi_c = O[:, cdim].min(0) - 0.02, O[:, cdim].max(0) + 0.02

    def in_workspace(x):
        return bool(np.all(x[0:3] >= lo_e) and np.all(x[0:3] <= hi_e) and np.all(x[cdim] >= lo_c)
                    and np.all(x[cdim] <= hi_c))

    def in_support(x):
        return tree.query((x - mu) / sd)[0] <= thr

    if nets:
        P = np.stack([n(O) for n in nets])                       # (K, frames, 7)
        ethr = float(np.percentile(P.var(0).sum(1), Q))

    def confident(x):
        return float(np.stack([n(x) for n in nets]).var(0).sum()) <= ethr

    gates = {"Support": lambda x: in_workspace(x) and in_support(x),
             "Ensemble": lambda x: in_workspace(x) and confident(x)}
    if m == "BC":
        pol = bc
    elif m == "Labeler":
        pol = lab
    else:
        g = gates[m]
        pol = lambda x: bc(x) if g(x) else lab(x)  # noqa: E731
    env = L.make_env(kw) if task == "lift" else C.make_env(kw)
    erng = np.random.default_rng(S.mseed(task + "-ens-eval", held))     # same starts for every method and seed
    if task == "lift":
        nom = []
        for _ in range(N_EVAL):
            env.reset()
            nom.append(env.sim.get_state().flatten().copy())
        off = [st for _, st in S.offnominal_starts(env, N_EVAL, erng)]
        roll = L.rollout
        fid_h = lambda d: max(L.HORIZON, len(d["obs"]) + 20)  # noqa: E731
    else:
        nom = [CS.start(env, erng, 0.0, False)[1] for _ in range(N_EVAL)]
        off = [CS.start(env, erng, 0.4, True)[1] for _ in range(N_EVAL)]
        roll = C.rollout
        fid_h = lambda d: min(CS.FID_MAX, max(C.HORIZON, len(d["obs"]) + 20))  # noqa: E731
    r = dict(task=task, held_out=held, seed=seed, method=m)
    succ, kept, frac = [], [], []
    for st in nom:
        X, U, s = roll(env, pol, state=st)
        succ.append(s)
        kept.append(all(in_workspace(x) for x in X))
        if m in gates:
            frac.append(np.mean([gates[m](x) for x in X]))
    r.update(nom_success=np.mean(succ), retention=np.mean(kept), bc_frac=np.mean(frac) if frac else np.nan)
    r["recovery"] = np.mean([roll(env, pol, state=st)[2] for st in off])
    dist, fs = [], []
    for d in data[held][::int(os.environ.get("ENS_FIDSTEP", "2"))]:
        X, U, s = roll(env, pol, state=d["states"][0], horizon=fid_h(d))
        Pp, R = X[:, 0:3], d["obs"][:, 0:3]
        T = max(len(Pp), len(R))
        Pp = np.concatenate([Pp, np.repeat(Pp[-1:], T - len(Pp), 0)])
        R = np.concatenate([R, np.repeat(R[-1:], T - len(R), 0)])
        dist.append(np.sqrt(np.mean(np.sum((Pp - R) ** 2, 1))))
        fs.append(s)
    r.update(fid_dist=float(np.mean(dist)), fid_success=float(np.mean(fs)))
    print(task, held, seed, m, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([r]).to_csv(path, index=False)
    return path


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tasks", default="can,lift")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    units = [(t, op, s) for t in a.tasks.split(",") for s in seeds for op in L.OPERATORS]
    with ProcessPoolExecutor(a.workers) as ex:
        for j in ex.map(train_job, [u + (k,) for u in units for k in range(K)]):
            print("trained", j, flush=True)
    jobs = [u + (m,) for u in units for m in METHODS if m != "Labeler" or u[2] == seeds[0]]
    with ProcessPoolExecutor(a.workers) as ex:
        for p in ex.map(eval_job, jobs):
            print("done", p, flush=True)
    rows = [pd.read_csv(os.path.join(OUT + "_parts", f"{t}_p{op}_s{s}_{m}.csv")) for t, op, s, m in jobs]
    os.makedirs(OUT, exist_ok=True)
    pd.concat(rows).to_csv(os.path.join(OUT, "ens_results.csv"), index=False)


if __name__ == "__main__":
    main()
