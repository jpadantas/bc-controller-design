"""Look-ahead switch on robomimic Lift: the recommended full-state monitor "where a model is available".

Per (held-out operator, seed): the same BC policy as lift_study.py (same data, seed and recipe) and the same
support statistics; at every step the simulator state is saved, BC is rolled forward H steps, and BC's action is
applied only if every predicted state stays in the demonstrated workspace and in the operators' support;
otherwise the labeler acts. The simulator is then restored. Same evaluation as lift_study.py.
Writes outputs/robo/<LIFT_OUT>_la/p{op}_s{seed}.csv.

    MUJOCO_GL=disable ROBO_DATA=<path> LIFT_OUT=lift_big LIFT_STEPS=60000 LIFT_HIDDEN=1024 python lift_lookahead.py --workers 6
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lift_common as L  # noqa: E402
import lift_study as S  # noqa: E402

H = 10   # 0.5 s at 20 Hz


def run(args):
    import pandas as pd
    from scipy.spatial import cKDTree
    held, seed = args
    out = S.OUT + "_la"
    path = os.path.join(out, f"p{held}_s{seed}.csv")
    if os.path.exists(path):
        return path
    data, kw = L.load()
    env = L.make_env(kw)
    tr_ops = [o for o in L.OPERATORS if o != held]
    O = np.concatenate([d["obs"] for o in tr_ops for d in data[o]])
    A = np.concatenate([d["actions"] for o in tr_ops for d in data[o]])
    opid = np.concatenate([np.full(len(d["obs"]), i) for i, o in enumerate(tr_ops) for d in data[o]])
    bc = S.train(O, A, O, S.mseed("lift", held, seed, "BC"))
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

    def ok(x):
        return bool(np.all(x[0:3] >= lo_e) and np.all(x[0:3] <= hi_e) and np.all(x[9:11] >= lo_c)
                    and np.all(x[9:11] <= hi_c)) and tree.query((x - mu) / sd)[0] <= thr

    lab = L.labeler
    stats = {"bc": 0, "all": 0}

    def lookahead(x):
        stats["all"] += 1
        if not ok(x):
            return lab(x)
        saved = env.sim.get_state()
        xp, safe = x, True
        for _ in range(H):
            o, _, _, _ = env.step(np.clip(bc(xp), -1, 1))
            xp = L.obs_vec(o)
            if not ok(xp):
                safe = False
                break
        env.sim.set_state(saved)
        env.sim.forward()
        env.robots[0].controller.update(force=True)
        env.robots[0].controller.reset_goal()
        if safe:
            stats["bc"] += 1
            return bc(x)
        return lab(x)

    erng = np.random.default_rng(S.mseed("lift-eval", held))
    nom_states = []
    for _ in range(S.N_EVAL):
        env.reset()
        nom_states.append(env.sim.get_state().flatten().copy())
    off = S.offnominal_starts(env, S.N_EVAL, erng)
    r = dict(held_out=held, seed=seed, method="Look-ahead")
    succ, kept = [], []
    for st in nom_states:
        X, U, s = L.rollout(env, lookahead, state=st)
        succ.append(s)
        kept.append(all(bool(np.all(x[0:3] >= lo_e) and np.all(x[0:3] <= hi_e) and np.all(x[9:11] >= lo_c)
                             and np.all(x[9:11] <= hi_c)) for x in X))
    r.update(nom_success=np.mean(succ), retention=np.mean(kept))
    r["recovery"] = np.mean([L.rollout(env, lookahead, state=st)[2] for _, st in off])
    stats["bc"] = stats["all"] = 0
    dist, fs = [], []
    for d in data[held][::2]:
        X, U, s = L.rollout(env, lookahead, state=d["states"][0], horizon=max(L.HORIZON, len(d["obs"]) + 20))
        P, R = X[:, 0:3], d["obs"][:, 0:3]
        T = max(len(P), len(R))
        P = np.concatenate([P, np.repeat(P[-1:], T - len(P), 0)])
        R = np.concatenate([R, np.repeat(R[-1:], T - len(R), 0)])
        dist.append(np.sqrt(np.mean(np.sum((P - R) ** 2, 1))))
        fs.append(s)
    r.update(fid_dist=float(np.mean(dist)), fid_success=float(np.mean(fs)),
             bc_frac_fid=stats["bc"] / max(stats["all"], 1))
    os.makedirs(out, exist_ok=True)
    pd.DataFrame([r]).to_csv(path, index=False)
    print(held, seed, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    jobs = [(op, s) for s in S.SEEDS for op in L.OPERATORS]
    with ProcessPoolExecutor(a.workers) as ex:
        for p in ex.map(run, jobs):
            print("done", p, flush=True)


if __name__ == "__main__":
    main()
