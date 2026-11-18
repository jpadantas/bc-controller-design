"""Fidelity per rollout on Lift and Can, and the support switch's threshold (reviewer requests).

On the networks of ens_study.py (BC = member 0, the same BC as the main studies), per (task, held-out operator, seed):
  - from every other initial state of the held-out operator, each policy's rollout is logged with its success, the
    time-indexed RMS gap to the operator's end-effector path (the shorter path held at its last point, as in the
    main studies) and the DTW gap (mean distance along the dynamic-time-warping alignment), so that fidelity can be
    compared on successful rollouts only and without penalizing a different speed;
  - the support switch is also run with its threshold at the 80th, 90th and 99th percentile (default 95) from the
    nominal and off-nominal starts of ens_study.py.
Policies: BC, error switch (workspace only), support switch @80/90/95/99, ensemble switch, labeler (first seed only).

    MUJOCO_GL=disable ROBO_DATA=<lift> ROBO_CAN=<can> LIFT_STEPS=60000 LIFT_HIDDEN=1024 LIFT_THREADS=2 \
        python fid_study.py --workers 16 --seeds 0,1,2
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import can_common as C  # noqa: E402
import can_study as CS  # noqa: E402
import ens_study as E  # noqa: E402
import lift_common as L  # noqa: E402
import lift_study as S  # noqa: E402

OUT = os.path.join(S.ROOT, "outputs", "robo", "fid")
QS = (80, 90, 95, 99)
METHODS = ["BC", "Switch", "Support@80", "Support@90", "Support@95", "Support@99", "Ensemble", "Labeler"]
FULL = {"Support@80", "Support@90", "Support@99"}      # also evaluated from nominal and off-nominal starts


def dtw(P, R):
    """Mean point distance along the optimal DTW alignment of two paths (n x 3, m x 3)."""
    Dm = np.sqrt(((P[:, None, :] - R[None, :, :]) ** 2).sum(-1))
    n, m = Dm.shape
    acc = np.full((n + 1, m + 1), np.inf)
    ln = np.zeros((n + 1, m + 1))
    acc[0, 0] = 0.0
    acc, ln, Dm = acc.tolist(), ln.tolist(), Dm.tolist()
    for i in range(1, n + 1):
        a0, a1, l0, l1, d = acc[i - 1], acc[i], ln[i - 1], ln[i], Dm[i - 1]
        for j in range(1, m + 1):
            best, bl = a0[j - 1], l0[j - 1]
            if a0[j] < best:
                best, bl = a0[j], l0[j]
            if a1[j - 1] < best:
                best, bl = a1[j - 1], l1[j - 1]
            a1[j] = d[j - 1] + best
            l1[j] = bl + 1
    return float(acc[n][m] / ln[n][m])


def job(args):
    import pandas as pd
    from scipy.spatial import cKDTree
    task, held, seed, m = args
    path = os.path.join(OUT + "_parts", f"{task}_p{held}_s{seed}_{m}.csv")
    if os.path.exists(path):
        return path
    data, kw, O, A, opid = E.task_data(task, held)
    lab = L.labeler if task == "lift" else C.labeler
    nets = [E.net(task, held, seed, k, O, A) for k in range(E.K)] if m != "Labeler" else []
    bc = nets[0] if nets else None
    mu, sd = O.mean(0), O.std(0) + 1e-6
    Z = (O - mu) / sd
    tree = cKDTree(Z)
    dd = np.empty(len(Z))
    for i in np.unique(opid):
        a = opid == i
        dd[a] = cKDTree(Z[~a]).query(Z[a])[0]
    cdim = slice(9, 11) if task == "lift" else slice(9, 12)
    lo_e, hi_e = O[:, 0:3].min(0) - 0.02, O[:, 0:3].max(0) + 0.02
    lo_c, hi_c = O[:, cdim].min(0) - 0.02, O[:, cdim].max(0) + 0.02

    def in_workspace(x):
        return bool(np.all(x[0:3] >= lo_e) and np.all(x[0:3] <= hi_e) and np.all(x[cdim] >= lo_c)
                    and np.all(x[cdim] <= hi_c))

    gates = {"Switch": in_workspace}
    for q in QS:
        thr = float(np.percentile(dd, q))
        gates[f"Support@{q}"] = (lambda x, thr=thr: in_workspace(x) and tree.query((x - mu) / sd)[0] <= thr)
    if nets:
        P = np.stack([n(O) for n in nets])
        ethr = float(np.percentile(P.var(0).sum(1), E.Q))
        gates["Ensemble"] = (lambda x: in_workspace(x)
                             and float(np.stack([n(x) for n in nets]).var(0).sum()) <= ethr)
    if m == "BC":
        pol = bc
    elif m == "Labeler":
        pol = lab
    else:
        g = gates[m]
        pol = lambda x: bc(x) if g(x) else lab(x)  # noqa: E731
    env = L.make_env(kw) if task == "lift" else C.make_env(kw)
    if task == "lift":
        roll = L.rollout
        fid_h = lambda d: max(L.HORIZON, len(d["obs"]) + 20)  # noqa: E731
    else:
        roll = C.rollout
        fid_h = lambda d: min(CS.FID_MAX, max(C.HORIZON, len(d["obs"]) + 20))  # noqa: E731
    rows = []
    for j, d in enumerate(data[held][::2]):
        X, U, s = roll(env, pol, state=d["states"][0], horizon=fid_h(d))
        Pp, R = X[:, 0:3], d["obs"][:, 0:3]
        T = max(len(Pp), len(R))
        Pa = np.concatenate([Pp, np.repeat(Pp[-1:], T - len(Pp), 0)])
        Ra = np.concatenate([R, np.repeat(R[-1:], T - len(R), 0)])
        rows.append(dict(task=task, held_out=held, seed=seed, method=m, kind="fid", start=j, success=float(s),
                         rms=float(np.sqrt(np.mean(np.sum((Pa - Ra) ** 2, 1)))), dtw=dtw(Pp, R),
                         steps=len(Pp), ref_steps=len(R),
                         bc_frac=float(np.mean([gates[m](x) for x in X])) if m in gates else np.nan))
    if m in FULL:
        erng = np.random.default_rng(S.mseed(task + "-ens-eval", held))     # the starts of ens_study.py
        if task == "lift":
            nom = []
            for _ in range(E.N_EVAL):
                env.reset()
                nom.append(env.sim.get_state().flatten().copy())
            off = [st for _, st in S.offnominal_starts(env, E.N_EVAL, erng)]
        else:
            nom = [CS.start(env, erng, 0.0, False)[1] for _ in range(E.N_EVAL)]
            off = [CS.start(env, erng, 0.4, True)[1] for _ in range(E.N_EVAL)]
        for kind, starts in (("nom", nom), ("off", off)):
            for j, st in enumerate(starts):
                X, U, s = roll(env, pol, state=st)
                rows.append(dict(task=task, held_out=held, seed=seed, method=m, kind=kind, start=j, success=float(s),
                                 kept=float(all(in_workspace(x) for x in X)),
                                 bc_frac=float(np.mean([gates[m](x) for x in X]))))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(task, held, seed, m, flush=True)
    return path


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tasks", default="can,lift")
    ap.add_argument("--reverse", action="store_true", help="run the job list backwards (a second instance)")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    units = [(t, op, s) for t in a.tasks.split(",") for s in seeds for op in L.OPERATORS]
    jobs = [u + (m,) for u in units for m in METHODS if m != "Labeler" or u[2] == seeds[0]]
    jobs.sort(key=lambda j: j[3] not in FULL and j[3] != "Ensemble")      # the slowest first
    with ProcessPoolExecutor(a.workers) as ex:
        for p in ex.map(job, jobs[::-1] if a.reverse else jobs):
            print("done", p, flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.concat([pd.read_csv(os.path.join(OUT + "_parts", f"{t}_p{op}_s{s}_{m}.csv")) for t, op, s, m in jobs]) \
        .to_csv(os.path.join(OUT, "fid_rollouts.csv"), index=False)


if __name__ == "__main__":
    main()
