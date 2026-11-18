"""Human-demonstration study on robomimic Square (NutAssemblySquare, multi-human): a confirmatory test.

Six operators, leave one out, three training seeds. Policies (protocol of can_study.py and ens_study.py, fixed
before this task was run; predictions in paper/notes/prediction_square_2026-11-18.md):
  BC            3x1024 MLP, 60k steps (LIFT_HIDDEN, LIFT_STEPS)
  Switch        error switch: the labeler acts when the end effector or the nut leaves the demonstrated workspace
  Support       support switch: also when the z-scored observation is farther from the training frames than the
                95th percentile of leave-one-operator-out nearest-neighbour distances
  Ensemble      ensemble switch: also when the summed action variance of five networks (BC and four more seeds)
                exceeds its 95th percentile on the training frames
  Residual-box  the labeler plus a network fitted to the operators' departure from it and to zero on synthetic
                states (arm joints perturbed by N(0, 0.5 rad), nut anywhere in a widened region; 30% of the frames)
  Labeler       the scripted law of square_common.labeler (evaluated once per held-out operator)
Evaluation, as in the Can study: 20 nominal resets (success; retention in the workspace), 20 off-nominal starts
(joints perturbed by 0.4 rad, nut anywhere in the widened region, any yaw: recovery), and the initial state of every
other one of the held-out operator's demonstrations: per rollout, success, the time-indexed RMS gap to the operator's
end-effector path and the DTW gap (fid_study.dtw).

    MUJOCO_GL=disable ROBO_SQUARE=<path> LIFT_STEPS=60000 LIFT_HIDDEN=1024 LIFT_THREADS=2 \
        python square_study.py --workers 16 --seeds 0,1,2
Networks are trained one per process (outputs/robo/square_nets), then each (unit, policy) is evaluated in its own
process (outputs/robo/square_parts); merged into outputs/robo/square/square_results.csv and square_rollouts.csv.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fid_study as F  # noqa: E402
import lift_study as S  # noqa: E402
import square_common as Q  # noqa: E402

OUT = os.path.join(S.ROOT, "outputs", "robo", os.environ.get("SQ_OUT", "square"))
NETS = OUT + "_nets"
N_EVAL = int(os.environ.get("SQ_NEVAL", "20"))
K, QE = 5, 95
AUG_FRAC = 0.3
NUT_XY = (np.array([-0.20, 0.02]), np.array([-0.02, 0.30]))     # widened nut region (operators: x -0.115..-0.110,
                                                                 # y 0.11..0.225), clear of the peg
FID_MAX = 700
METHODS = ["BC", "Switch", "Support", "Ensemble", "Residual-box", "Labeler"]


def place_nut(env, xy, yaw):
    sim = env.sim
    jid = sim.model.joint_name2id(env.nuts[env.nut_id].joints[0])
    adr = sim.model.jnt_qposadr[jid]
    sim.data.qpos[adr:adr + 2] = xy
    sim.data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]      # wxyz


def start(env, rng, joint_sd, wide):
    env.reset()
    if wide:
        place_nut(env, rng.uniform(*NUT_XY), rng.uniform(-np.pi, np.pi))
    sim = env.sim
    robot = env.robots[0]
    if joint_sd > 0:
        idx = robot._ref_joint_pos_indexes
        sim.data.qpos[idx] = sim.data.qpos[idx] + rng.normal(0, joint_sd, len(idx))
    sim.forward()
    robot.controller.update(force=True)
    robot.controller.reset_goal()
    x = Q.obs_vec(env._get_observations(force_update=True))
    return x, sim.get_state().flatten().copy()


def unit_data(held):
    data, kw = Q.load()
    tr = [o for o in Q.OPERATORS if o != held]
    O = np.concatenate([d["obs"] for o in tr for d in data[o]])
    A = np.concatenate([d["actions"] for o in tr for d in data[o]])
    opid = np.concatenate([np.full(len(d["obs"]), i) for i, o in enumerate(tr) for d in data[o]])
    return data, kw, O, A, opid


def synth(env, held, seed, n):
    rng = np.random.default_rng(S.mseed("square", held, seed))
    return np.array([start(env, rng, 0.5, True)[0] for _ in range(n)], dtype=np.float32)


def nets_for(held, seed, which, O, A, env=None):
    cache = lambda k: os.path.join(NETS, f"p{held}_s{seed}_{k}.pt")  # noqa: E731
    if which == "residual":
        Ssyn = synth(env, held, seed, int(AUG_FRAC * len(O)))
        lab = Q.labeler
        return S.train(np.concatenate([O, Ssyn]), np.concatenate([A - lab(O), np.zeros((len(Ssyn), 7), np.float32)]),
                       O, S.mseed("square", held, seed, "Residual"), cache=cache("residual"))
    k = int(which)
    return S.train(O, A, O, S.mseed("square", held, seed, "BC" if k == 0 else f"Ens{k}"), cache=cache(str(k)))


def train_job(args):
    held, seed, which = args
    os.makedirs(NETS, exist_ok=True)
    data, kw, O, A, _ = unit_data(held)
    env = Q.make_env(kw) if which == "residual" else None
    nets_for(held, seed, which, O, A, env)
    return args


def eval_job(args):
    import pandas as pd
    from scipy.spatial import cKDTree
    held, seed, m = args
    path = os.path.join(OUT + "_parts", f"p{held}_s{seed}_{m}.csv")
    if os.path.exists(path):
        return path
    data, kw, O, A, opid = unit_data(held)
    env = Q.make_env(kw)
    lab = Q.labeler
    if m != "Labeler":
        nets = [nets_for(held, seed, str(k), O, A) for k in range(K)]
        bc = nets[0]
    mu, sd = O.mean(0), O.std(0) + 1e-6
    Z = (O - mu) / sd
    tree = cKDTree(Z)
    dd = np.empty(len(Z))
    for i in np.unique(opid):
        a = opid == i
        dd[a] = cKDTree(Z[~a]).query(Z[a])[0]
    thr = float(np.percentile(dd, 95))
    lo_e, hi_e = O[:, 0:3].min(0) - 0.02, O[:, 0:3].max(0) + 0.02
    lo_c, hi_c = O[:, 9:12].min(0) - 0.02, O[:, 9:12].max(0) + 0.02

    def in_workspace(x):
        return bool(np.all(x[0:3] >= lo_e) and np.all(x[0:3] <= hi_e) and np.all(x[9:12] >= lo_c)
                    and np.all(x[9:12] <= hi_c))

    gates = {"Switch": in_workspace,
             "Support": lambda x: in_workspace(x) and tree.query((x - mu) / sd)[0] <= thr}
    if m != "Labeler":
        P = np.stack([n(O) for n in nets])
        ethr = float(np.percentile(P.var(0).sum(1), QE))
        gates["Ensemble"] = (lambda x: in_workspace(x) and float(np.stack([n(x) for n in nets]).var(0).sum()) <= ethr)
    if m == "BC":
        pol = bc
    elif m == "Labeler":
        pol = lab
    elif m == "Residual-box":
        rnet = nets_for(held, seed, "residual", O, A, env)
        pol = lambda x: lab(x) + rnet(x)  # noqa: E731
    else:
        g = gates[m]
        pol = lambda x: bc(x) if g(x) else lab(x)  # noqa: E731
    erng = np.random.default_rng(S.mseed("square-eval", held))           # same starts for every method and seed
    nom = [start(env, erng, 0.0, False)[1] for _ in range(N_EVAL)]
    off = [start(env, erng, 0.4, True)[1] for _ in range(N_EVAL)]
    rows = []
    for kind, starts in (("nom", nom), ("off", off)):
        for j, st in enumerate(starts):
            X, U, s = Q.rollout(env, pol, state=st)
            rows.append(dict(held_out=held, seed=seed, method=m, kind=kind, start=j, success=float(s),
                             kept=float(all(in_workspace(x) for x in X)),
                             bc_frac=float(np.mean([gates[m](x) for x in X])) if m in gates else np.nan))
    for j, d in enumerate(data[held][::2]):
        X, U, s = Q.rollout(env, pol, state=d["states"][0], horizon=min(FID_MAX, max(Q.HORIZON, len(d["obs"]) + 20)))
        Pp, R = X[:, 0:3], d["obs"][:, 0:3]
        T = max(len(Pp), len(R))
        Pa = np.concatenate([Pp, np.repeat(Pp[-1:], T - len(Pp), 0)])
        Ra = np.concatenate([R, np.repeat(R[-1:], T - len(R), 0)])
        rows.append(dict(held_out=held, seed=seed, method=m, kind="fid", start=j, success=float(s),
                         rms=float(np.sqrt(np.mean(np.sum((Pa - Ra) ** 2, 1)))), dtw=F.dtw(Pp, R),
                         bc_frac=float(np.mean([gates[m](x) for x in X])) if m in gates else np.nan))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    r = pd.DataFrame(rows)
    print(held, seed, m, r.groupby("kind").success.mean().round(2).to_dict(), flush=True)
    return path


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    units = [(op, s) for s in seeds for op in Q.OPERATORS]
    tjobs = [u + (w,) for u in units for w in [str(k) for k in range(K)] + ["residual"]]
    with ProcessPoolExecutor(a.workers) as ex:
        for j in ex.map(train_job, tjobs[::-1] if a.reverse else tjobs):
            print("trained", j, flush=True)
    jobs = [u + (m,) for u in units for m in METHODS if m != "Labeler" or u[1] == seeds[0]]
    jobs.sort(key=lambda j: j[2] not in ("Ensemble", "Support"))
    with ProcessPoolExecutor(a.workers) as ex:
        for p in ex.map(eval_job, jobs[::-1] if a.reverse else jobs):
            print("done", p, flush=True)
    os.makedirs(OUT, exist_ok=True)
    d = pd.concat([pd.read_csv(os.path.join(OUT + "_parts", f"p{op}_s{s}_{m}.csv")) for op, s, m in jobs])
    d.to_csv(os.path.join(OUT, "square_rollouts.csv"), index=False)


if __name__ == "__main__":
    main()
