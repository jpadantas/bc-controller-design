"""Human-demonstration study on robomimic Can (PickPlaceCan, multi-human): six operators, leave one out.

The protocol of lift_study.py on a longer pick-and-place task, with both a feed-forward BC policy and an
action-chunking one (lift_chunk.py), each alone and behind the error and support switches, plus the box and
residual blends and the scripted labeler (can_common.labeler). Same networks, steps, monitors and metrics:
  nominal   20 resets: success, retention (never leaves the demonstrated workspace: end effector and can
            position within the training demonstrations' range + 2 cm)
  offnom    20 starts with the arm's joints perturbed (sd 0.4 rad) and the can anywhere in its bin: recovery
  fidelity  from the initial state of every other one of the held-out operator's demonstrations: RMS distance of
            the end-effector path to the recorded one, and success
Writes outputs/robo/<CAN_OUT>/p{op}_s{seed}.csv.

    MUJOCO_GL=disable ROBO_CAN=<path> LIFT_STEPS=60000 LIFT_HIDDEN=1024 python can_study.py --workers 30 --seeds 0,1,2,3,4
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import can_common as C  # noqa: E402
import lift_chunk as LK  # noqa: E402
import lift_study as S  # noqa: E402

OUT = os.path.join(S.ROOT, "outputs", "robo", os.environ.get("CAN_OUT", "can"))
N_EVAL = int(os.environ.get("CAN_NEVAL", "20"))
AUG_FRAC = 0.3
CAN_XY = (np.array([-0.07, -0.47]), np.array([0.27, -0.03]))   # the can's bin, 2.5 cm from its walls
FID_MAX = 600                                                     # cap on fidelity rollouts (steps)


def place_can(env, xy):
    sim = env.sim
    jid = sim.model.joint_name2id(env.objects[0].joints[0])
    adr = sim.model.jnt_qposadr[jid]
    sim.data.qpos[adr:adr + 2] = xy


def start(env, rng, joint_sd, wide_can):
    env.reset()
    if wide_can:
        place_can(env, rng.uniform(*CAN_XY))
    x = C.perturb(env, rng, joint_sd)
    return x, env.sim.get_state().flatten().copy()


def run(args):
    import pandas as pd
    import torch
    from scipy.spatial import cKDTree
    torch.set_num_threads(1)
    held, seed = args
    path = os.path.join(OUT, f"p{held}_s{seed}.csv")
    if os.path.exists(path):
        return path
    data, kw = C.load()
    env = C.make_env(kw)
    tr_ops = [o for o in C.OPERATORS if o != held]
    demos = [d for o in tr_ops for d in data[o]]
    O = np.concatenate([d["obs"] for d in demos])
    A = np.concatenate([d["actions"] for d in demos])
    opid = np.concatenate([np.full(len(d["obs"]), i) for i, o in enumerate(tr_ops) for d in data[o]])
    rng = np.random.default_rng(S.mseed("can", held, seed))
    Ssyn = np.array([start(env, rng, 0.5, True)[0] for _ in range(int(AUG_FRAC * len(O)))], dtype=np.float32)
    lab = C.labeler
    bc = S.train(O, A, O, S.mseed("can", held, seed, "BC"))
    LK.K = 8
    cnet = S.train(O, LK.chunk_targets(demos), O, S.mseed("can", held, seed, "BC-chunk"))
    blend = S.train(np.concatenate([O, Ssyn]), np.concatenate([A, lab(Ssyn)]), O, S.mseed("can", held, seed, "Blend"))
    rnet = S.train(np.concatenate([O, Ssyn]), np.concatenate([A - lab(O), np.zeros((len(Ssyn), 7), np.float32)]), O,
                   S.mseed("can", held, seed, "Residual"))
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

    def in_support(x):
        return tree.query((x - mu) / sd)[0] <= thr

    class Gated:
        def __init__(self, cond):
            self.bc, self.cond = LK.Chunked(cnet), cond

        def reset(self):
            self.bc.reset()

        def __call__(self, x):
            a = self.bc(x)                      # always queried, so the chunk buffer stays current
            return a if self.cond(x) else lab(x)

    full = lambda x: in_workspace(x) and in_support(x)  # noqa: E731
    pols = {"BC": bc, "Blend-box": blend, "Residual-box": lambda x: lab(x) + rnet(x),
            "Switch": lambda x: bc(x) if in_workspace(x) else lab(x),
            "Support": lambda x: bc(x) if full(x) else lab(x),
            "BC-chunk": LK.Chunked(cnet), "Switch-chunk": Gated(in_workspace), "Support-chunk": Gated(full),
            "Labeler": lab}
    erng = np.random.default_rng(S.mseed("can-eval", held))       # same starts for every method and seed
    nom = [start(env, erng, 0.0, False)[1] for _ in range(N_EVAL)]
    off = [start(env, erng, 0.4, True)[1] for _ in range(N_EVAL)]
    Oh = np.concatenate([d["obs"] for d in data[held]])
    Ah = np.concatenate([d["actions"] for d in data[held]])
    rows = []
    for m, p in pols.items():
        r = dict(held_out=held, seed=seed, method=m)
        succ, kept, bcfrac = [], [], []
        for st in nom:
            X, U, s = C.rollout(env, p, state=st)
            succ.append(s)
            kept.append(all(in_workspace(x) for x in X))
            if m.startswith("Support"):
                bcfrac.append(np.mean([full(x) for x in X]))
        r.update(nom_success=np.mean(succ), retention=np.mean(kept),
                 support_bc_frac=np.mean(bcfrac) if bcfrac else np.nan)
        r["recovery"] = np.mean([C.rollout(env, p, state=st)[2] for st in off])
        dist, fs = [], []
        for d in data[held][::int(os.environ.get("CAN_FIDSTEP", "2"))]:
            X, U, s = C.rollout(env, p, state=d["states"][0], horizon=min(FID_MAX, max(C.HORIZON, len(d["obs"]) + 20)))
            P, R = X[:, 0:3], d["obs"][:, 0:3]
            T = max(len(P), len(R))
            P = np.concatenate([P, np.repeat(P[-1:], T - len(P), 0)])
            R = np.concatenate([R, np.repeat(R[-1:], T - len(R), 0)])
            dist.append(np.sqrt(np.mean(np.sum((P - R) ** 2, 1))))
            fs.append(s)
        r.update(fid_dist=float(np.mean(dist)), fid_success=float(np.mean(fs)))
        if "chunk" not in m:
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
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    jobs = [(op, int(s)) for s in a.seeds.split(",") for op in C.OPERATORS]
    with ProcessPoolExecutor(a.workers) as ex:
        for p in ex.map(run, jobs):
            print("done", p, flush=True)


if __name__ == "__main__":
    main()
