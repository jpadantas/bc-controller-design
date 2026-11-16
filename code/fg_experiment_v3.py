"""Flight case study, version 3 (addresses the round-7 reviews).

Changes from fg_experiment.py:
  * the policies output the pilots' aileron and elevator TRIM as well as stick,
    rudder and throttle (open-loop replay in FlightGear matches the log only with
    trims; see fg_replay.py), so that BC is not asked to reproduce stick inputs
    that compensated a trim it cannot set;
  * inputs are z-scored with the pilots' frames only, for every policy;
  * 3 training seeds per leave-one-pilot-out fold;
  * starts: the held-out pilot's first frame and four 15,000 ft displacements;
  * more baselines:
      Switch      -- BC within 10,000 ft of the slot, labeler beyond (as before)
      Simplex     -- runtime assurance on a safety envelope with hysteresis: the
                     labeler takes over when |roll| > 75 deg, pitch < -20 deg or the
                     slot error exceeds 10,000 ft, and hands back to BC only when
                     |roll| < 30 deg, pitch > -5 deg and the slot error < 8,000 ft
      SoftGate    -- convex combination (1-w) BC + w labeler, w ramping linearly
                     from 0 at 8,000 ft to 1 at 12,000 ft
      Blend       -- pilots + synthetic states labeled by the labeler (as before:
                     far slot offsets, every other input over its logged range)
      BlendPos    -- ablation: far slot offsets, every other input copied from a
                     random logged pilot frame (nominal attitude, speed, rates)
      Labeler     -- the formation law alone
  * runs N FlightGear instances in parallel (FG_INSTANCE ports).

    python fg_experiment_v3.py --workers 6
"""
import argparse
import os
import subprocess
import sys
import zlib

import numpy as np
import pandas as pd

OFFSETS = [(0.0, 0.0), (15000.0, 0.0), (-15000.0, 0.0), (0.0, 15000.0), (0.0, -15000.0)]
LEARNED = ["BC", "Blend", "BlendPos", "Switch", "Simplex", "SoftGate"]
SEEDS = [0, 1, 2]
PILOTS = [1, 2, 3, 4, 5]
AUG_FRACTION = 0.3
INPUTS = ["dx_body", "dy_body", "dz_leader", "roll_deg", "pitch_deg", "rel_hdg", "airspeed_kt",
          "vertical_speed_fps", "pilot_z_accel_fps_sec"]
OUTPUTS = ["aileron", "elevator", "rudder", "throttle", "aileron_trim", "elevator_trim"]


def mseed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31)


def out_dir():
    import flight_unified as F
    return os.path.join(F.ROOT, "outputs", "flightgear", "experiment_v3")


def load():
    import flight_unified as F
    df = F.prepare(pd.read_csv(F.DATA))
    df["rel_hdg"] = ((df.heading_deg - df.leader_heading + 180.0) % 360.0) - 180.0
    return df


def far_offsets(n, rng, slot):
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = rng.uniform(10000.0, 60000.0, n)
    P = np.empty((n, 3))
    P[:, 0] = slot[0] + rad * np.cos(ang)
    P[:, 1] = slot[1] + rad * np.sin(ang)
    P[:, 2] = slot[2] + rng.uniform(-3000.0, 3000.0, n)
    return P


def synthetic_box(X, n, rng, slot):
    lo, hi = X.min(0), X.max(0)
    S = np.column_stack([rng.uniform(lo[j], hi[j], n) for j in range(X.shape[1])])
    S[:, :3] = far_offsets(n, rng, slot)
    S[:, 5] = rng.uniform(-90.0, 90.0, n)
    return S


def synthetic_pos(X, n, rng, slot):
    S = X[rng.integers(0, len(X), n)].copy()
    S[:, :3] = far_offsets(n, rng, slot)
    return S


def train(X, Y, Xnorm, seed, epochs=50, batch=128):
    import torch
    import flight_unified as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m, s = F._norm(Xnorm)
    Xt = torch.tensor((X - m) / s, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    net = F.MLP(X.shape[1], Y.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        order = torch.tensor(rng.permutation(len(Xt)))
        for st in range(0, len(Xt), batch):
            b = order[st:st + batch]
            loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return F.wrap(net, m, s)


def labeler6(ctrl):
    def f(X):
        Z = np.atleast_2d(X)
        return np.concatenate([np.atleast_2d(ctrl(Z)), np.zeros((len(Z), 2))], axis=1)
    return f


class Simplex:
    def __init__(self, bc, lab, slot):
        self.bc, self.lab, self.slot, self.safe_mode = bc, lab, slot, False

    def __call__(self, x):
        x = np.asarray(x, dtype=float).ravel()
        e = float(np.linalg.norm(x[:3] - self.slot))
        roll, pitch = abs(x[3]), x[4]
        if not self.safe_mode and (roll > 75.0 or pitch < -20.0 or e > 10000.0):
            self.safe_mode = True
        elif self.safe_mode and roll < 30.0 and pitch > -5.0 and e < 8000.0:
            self.safe_mode = False
        return (self.lab if self.safe_mode else self.bc)(x[None])[0]


def policies(df, k, seed):
    """All learned policies for fold k (held-out pilot) and training seed."""
    import fg_bridge as B
    tr = df[df.pilot_id != k]
    X, Y = tr[INPUTS].values, tr[OUTPUTS].values
    lab = labeler6(B.FormationController())
    rng = np.random.default_rng(mseed("fg3", k, seed))
    n = int(AUG_FRACTION * len(X))
    Sb, Sp = synthetic_box(X, n, rng, B.SLOT), synthetic_pos(X, n, rng, B.SLOT)
    bc = train(X, Y, X, mseed("fg3", k, seed, "BC"))
    blend = train(np.concatenate([X, Sb]), np.concatenate([Y, lab(Sb)]), X, mseed("fg3", k, seed, "Blend"))
    blendpos = train(np.concatenate([X, Sp]), np.concatenate([Y, lab(Sp)]), X, mseed("fg3", k, seed, "BlendPos"))

    def dist(Z):
        return np.linalg.norm(np.atleast_2d(Z)[:, :3] - B.SLOT, axis=1)

    def switch(x):
        Z = np.atleast_2d(x)
        out = bc(Z)
        far = dist(Z) > 10000.0
        out[far] = lab(Z[far])
        return out

    def softgate(x):
        Z = np.atleast_2d(x)
        w = np.clip((dist(Z) - 8000.0) / 4000.0, 0.0, 1.0)[:, None]
        return (1 - w) * bc(Z) + w * lab(Z)

    makers = {"BC": lambda: bc, "Blend": lambda: blend, "BlendPos": lambda: blendpos, "Switch": lambda: switch,
              "Simplex": lambda: Simplex(bc, lab, B.SLOT), "SoftGate": lambda: softgate}
    va = df[df.pilot_id == k]
    Xv, Yv = va[INPUTS].values, va[OUTPUTS].values
    fid = {}
    for m, mk in makers.items():
        if m == "Simplex":
            continue  # stateful; on logged frames it equals BC except where the envelope trips
        P = np.atleast_2d(mk()(Xv))
        fid[m] = dict(mse4=float(np.mean((P[:, :4] - Yv[:, :4]) ** 2)), mse6=float(np.mean((P - Yv) ** 2)))
    P = lab(Xv)
    fid["Labeler"] = dict(mse4=float(np.mean((P[:, :4] - Yv[:, :4]) ** 2)), mse6=float(np.mean((P - Yv) ** 2)))
    return makers, lab, fid, va.reset_index(drop=True)


def run_group(k, seed):
    """Worker: one (fold, seed) group; seed -1 = the labeler alone."""
    import fg_eval as E
    df = load()
    od = out_dir()
    makers, lab, fid, g = policies(df, k, max(seed, 0))
    if seed >= 0:
        pd.DataFrame([dict(held_out=k, seed=seed, method=m, **v) for m, v in fid.items()]).to_csv(
            os.path.join(od, f"fidelity_p{k}_s{seed}.csv"), index=False)
        jobs = [(m, makers[m]) for m in LEARNED]
    else:
        jobs = [("Labeler", lambda: lab)]
    for off in OFFSETS:
        for m, mk in jobs:
            path = os.path.join(od, f"p{k}_s{seed}_{m}_lat{int(off[0])}_lon{int(off[1])}.csv")
            if os.path.exists(path):
                continue
            act = mk()
            log = E.rollout(lambda x, act=act: np.atleast_2d(act(x))[0], g, off, 120.0, tag=m)
            log["held_out"], log["seed"], log["method"], log["off_lat"], log["off_lon"] = k, seed, m, off[0], off[1]
            log.to_csv(path, index=False)
            last = log.iloc[-1]
            print(f"[inst {os.environ.get('FG_INSTANCE')}] pilot {k} seed {seed} {m:9s} off={off}: "
                  f"t_end={last.t:5.1f}s err={last.slot_err:8.0f} ft", flush=True)


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
    os.makedirs(out_dir(), exist_ok=True)
    groups = [(k, s) for s in SEEDS for k in PILOTS] + [(k, -1) for k in PILOTS]
    free = list(range(a.workers))
    running = []
    import time
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
