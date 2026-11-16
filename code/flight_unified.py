"""Unified flight pipeline (ICML 2027 revision). Replaces train_*.py /
train_loso_cv.py / evaluate_openloop.py for every flight number in the paper.

Fixed protocol (decided before running):
  * Every method uses the same inputs (leader body frame), the same
    row-shuffled minibatches (128), the same network and the same 50 epochs;
    only the per-sample weights / augmentation / penalty differ.
  * Leave-one-pilot-out x 5 training seeds. One-step MSE is measured on the
    held-out pilot, and the closed-loop rollout of each fold model starts from
    the HELD-OUT pilot's initial condition (out-of-sample), perturbed by
    {0, +-5,000 ft lateral, +-5,000 ft longitudinal} in the leader's frame.
  * Closed-loop metric: distance to the formation SLOT (not to the leader) at
    30/60/120 s, and time until the slot error first exceeds 10,000 ft
    (above the 99th percentile of the human pilots' own slot error).
  * Simulator inputs fixed: the 9th policy input is the normal acceleration of a
    coordinated turn, -g/cos(roll), matching the sign/scale of the logged
    pilot_z_accel feature (the earlier evaluate_openloop.py fed +g/cos(roll)).

    python flight_unified.py --workers 10
"""
import argparse
import copy
import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(1)
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE_DIR)
OUT = os.path.join(ROOT, "outputs", "flight_unified")
DATA = os.path.join(ROOT, "outputs", "master_aligned_dataset.csv")

INPUT_COLS = ["dx_body", "dy_body", "dz_leader", "roll_deg", "pitch_deg", "heading_deg",
              "airspeed_kt", "vertical_speed_fps", "pilot_z_accel_fps_sec"]
OUTPUT_COLS = ["aileron", "elevator", "rudder", "throttle"]
SLOT = np.array([-267.7, -11249.0, 193.4])  # body-frame slot (ft), calibrated once from one pilot
TAU_STABLE, TAU_RARE = 1500.0, 10000.0
EPOCHS, BATCH = 50, 128
AUG_FRACTION = 0.3
LAMBDA_PEN = 0.05
RL_STEPS, RL_BATCH = 30_000, 256
SEEDS = list(range(5))
OFFSETS = [(0.0, 0.0), (5000.0, 0.0), (-5000.0, 0.0), (0.0, 5000.0), (0.0, -5000.0)]
HORIZON_S, DT, R_FEET = 120.0, 0.1, 20902231.0
METHODS = ["BC", "BC_pen", "GAFIL_hard", "GAFIL_smooth", "GAFIL_smooth_pen", "Oversample",
           "IQL", "TD3BC", "Aug", "Aug_partial", "GAFIL_smooth_Aug", "Labeler", "BC_Fallback"]


def mseed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31)


def prepare(df):
    df = df.copy()
    psi = np.radians(df["leader_heading"])
    dN, dE = df["dy_leader"], df["dx_leader"]
    df["dx_body"] = dE * np.cos(psi) - dN * np.sin(psi)
    df["dy_body"] = dE * np.sin(psi) + dN * np.cos(psi)
    df["err"] = np.sqrt((df.dx_body - SLOT[0]) ** 2 + (df.dy_body - SLOT[1]) ** 2 + (df.dz_leader - SLOT[2]) ** 2)
    df = df.sort_values(["pilot_id", "time_s"]).reset_index(drop=True)
    # Data cleaning (declared in the paper): the first 1-2 frames of every pilot are
    # an alignment artifact (~70,000 ft slot error, back inside formation by
    # t=0.1-0.2 s). Drop each pilot's leading frames up to the first one within
    # TAU_RARE, so neither training nor the rollout start uses the artifact.
    keep = []
    for _, g in df.groupby("pilot_id", sort=False):
        first = int(np.argmax(g.err.values < TAU_RARE))
        keep.append(g.iloc[first:])
    return pd.concat(keep).reset_index(drop=True)


def slot_error(X):
    return np.sqrt((X[:, 0] - SLOT[0]) ** 2 + (X[:, 1] - SLOT[1]) ** 2 + (X[:, 2] - SLOT[2]) ** 2)


def weights(err, ep, kind):
    w = np.ones(len(err))
    for e_id in np.unique(ep):
        idx = np.where(ep == e_id)[0]
        e = err[idx]
        pw = np.ones(len(idx))
        if kind == "hard":
            for t in range(len(e) - 1):
                pw[t] = 1.0 if e[t] < TAU_STABLE else (2.0 if e[t + 1] < e[t] else 0.0)
        else:
            rel = (e[:-1] - e[1:]) / np.maximum(e[:-1], 1.0)
            pw[:-1] = 0.5 + 0.8 / (1.0 + np.exp(-np.clip(rel / 0.15, -50, 50)))
        w[idx] = pw
    return w


def mechanism(err, ep):
    nxt = np.full(len(err), np.nan)
    for e_id in np.unique(ep):
        idx = np.where(ep == e_id)[0]
        nxt[idx[:-1]] = err[idx[1:]]
    rows = []
    for tau in (TAU_STABLE, 5000.0, TAU_RARE):
        v = ~np.isnan(nxt)
        R = (err > tau) & v
        row = dict(tau=tau, n_rare=int(R.sum()), p_tau=float((err > tau).mean()),
                   q_rare=float((nxt[R] < err[R]).mean()) if R.any() else np.nan,
                   pilots_in_rare=int(len(np.unique(ep[R]))))
        for kind in ("hard", "smooth"):
            w = weights(err, ep, kind)
            row[f"{kind}_wmean_rare"] = float(w[R].mean()) if R.any() else np.nan
            row[f"{kind}_wmean_rest"] = float(w[~R].mean())
            row[f"{kind}_ess_rare"] = float(w[R].sum() ** 2 / max((w[R] ** 2).sum(), 1e-12)) if R.any() else np.nan
            row[f"{kind}_peff_over_p"] = float((w[R].sum() / w.sum()) / (R.sum() / len(w))) if R.any() else np.nan
        rows.append(row)
    return rows


class MLP(nn.Module):
    def __init__(self, i, o, sizes=(128, 64, 32), tanh_out=False):
        super().__init__()
        layers, d = [], i
        for h in sizes:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, o))
        self.net = nn.Sequential(*layers)
        self.tanh_out = tanh_out

    def forward(self, x):
        y = self.net(x)
        return torch.tanh(y) if self.tanh_out else y


def _norm(X):
    m, s = X.mean(0), X.std(0)
    s[s == 0] = 1.0
    return m, s


def wrap(net, m, s):
    net.eval()

    def act(X):
        with torch.no_grad():
            return net(torch.tensor((np.atleast_2d(X) - m) / s, dtype=torch.float32)).numpy()
    return act


def train_bc(X, Y, w, seed, penalty=0.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m, s = _norm(X)
    Xt = torch.tensor((X - m) / s, dtype=torch.float32)
    Yt, Wt = torch.tensor(Y, dtype=torch.float32), torch.tensor(w, dtype=torch.float32)
    net = MLP(X.shape[1], Y.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(Xt)
    for _ in range(EPOCHS):
        order = torch.tensor(rng.permutation(n))
        for st in range(0, n, BATCH):
            b = order[st:st + BATCH]
            pred = net(Xt[b])
            loss = (((pred - Yt[b]) ** 2) * Wt[b].unsqueeze(1)).mean()
            if penalty > 0:
                loss = loss + penalty * (pred[:, 0] ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return wrap(net, m, s)


def _soft(t, s, tau=0.005):
    with torch.no_grad():
        for tp, sp in zip(t.parameters(), s.parameters()):
            tp.mul_(1 - tau).add_(tau * sp)


class TwinQ(nn.Module):
    def __init__(self, s, a):
        super().__init__()
        self.q1, self.q2 = MLP(s + a, 1), MLP(s + a, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def _transitions(X, ep):
    Xn = np.empty_like(X)
    done = np.zeros(len(X), dtype=np.float32)
    for e_id in np.unique(ep):
        idx = np.where(ep == e_id)[0]
        Xn[idx[:-1]] = X[idx[1:]]
        Xn[idx[-1]] = X[idx[-1]]
        done[idx[-1]] = 1.0
    r = (-slot_error(Xn) / TAU_RARE).astype(np.float32)
    return r, Xn, done


def train_offline_rl(X, Y, ep, seed, algo, gamma=0.99):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    r, Xn, done = _transitions(X, ep)
    m, s = _norm(X)
    St, Snt = (torch.tensor((z - m) / s, dtype=torch.float32) for z in (X, Xn))
    At, Rt, Dt = (torch.tensor(z, dtype=torch.float32) for z in (Y, r, done))
    pi = MLP(X.shape[1], Y.shape[1], tanh_out=True)
    Q = TwinQ(X.shape[1], Y.shape[1])
    Qt = copy.deepcopy(Q)
    oQ, oP = torch.optim.Adam(Q.parameters(), lr=3e-4), torch.optim.Adam(pi.parameters(), lr=3e-4)
    n = len(St)
    if algo == "IQL":
        V = MLP(X.shape[1], 1)
        oV = torch.optim.Adam(V.parameters(), lr=3e-4)
    else:
        pit = copy.deepcopy(pi)
    for step in range(RL_STEPS):
        b = torch.tensor(rng.integers(0, n, RL_BATCH))
        bs, ba, br, bsn, bd = St[b], At[b], Rt[b], Snt[b], Dt[b]
        if algo == "IQL":
            with torch.no_grad():
                qt = torch.minimum(*Qt(bs, ba))
            diff = qt - V(bs).squeeze(-1)
            lv = (torch.where(diff > 0, 0.7, 0.3) * diff ** 2).mean()
            oV.zero_grad(); lv.backward(); oV.step()
            with torch.no_grad():
                tgt = br + gamma * (1 - bd) * V(bsn).squeeze(-1)
            q1, q2 = Q(bs, ba)
            lq = ((q1 - tgt) ** 2).mean() + ((q2 - tgt) ** 2).mean()
            oQ.zero_grad(); lq.backward(); oQ.step()
            _soft(Qt, Q)
            with torch.no_grad():
                wt = torch.clamp(torch.exp(3.0 * (torch.minimum(*Qt(bs, ba)) - V(bs).squeeze(-1))), max=100.0)
            lp = (wt * ((pi(bs) - ba) ** 2).sum(-1)).mean()
            oP.zero_grad(); lp.backward(); oP.step()
        else:
            with torch.no_grad():
                na = (pit(bsn) + (torch.randn_like(ba) * 0.2).clamp(-0.5, 0.5)).clamp(-1.0, 1.0)
                tgt = br + gamma * (1 - bd) * torch.minimum(*Qt(bsn, na))
            q1, q2 = Q(bs, ba)
            lq = ((q1 - tgt) ** 2).mean() + ((q2 - tgt) ** 2).mean()
            oQ.zero_grad(); lq.backward(); oQ.step()
            if step % 2 == 0:
                pa = pi(bs)
                qpi = Q(bs, pa)[0]
                lp = -2.5 / (qpi.abs().mean().detach() + 1e-6) * qpi.mean() + ((pa - ba) ** 2).mean()
                oP.zero_grad(); lp.backward(); oP.step()
                _soft(Qt, Q); _soft(pit, pi)
    return wrap(pi, m, s)


def fit_gains(tr):
    def fit(x, y):
        A = np.vstack([x, np.ones_like(x)]).T
        return np.linalg.lstsq(A, y, rcond=None)[0]
    a1, a0 = fit(tr.dx_body.values - SLOT[0], tr.aileron.values)
    b1, b0 = fit(tr.dz_leader.values - SLOT[2], tr.elevator.values)
    c1, c0 = fit(tr.dy_body.values - SLOT[1], tr.throttle.values)
    return a0, a1, b0, b1, c0, c1


def labeler(gains):
    a0, a1, b0, b1, c0, c1 = gains

    def act(X):
        X = np.atleast_2d(X)
        return np.stack([np.clip(a0 + a1 * (X[:, 0] - SLOT[0]), -0.6, 0.6),
                         np.clip(b0 + b1 * (X[:, 2] - SLOT[2]), -0.6, 0.6),
                         np.zeros(len(X)),
                         np.clip(c0 + c1 * (X[:, 1] - SLOT[1]), 0.3, 1.0)], axis=1)
    return act


def aug_states(Xtr, n, rng, mode):
    base = Xtr[rng.integers(0, len(Xtr), n)].copy()
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = rng.uniform(TAU_RARE, 60000.0, n)
    base[:, 0] = SLOT[0] + rad * np.cos(ang)
    base[:, 1] = SLOT[1] + rad * np.sin(ang)
    base[:, 2] = SLOT[2] + rng.normal(0, 1000.0, n)
    if mode == "full":  # every non-error feature uniform over the training log's range
        lo, hi = Xtr.min(0), Xtr.max(0)
        for j in range(3, Xtr.shape[1]):
            base[:, j] = rng.uniform(lo[j], hi[j], n)
    return base


def rollout(act, dfp, offsets):
    """Vectorized closed-loop rollout in the kinematic simulator from the held-out
    pilot's first frame, one trajectory per (lateral, longitudinal) body offset."""
    n = len(offsets)
    r0 = dfp.iloc[0]
    psi0 = np.radians(r0.leader_heading)
    off = np.array(offsets)
    dE = off[:, 0] * np.cos(psi0) + off[:, 1] * np.sin(psi0)
    dN = -off[:, 0] * np.sin(psi0) + off[:, 1] * np.cos(psi0)
    lat = r0.latitude_deg + np.degrees(dN / R_FEET)
    lon = r0.longitude_deg + np.degrees(dE / (R_FEET * np.cos(np.radians(r0.latitude_deg))))
    alt = np.full(n, r0.altitude_ft)
    roll, pitch = np.full(n, r0.roll_deg), np.full(n, r0.pitch_deg)
    hdg, spd = np.full(n, r0.heading_deg), np.full(n, r0.airspeed_kt)
    steps = min(int(HORIZON_S / DT), len(dfp))
    L = dfp[["leader_lat", "leader_lon", "leader_alt_ft", "leader_heading"]].values
    errs = np.zeros((n, steps))
    for k in range(steps):
        llat, llon, lalt, lhdg = L[k]
        dlon, dlat = np.radians(lon - llon), np.radians(lat - llat)
        dx = R_FEET * np.cos(np.radians(lat)) * dlon
        dy = R_FEET * dlat
        dz = alt - lalt
        psi = np.radians(lhdg)
        xb, yb = dx * np.cos(psi) - dy * np.sin(psi), dx * np.sin(psi) + dy * np.cos(psi)
        vs = spd * 1.68781 * np.sin(np.radians(pitch))
        nz = -32.174 / np.maximum(np.cos(np.radians(roll)), 0.1)
        X = np.stack([xb, yb, dz, roll, pitch, hdg, spd, vs, nz], axis=1)
        errs[:, k] = slot_error(X)
        u = act(X)
        ail, ele = np.clip(u[:, 0], -1, 1), np.clip(u[:, 1], -1, 1)
        thr = np.clip(u[:, 3], 0, 1)
        roll = np.clip(roll + 26.10 * ail * DT, -75.0, 75.0) * 0.95
        pitch = np.clip(pitch + -4.14 * ele * DT, -20.0, 20.0) * 0.95
        spd = np.clip(spd + 1.02 * (thr - 0.67) * DT, 250.0, 750.0)
        V = spd * 1.68781
        hdg = (hdg + 1843.0 * np.tan(np.radians(roll)) / np.maximum(V, 50.0) * DT + 180) % 360 - 180
        Vh, Vz = V * np.cos(np.radians(pitch)), V * np.sin(np.radians(pitch))
        alt = alt + Vz * DT
        lat = lat + np.degrees(Vh * np.cos(np.radians(hdg)) * DT / R_FEET)
        lon = lon + np.degrees(Vh * np.sin(np.radians(hdg)) * DT / (R_FEET * np.cos(np.radians(lat))))
    return errs


def run_job(args):
    held_out, seed = args
    tag = f"fold{held_out}_s{seed}"
    paths = {k: os.path.join(OUT, k, tag + ".csv") for k in ("rollout", "mse")}
    if all(os.path.exists(p) for p in paths.values()):
        return tag, 0.0
    t0 = time.time()
    df = prepare(pd.read_csv(DATA))
    tr, va = df[df.pilot_id != held_out], df[df.pilot_id == held_out]
    X, Y, ep = tr[INPUT_COLS].values, tr[OUTPUT_COLS].values, tr.pilot_id.values
    err = tr.err.values
    ones = np.ones(len(X))
    n_aug = int(len(X) * AUG_FRACTION)
    gains = fit_gains(tr)
    lab = labeler(gains)

    def aug(mode):
        rng = np.random.default_rng(mseed("aug", mode, held_out, seed))
        Xa = aug_states(X, n_aug, rng, mode)
        return np.concatenate([X, Xa]), np.concatenate([Y, lab(Xa)])

    pol = {}
    for m in METHODS:
        s = mseed("flight", m, held_out, seed)
        if m == "BC":
            pol[m] = train_bc(X, Y, ones, s)
        elif m == "BC_pen":
            pol[m] = train_bc(X, Y, ones, s, LAMBDA_PEN)
        elif m == "GAFIL_hard":
            pol[m] = train_bc(X, Y, weights(err, ep, "hard"), s)
        elif m == "GAFIL_smooth":
            pol[m] = train_bc(X, Y, weights(err, ep, "smooth"), s)
        elif m == "GAFIL_smooth_pen":
            pol[m] = train_bc(X, Y, weights(err, ep, "smooth"), s, LAMBDA_PEN)
        elif m == "Oversample":
            w = ones.copy()
            rare = err > TAU_RARE
            frac = AUG_FRACTION / (1 + AUG_FRACTION)
            if rare.any():
                w[rare] = max(frac * (~rare).sum() / ((1 - frac) * rare.sum()), 1.0)
            pol[m] = train_bc(X, Y, w, s)
        elif m in ("IQL", "TD3BC"):
            pol[m] = train_offline_rl(X, Y, ep, s, m)
        elif m in ("Aug", "Aug_partial"):
            Xa, Ya = aug("full" if m == "Aug" else "partial")
            pol[m] = train_bc(Xa, Ya, np.ones(len(Xa)), s)
        elif m == "GAFIL_smooth_Aug":
            Xa, Ya = aug("full")
            pol[m] = train_bc(Xa, Ya, np.concatenate([weights(err, ep, "smooth"), np.ones(n_aug)]), s)
        elif m == "Labeler":
            pol[m] = lab
        elif m == "BC_Fallback":
            bc = pol["BC"]

            def fb(Z, bc=bc):
                Z = np.atleast_2d(Z)
                out = bc(Z)
                far = slot_error(Z) > TAU_RARE
                out[far] = lab(Z[far])
                return out
            pol[m] = fb

    Xv, Yv = va[INPUT_COLS].values, va[OUTPUT_COLS].values
    mse_rows, ro_rows = [], []
    for m, act in pol.items():
        if m not in ("Labeler", "BC_Fallback"):
            mse_rows.append(dict(method=m, held_out=held_out, seed=seed, one_step_mse=float(np.mean((act(Xv) - Yv) ** 2))))
        e = rollout(act, va, OFFSETS)
        for i, (ox, oy) in enumerate(OFFSETS):
            above = np.where(e[i] > TAU_RARE)[0]
            ro_rows.append(dict(method=m, held_out=held_out, seed=seed, off_lat=ox, off_lon=oy,
                                e30=e[i, 299], e60=e[i, 599], e120=e[i, -1], e_mean=float(e[i].mean()),
                                frac_within=float((e[i] < TAU_RARE).mean()),
                                t_loss=float(above[0] * DT) if len(above) else HORIZON_S))
    for k, rows in (("rollout", ro_rows), ("mse", mse_rows)):
        os.makedirs(os.path.dirname(paths[k]), exist_ok=True)
        pd.DataFrame(rows).to_csv(paths[k], index=False)
    return tag, time.time() - t0


def human_reference():
    df = prepare(pd.read_csv(DATA))
    rows = []
    for pid, g in df.groupby("pilot_id"):
        e = g.err.values
        above = np.where(e[: int(HORIZON_S / DT)] > TAU_RARE)[0]
        rows.append(dict(pilot=pid, e30=e[299], e60=e[599], e120=e[1199], e_mean=e[:1200].mean(),
                         frac_within=(e[:1200] < TAU_RARE).mean(), n_tail=int((e > TAU_RARE).sum()),
                         t_loss=above[0] * DT if len(above) else HORIZON_S))
    mech = []
    for row in mechanism(df.err.values, df.pilot_id.values):
        mech.append(row)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "human_reference.csv"), index=False)
    pd.DataFrame(mech).to_csv(os.path.join(OUT, "mechanism.csv"), index=False)
    print(pd.DataFrame(rows).round(0).to_string(index=False))
    print(pd.DataFrame(mech).round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--human", action="store_true")
    a = ap.parse_args()
    if a.human:
        human_reference()
        return
    if a.smoke:
        global OUT
        OUT = os.path.join(OUT, "_smoke")
        print(run_job((1, 0)))
        return
    jobs = [(p, s) for p in range(1, 6) for s in SEEDS]
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (tag, dt) in enumerate(pool.imap_unordered(run_job, jobs), 1):
            print(f"[{i}/{len(jobs)}] {tag} {dt:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
