"""Stronger policy classes on Lift, Can and Square (reviewer request: is "BC fails inside the support" an artifact of a
unimodal MLP?).

Two policy classes, trained on the same leave-one-operator-out data, with the same 3x1024 trunk and gradient steps as BC:
  GMM        an MLP with a five-component Gaussian-mixture head (robomimic's BC-GMM); at run time a component is drawn
             from the mixture weights and its mean is applied
  Diffusion  a DDPM over chunks of 16 actions conditioned on the observation (Diffusion Policy, state-based, MLP
             denoiser, 100 training noise levels, 10 deterministic DDIM steps at run time); the first 8 actions of a
             chunk are applied before planning again
Each is run alone, behind the support switch (as in the main studies) and behind an uncertainty switch that needs no
extra training: the labeler acts when the policy's own predictive spread exceeds its 95th percentile on the training
frames (GMM: total variance of the mixture; Diffusion: variance of the first action over four samples).
Starts and metrics are those of ens_study.py (Lift, Can) and square_study.py (Square).

    MUJOCO_GL=disable ROBO_DATA=<lift> ROBO_CAN=<can> ROBO_SQUARE=<square> LIFT_STEPS=60000 LIFT_HIDDEN=1024 \
        LIFT_THREADS=2 python strong_study.py --workers 12 --seeds 0,1,2
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import can_common as C  # noqa: E402
import can_study as CS  # noqa: E402
import fid_study as F  # noqa: E402
import lift_common as L  # noqa: E402
import lift_study as S  # noqa: E402
import square_common as Q  # noqa: E402

OUT = os.path.join(S.ROOT, "outputs", "robo", os.environ.get("STRONG_OUT", "strong"))
NETS = OUT + "_nets"
N_EVAL = int(os.environ.get("STRONG_NEVAL", "20"))
QE = 95
NMIX, H, EXE, TN, TI, NS = 5, 16, 8, 100, 10, 4
CLASSES = ["GMM", "Diffusion"]
GATES = ["alone", "Support", "Uncertainty"]
TASKS = ["lift", "can", "square"]


def load(task):
    return {"lift": L.load, "can": C.load, "square": Q.load}[task]()


def data(task, held):
    d, kw = load(task)
    tr = [o for o in L.OPERATORS if o != held]
    demos = [x for o in tr for x in d[o]]
    O = np.concatenate([x["obs"] for x in demos])
    A = np.concatenate([x["actions"] for x in demos])
    opid = np.concatenate([np.full(len(x["obs"]), i) for i, o in enumerate(tr) for x in d[o]])
    idx = [np.minimum(np.arange(len(x["actions"]))[:, None] + np.arange(H)[None, :], len(x["actions"]) - 1)
           for x in demos]
    Ach = np.concatenate([x["actions"][i].reshape(len(i), -1) for x, i in zip(demos, idx)]).astype(np.float32)
    return d, kw, O, A, Ach, opid


def trunk(nin, nout, hidden):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(nin, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, nout))


def temb(t, dim=32):
    import torch
    f = torch.exp(-np.log(1000.0) * torch.arange(dim // 2, dtype=torch.float32, device=t.device) / (dim // 2))
    a = t.float()[:, None] * f[None, :]
    return torch.cat([a.sin(), a.cos()], 1)


def alphas_bar():
    s = 0.008
    t = np.arange(TN + 1) / TN
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    return np.clip(f[1:] / f[0], 1e-5, 1.0).astype(np.float32)


def fit(cls, O, A, Ach, seed, cache):
    import torch
    torch.set_num_threads(int(os.environ.get("LIFT_THREADS", "2")))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    hidden, steps, batch = int(os.environ.get("LIFT_HIDDEN", "1024")), S.STEPS, 256
    m, s = O.mean(0), O.std(0) + 1e-6
    dev = os.environ.get("STRONG_DEVICE", "cpu")
    Xt = torch.tensor((O - m) / s, dtype=torch.float32, device=dev)
    if cls == "GMM":
        net = trunk(O.shape[1], NMIX * (1 + 2 * 7), hidden)
        Yt = torch.tensor(A, dtype=torch.float32, device=dev)
    else:
        net = trunk(O.shape[1] + H * 7 + 32, H * 7, hidden)
        Yt = torch.tensor(Ach, dtype=torch.float32, device=dev)
        ab = torch.tensor(alphas_bar(), device=dev)
    net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4 if cls == "Diffusion" else 1e-3)

    def gmm(out):
        lg = out[:, :NMIX]
        mu = out[:, NMIX:NMIX * 8].reshape(-1, NMIX, 7)
        ls = out[:, NMIX * 8:].reshape(-1, NMIX, 7).clamp(-5, 2)
        return lg, mu, ls

    if os.path.exists(cache):
        net.load_state_dict(torch.load(cache, map_location=dev))
    else:
        done = 0
        while done < steps:
            perm = torch.tensor(rng.permutation(len(Xt)), device=dev)
            for b in range(0, len(Xt), batch):
                j = perm[b:b + batch]
                if cls == "GMM":
                    lg, mu, ls = gmm(net(Xt[j]))
                    y = Yt[j][:, None, :]
                    lp = (-0.5 * ((y - mu) / ls.exp()) ** 2 - ls - 0.5 * np.log(2 * np.pi)).sum(-1)
                    loss = -torch.logsumexp(torch.log_softmax(lg, 1) + lp, 1).mean()
                else:
                    t = torch.randint(0, TN, (len(j),), device=dev)
                    eps = torch.randn(len(j), H * 7, device=dev)
                    a = ab[t][:, None]
                    xn = a.sqrt() * Yt[j] + (1 - a).sqrt() * eps
                    loss = ((net(torch.cat([Xt[j], xn, temb(t)], 1)) - eps) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                done += 1
                if done >= steps:
                    break
        tmp = cache + f".{os.getpid()}.tmp"
        torch.save({k: v.cpu() for k, v in net.state_dict().items()}, tmp)
        os.replace(tmp, cache)
    net.to("cpu")
    net.eval()
    return Policy(cls, net, m, s, gmm, seed)


class Policy:
    """Stateful policy (chunk buffer, sampling rng) with an uncertainty score; reset() at each rollout."""

    def __init__(self, cls, net, m, s, gmm, seed):
        self.cls, self.net, self.m, self.s, self.gmm, self.seed = cls, net, m, s, gmm, seed
        self.ab = alphas_bar()
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.buf = []

    def invalidate(self):
        self.buf = []

    def _z(self, x):
        import torch
        return torch.tensor((np.atleast_2d(x) - self.m) / self.s, dtype=torch.float32)

    def _mix(self, x):
        import torch
        with torch.no_grad():
            lg, mu, ls = self.gmm(self.net(self._z(x)))
        return torch.softmax(lg, 1).numpy(), mu.numpy(), ls.exp().numpy()

    def _sample(self, x, n):
        """n chunks by deterministic DDIM from seeded noise, for each row of x (rows x n x H x 7)."""
        import torch
        z = self._z(x)
        B = len(z)
        zz = z.repeat_interleave(n, 0)
        xt = torch.tensor(self.rng.standard_normal((B * n, H * 7)), dtype=torch.float32)
        ts = np.linspace(TN - 1, 0, TI).round().astype(int)
        with torch.no_grad():
            for k, t in enumerate(ts):
                a = float(self.ab[t])
                ap = float(self.ab[ts[k + 1]]) if k + 1 < len(ts) else 1.0
                eps = self.net(torch.cat([zz, xt, temb(torch.full((B * n,), int(t)))], 1))
                x0 = ((xt - (1 - a) ** 0.5 * eps) / a ** 0.5).clamp(-1.5, 1.5)
                xt = ap ** 0.5 * x0 + (1 - ap) ** 0.5 * eps
        return xt.numpy().reshape(B, n, H, 7)

    def uncertainty(self, x):
        """Predictive spread (rows of x): total variance of the mixture, or of the first action over NS samples."""
        if self.cls == "GMM":
            p, mu, sd = self._mix(x)
            mean = (p[:, :, None] * mu).sum(1)
            second = (p[:, :, None] * (sd ** 2 + mu ** 2)).sum(1)
            return (second - mean ** 2).sum(1)
        return self._sample(x, NS)[:, :, 0, :].var(1).sum(1)

    def __call__(self, x):
        if self.cls == "GMM":
            p, mu, _ = self._mix(x)
            k = self.rng.choice(NMIX, p=p[0] / p[0].sum())
            return mu[0, k]
        if not self.buf:
            self.buf = list(self._sample(x, 1)[0, 0, :EXE])
        return self.buf.pop(0)


class Gated:
    """The labeler acts outside the gate; the policy's chunk buffer is dropped whenever the labeler acts."""

    def __init__(self, pol, lab, gate):
        self.pol, self.lab, self.gate = pol, lab, gate

    def reset(self):
        self.pol.reset()

    def __call__(self, x):
        if self.gate(x):
            return self.pol(x)
        self.pol.invalidate()
        return self.lab(x)


def train_job(args):
    task, held, seed, cls = args
    os.makedirs(NETS, exist_ok=True)
    _, _, O, A, Ach, _ = data(task, held)
    fit(cls, O, A, Ach, S.mseed("strong", task, held, seed, cls), os.path.join(NETS, f"{task}_p{held}_s{seed}_{cls}.pt"))
    return args


def eval_job(args):
    import pandas as pd
    from scipy.spatial import cKDTree
    task, held, seed, cls = args
    path = os.path.join(OUT + "_parts", f"{task}_p{held}_s{seed}_{cls}.csv")
    if os.path.exists(path):
        return path
    d, kw, O, A, Ach, opid = data(task, held)
    pol = fit(cls, O, A, Ach, S.mseed("strong", task, held, seed, cls),
              os.path.join(NETS, f"{task}_p{held}_s{seed}_{cls}.pt"))
    lab = {"lift": L.labeler, "can": C.labeler, "square": Q.labeler}[task]
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

    def in_ws(x):
        return bool(np.all(x[0:3] >= lo_e) and np.all(x[0:3] <= hi_e) and np.all(x[cdim] >= lo_c)
                    and np.all(x[cdim] <= hi_c))

    sub = O[np.random.default_rng(0).choice(len(O), min(2000, len(O)), replace=False)]
    uthr = float(np.percentile(np.concatenate([pol.uncertainty(sub[i:i + 250]) for i in range(0, len(sub), 250)]), QE))
    pol.reset()
    gates = {"Support": lambda x: in_ws(x) and tree.query((x - mu) / sd)[0] <= thr,
             "Uncertainty": lambda x: in_ws(x) and float(pol.uncertainty(x)[0]) <= uthr}
    if task == "lift":
        env, roll = L.make_env(kw), L.rollout
        erng = np.random.default_rng(S.mseed("lift-ens-eval", held))
        nom = []
        for _ in range(N_EVAL):
            env.reset()
            nom.append(env.sim.get_state().flatten().copy())
        off = [st for _, st in S.offnominal_starts(env, N_EVAL, erng)]
        fid_h = lambda x: max(L.HORIZON, len(x["obs"]) + 20)  # noqa: E731
    elif task == "can":
        env, roll = C.make_env(kw), C.rollout
        erng = np.random.default_rng(S.mseed("can-ens-eval", held))
        nom = [CS.start(env, erng, 0.0, False)[1] for _ in range(N_EVAL)]
        off = [CS.start(env, erng, 0.4, True)[1] for _ in range(N_EVAL)]
        fid_h = lambda x: min(CS.FID_MAX, max(C.HORIZON, len(x["obs"]) + 20))  # noqa: E731
    else:
        import square_study as SQ
        env, roll = Q.make_env(kw), Q.rollout
        erng = np.random.default_rng(S.mseed("square-eval", held))
        nom = [SQ.start(env, erng, 0.0, False)[1] for _ in range(N_EVAL)]
        off = [SQ.start(env, erng, 0.4, True)[1] for _ in range(N_EVAL)]
        fid_h = lambda x: min(SQ.FID_MAX, max(Q.HORIZON, len(x["obs"]) + 20))  # noqa: E731
    rows = []
    for g in GATES:
        run = pol if g == "alone" else Gated(pol, lab, gates[g])
        frac = (lambda X: np.nan) if g == "alone" else (lambda X: float(np.mean([gates[g](x) for x in X])))
        base = dict(task=task, held_out=held, seed=seed, cls=cls, gate=g)
        for kind, starts in (("nom", nom), ("off", off)):
            for j, st in enumerate(starts):
                X, U, s = roll(env, run, state=st)
                rows.append(dict(base, kind=kind, start=j, success=float(s), kept=float(all(in_ws(x) for x in X)),
                                 bc_frac=frac(X)))
        for j, x in enumerate(d[held][::2]):
            X, U, s = roll(env, run, state=x["states"][0], horizon=fid_h(x))
            Pp, R = X[:, 0:3], x["obs"][:, 0:3]
            T = max(len(Pp), len(R))
            Pa = np.concatenate([Pp, np.repeat(Pp[-1:], T - len(Pp), 0)])
            Ra = np.concatenate([R, np.repeat(R[-1:], T - len(R), 0)])
            rows.append(dict(base, kind="fid", start=j, success=float(s),
                             rms=float(np.sqrt(np.mean(np.sum((Pa - Ra) ** 2, 1)))), dtw=F.dtw(Pp, R), bc_frac=frac(X)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = pd.DataFrame(rows)
    r.to_csv(path, index=False)
    print(task, held, seed, cls, r[r.kind == "nom"].groupby("gate").success.mean().round(2).to_dict(), flush=True)
    return path


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    ap.add_argument("--train-only", action="store_true", help="only train (e.g. STRONG_DEVICE=cuda, --reverse)")
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    jobs = [(t, op, s, c) for t in a.tasks.split(",") for s in seeds for op in L.OPERATORS for c in CLASSES]
    if a.smoke:
        jobs = jobs[:2]
    with ProcessPoolExecutor(a.workers) as ex:
        for j in ex.map(train_job, jobs[::-1] if a.reverse else jobs):
            print("trained", j, flush=True)
    if a.train_only:
        return
    with ProcessPoolExecutor(a.workers) as ex:
        for p in ex.map(eval_job, jobs):
            print("done", p, flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.concat([pd.read_csv(os.path.join(OUT + "_parts", f"{t}_p{op}_s{s}_{c}.csv")) for t, op, s, c in jobs]) \
        .to_csv(os.path.join(OUT, "strong_rollouts.csv"), index=False)


if __name__ == "__main__":
    main()
