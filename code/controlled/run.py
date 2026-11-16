"""v3 controlled experiments: blending a cloned demonstrator with an engineered labeler.

Per (domain, seed):
  demonstrations: demonstrator rollouts from nominal starts (Markovian, own style)
  methods (all MLPs share architecture, optimizer, epochs; inputs z-scored on demos):
    BC          demos only
    Blend-box   demos + labeler labels on full-state box samples (error >= r_in)
    Blend-part  demos + labels on far positions with other coordinates copied from demos
    Blend-tube  demos + labels along the labeler's own rollouts from box starts (error >= r_in)
    DAgger-L    Blend-box, then 3 rounds: roll the policy out from fresh box starts, label the
                visited states with error >= r_in by the labeler, aggregate, retrain
    DART        demos collected with Gaussian action noise injected (demonstrator labels)
    Switch      BC if error < r_in else labeler
    SoftGate    (1-w) BC + w labeler, w ramps from 0 at 0.75 r_in to 1 at 1.25 r_in
    Simplex     full-state runtime assurance with hysteresis (domain-specific envelope)
    Labeler, Demonstrator   references
  r_in sweep for Blend-box (robustness/fidelity frontier).
Metrics per probe rollout: final error, success (final error < success), time to
enter N (error < r_nominal), fraction of post-entry time inside N, peak |du|.
Closed-loop fidelity from nominal starts: RMS distance of the policy's state
trajectory to the demonstrator's from the same start, and on-policy action MSE
to the demonstrator.

    python run.py --domain quad3d --seeds 0-14 --workers 12
"""
import argparse
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import domains as Dm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "outputs", "v3")
AUG_FRAC = 0.3
N_PROBE = 24
R_IN_SWEEP = [0.5, 1.0, 2.0, 3.0]   # multiples of r_nominal


def mseed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31)


# ---------------------------------------------------------------------------
def demos(D, seed, noise=0.0):
    rng = np.random.default_rng(mseed("demo", D.name, seed))
    x = D.nominal_starts(D.demo_eps, rng)
    X, O, A = [], [], []
    for _ in range(D.demo_T):
        o = D.obs(x)
        a = D.demonstrator(o)
        X.append(x); O.append(o); A.append(a)
        ex = a + (rng.normal(0, noise, a.shape) if noise > 0 else 0.0)
        x = D.process_noise(D.step(x, np.clip(ex, -1, 1)), rng)
    return np.concatenate(X), np.concatenate(O), np.concatenate(A)


STEPS = 6000   # every policy gets the same number of gradient steps, whatever its data size


def train(O, A, Onorm, seed, hidden=128, steps=None, batch=128):
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    steps = STEPS if steps is None else steps
    m, s = Onorm.mean(0), Onorm.std(0)
    s[s < 1e-6] = 1.0
    net = nn.Sequential(nn.Linear(O.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, A.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.tensor((O - m) / s, dtype=torch.float32)
    Yt = torch.tensor(A, dtype=torch.float32)
    done = 0
    while done < steps:
        idx = torch.tensor(rng.permutation(len(Xt)))
        for b in range(0, len(Xt), batch):
            j = idx[b:b + batch]
            loss = ((net(Xt[j]) - Yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            done += 1
            if done >= steps:
                break
    net.eval()

    def act(o):
        with torch.no_grad():
            return net(torch.tensor((o - m) / s, dtype=torch.float32)).numpy()
    return act


def tube_states(D, n, rng, r_in, n_starts=256):
    x = D.box_states(n_starts, rng, r_in)
    keep = []
    for _ in range(D.horizon):
        x = D.step(x, D.labeler(D.obs(x)))
        keep.append(x[D.error(x) >= r_in])
    T = np.concatenate(keep)
    return T[rng.integers(0, len(T), n)]


# ---------------------------------------------------------------------------
class Stateful:
    """Policy wrapper with per-environment memory (Simplex)."""

    def __init__(self, fn):
        self.fn = fn

    def reset(self, n):
        self.mode = np.zeros(n, dtype=bool)


def make_simplex(D, bc, r_in):
    pol = Stateful(None)

    def unsafe(o):
        return D.envelope(o, r_in)

    def fn(o):
        bad, good = unsafe(o)
        pol.mode = np.where(pol.mode, ~good, bad)
        out = bc(o)
        out[pol.mode] = D.labeler(o[pol.mode])
        return out
    pol.fn = fn
    return pol


class Predictive:
    """Predictive safety filter: at every step, simulate the cloned policy H steps ahead from the
    current (full) state with the true dynamics; apply it only if the prediction stays in the
    nominal set and inside the domain's envelope, otherwise apply the labeler."""

    def __init__(self, D, bc, r_in, horizon_s=0.5):
        self.D, self.bc, self.r_in = D, bc, r_in
        self.H = max(1, int(round(horizon_s / D.dt)))

    def act(self, o, x):
        D = self.D
        ok = np.ones(len(x), dtype=bool)
        xp = x.copy()
        for _ in range(self.H):
            op = D.obs(xp)
            bad, _ = D.envelope(op, self.r_in)
            ok &= ~bad
            xp = D.step(xp, np.clip(self.bc(op), -1, 1))
        bad, _ = D.envelope(D.obs(xp), self.r_in)
        ok &= ~bad
        out = self.bc(o)
        out[~ok] = D.labeler(o[~ok])
        return out



class Cascade(Predictive):
    """Look-ahead cascade: apply the first candidate policy (cloned policy, then residual blend) whose
    H-step prediction with the true dynamics stays in the nominal set and the envelope; otherwise the
    labeler. With the cloned policy alone it is the look-ahead switch."""

    def __init__(self, D, cands, r_in, horizon_s=0.5):
        super().__init__(D, cands[0], r_in, horizon_s)
        self.cands = cands

    def _ok(self, pol, x):
        D = self.D
        ok = np.ones(len(x), dtype=bool)
        xp = x.copy()
        for _ in range(self.H):
            op = D.obs(xp)
            ok &= ~D.envelope(op, self.r_in)[0]
            xp = D.step(xp, np.clip(pol(op), -1, 1))
        return ok & ~D.envelope(D.obs(xp), self.r_in)[0]

    def act(self, o, x):
        out = self.D.labeler(o)
        todo = np.ones(len(x), dtype=bool)
        for pol in self.cands:
            if not todo.any():
                break
            idx = np.where(todo)[0]
            ok = self._ok(pol, x[idx])
            out[idx[ok]] = pol(o[idx[ok]]) if ok.any() else out[idx[ok]]
            todo[idx[ok]] = False
        return out



def learn_model(D, X, A, rng, seed, n_starts=256):
    """One-step dynamics model x' = x + f(x, u), fitted on logged transitions only: the demonstrations
    (with the demonstrator's actions) and the labeler's rollouts from box starts, both with small action noise.
    Uses the simulator only to generate the log, as one would log a real system."""
    xs, us = [X], [A]
    x = D.box_states(n_starts, rng, D.r_nominal)
    for _ in range(D.horizon):
        u = np.clip(D.labeler(D.obs(x)) + rng.normal(0, 0.2, (len(x), D.nu)), -1, 1)
        xs.append(x); us.append(u)
        x = D.step(x, u)
    Xs, Us = np.concatenate(xs), np.concatenate(us)
    Us = np.clip(Us + rng.normal(0, 0.1, Us.shape), -1, 1)
    # in chunks: the MuJoCo domains keep one simulator state per batch row
    Xn = np.concatenate([D.step(Xs[i:i + 512], Us[i:i + 512]) for i in range(0, len(Xs), 512)])
    dX = Xn - Xs
    Z = np.concatenate([Xs, Us], 1)
    net = train(Z, dX / (dX.std(0) + 1e-9), Z, mseed("model", D.name, seed))
    sc = dX.std(0) + 1e-9

    def model(x, u):
        return x + net(np.concatenate([x, np.clip(u, -1, 1)], 1)) * sc
    return model


class PredictiveLearned(Predictive):
    """The look-ahead switch with a learned one-step model instead of the true dynamics."""

    def __init__(self, D, bc, r_in, model, horizon_s=0.5):
        super().__init__(D, bc, r_in, horizon_s)
        self.model = model

    def act(self, o, x):
        D = self.D
        ok = np.ones(len(x), dtype=bool)
        xp = x.copy()
        for _ in range(self.H):
            op = D.obs(xp)
            bad, _ = D.envelope(op, self.r_in)
            ok &= ~bad
            xp = self.model(xp, np.clip(self.bc(op), -1, 1))
        bad, _ = D.envelope(D.obs(xp), self.r_in)
        ok &= ~bad
        out = self.bc(o)
        out[~ok] = D.labeler(o[~ok])
        return out


class BackupFilter(Predictive):
    """Backup-trajectory safety filter (gatekeeper / backup-CBF style): a candidate action is
    accepted only if, after applying it for one step, the labeler (the backup controller) keeps
    the predicted state inside the nominal set and the envelope for the next H steps. Candidates
    move from the cloned policy's action toward the labeler's, (1-l) bc + l kappa for l in GRID;
    the first certified one is applied (minimal modification along that segment), and the
    labeler otherwise. Uses the true dynamics, like Predictive."""
    GRID = (0.0, 0.25, 0.5, 0.75)

    def __init__(self, D, bc, r_in, horizon_s=1.0):
        super().__init__(D, bc, r_in, horizon_s)

    def _certified(self, x, a):
        D = self.D
        ok = np.ones(len(x), dtype=bool)
        xp = D.step(x, np.clip(a, -1, 1))
        for _ in range(self.H):
            op = D.obs(xp)
            bad, _ = D.envelope(op, self.r_in)
            ok &= ~bad
            xp = D.step(xp, np.clip(D.labeler(op), -1, 1))
        return ok & ~D.envelope(D.obs(xp), self.r_in)[0]

    def act(self, o, x):
        D = self.D
        ab, ak = self.bc(o), D.labeler(o)
        out = ak.copy()
        todo = np.ones(len(x), dtype=bool)
        for lam in self.GRID:
            if not todo.any():
                break
            idx = np.where(todo)[0]
            a = (1 - lam) * ab[idx] + lam * ak[idx]
            ok = self._certified(x[idx], a)
            out[idx[ok]] = a[ok]
            todo[idx[ok]] = False
        return out


def support_switch(D, bc, O, X, r_in, q=95.0):
    """Switch with a monitor on the demonstrations' support in the full observation: the labeler
    acts when the task error is >= r_in OR the observation is farther from every demonstration
    frame (z-scored, nearest neighbour) than the q-th percentile of the demonstrations' own
    leave-one-episode-out nearest-neighbour distances."""
    from scipy.spatial import cKDTree
    m, s = O.mean(0), O.std(0)
    s[s < 1e-6] = 1.0
    Z = (O - m) / s
    ep = np.tile(np.arange(D.demo_eps), D.demo_T)        # demos() stacks time steps of all episodes
    tree = cKDTree(Z)
    d_self = np.empty(len(Z))
    for e in np.unique(ep):
        idx = ep == e
        t = cKDTree(Z[~idx]) if (~idx).any() else tree
        d_self[idx] = t.query(Z[idx])[0]
    tau = np.percentile(d_self, q)

    def f(o):
        out = bc(o)
        far = (D.obs_error(o) >= r_in) | (tree.query((o - m) / s)[0] > tau)
        out[far] = D.labeler(o[far])
        return out
    return f


def rollout(D, pol, x0):
    x = x0.copy()
    if isinstance(pol, Stateful):
        pol.reset(len(x))
        f = pol.fn
    elif isinstance(pol, Predictive):
        f = None
    else:
        f = pol
    n = len(x)
    err = np.zeros((D.horizon, n))
    U = np.zeros((D.horizon, n, D.nu))
    traj = np.zeros((D.horizon, n, D.nx))
    dv = np.zeros(n, dtype=bool)
    for k in range(D.horizon):
        o = D.obs(x)
        u = np.clip(pol.act(o, x) if f is None else f(o), -1, 1)
        err[k] = D.error(x)
        U[k] = u
        traj[k] = x
        x = D.step(x, u)
        dv |= D.diverged(x)
    return err, U, traj, dv


def probe_metrics(D, err, U, dv):
    rn = D.r_nominal
    T = len(err)
    inside = err < rn
    first = np.where(inside.any(0), inside.argmax(0), -1)
    stay = np.array([inside[f:, i].mean() if f >= 0 else 0.0 for i, f in enumerate(first)])
    du = np.linalg.norm(np.diff(U, axis=0), axis=2)
    return dict(final=err[-1], success=(err[-1] < D.success) & ~dv, t_enter=np.where(first >= 0, first * D.dt, np.nan),
                stay=stay, time_in=inside.mean(0), du_peak=du.max(0), diverged=dv)


def fidelity(D, pol, x0, ref_traj, ref_kept=None):
    """Closed-loop fidelity from nominal starts: RMS distance of the policy's trajectory to the
    demonstrator's (all rollouts, and only rollouts that never leave the nominal set), and the
    on-policy action MSE to the demonstrator."""
    err, U, traj, dv = rollout(D, pol, x0)
    dpos = D.fid_distance(traj, ref_traj)                 # (T, n)
    kept = (err < D.r_nominal).all(0) & ~dv
    o = D.obs(traj.reshape(-1, D.nx))
    amse = ((U.reshape(-1, D.nu) - D.demonstrator(o)) ** 2).mean()
    inner = D.error(x0) < 0.8 * D.r_nominal              # starts strictly inside N
    common = ref_kept if ref_kept is not None else np.ones(len(x0), dtype=bool)
    return dict(fid_common=float(np.sqrt((dpos[:, common] ** 2).mean())) if common.any() else np.nan,
                kept_inner=float(kept[inner].mean()) if inner.any() else np.nan,
                fid_state=float(np.sqrt((dpos ** 2).mean())),
                fid_kept=float(np.sqrt((dpos[:, kept] ** 2).mean())) if kept.any() else np.nan,
                kept_frac=float(kept.mean()), fid_action=float(amse),
                nominal_left=float((err[len(err) // 3:] > D.r_nominal).any(0).mean()))


# ---------------------------------------------------------------------------
def run_seed(args):
    dname, seed, eps = args
    import pandas as pd
    D = Dm.DOMAINS[dname]()
    if eps:
        D.demo_eps = eps
    path = os.path.join(OUT, dname, f"eps{D.demo_eps}", f"seed{seed}.csv")
    if os.path.exists(path) and not os.environ.get("ONLY"):
        return path
    X, O, A = demos(D, seed)
    n_aug = int(AUG_FRAC * D.demo_T * 30)  # synthetic states are free: fixed budget, 30% of the full 30-episode set
    r0 = D.r_nominal
    rng = np.random.default_rng(mseed("aug", dname, seed))
    pols = {}
    bc = train(O, A, O, mseed(dname, seed, "BC"))
    pols["BC"] = bc

    def blend(S, tag):
        So = D.obs(S)
        return train(np.concatenate([O, So]), np.concatenate([A, D.labeler(So)]), O, mseed(dname, seed, tag))

    for k in R_IN_SWEEP:
        pols[f"Blend-box@{k}"] = blend(D.box_states(n_aug, rng, k * r0), f"box{k}")
    pols["Blend-box"] = pols["Blend-box@1.0"]
    pols["Blend-part"] = blend(D.box_partial(n_aug, rng, r0, X), "part")
    pols["Blend-tube"] = blend(tube_states(D, n_aug, rng, r0), "tube")
    # DAgger with the labeler as expert on off-nominal states the learner visits
    S_agg = D.box_states(n_aug, rng, r0)
    pi = pols["Blend-box"]
    for it in range(3):
        x = D.box_states(64, rng, r0)
        vis = []
        for _ in range(D.horizon):
            x = D.step(x, np.clip(pi(D.obs(x)), -1, 1))
            vis.append(x[D.error(x) >= r0])
        V = np.concatenate(vis)
        S_agg = np.concatenate([S_agg, V[rng.integers(0, len(V), n_aug // 3)]])
        pi = blend(S_agg, f"dagger{it}")
    pols["DAgger-L"] = pi
    # DART: same demonstration budget, action noise injected during collection
    _, Od, Ad = demos(D, seed, noise=0.3)
    pols["DART"] = train(Od, Ad, Od, mseed(dname, seed, "DART"))
    dart = pols["DART"]

    def switch(o):
        out = bc(o)
        far = D.obs_error(o) >= r0
        out[far] = D.labeler(o[far])
        return out

    def softgate(o):
        w = np.clip((D.obs_error(o) - 0.75 * r0) / (0.5 * r0), 0, 1)[:, None]
        return (1 - w) * bc(o) + w * D.labeler(o)

    pols["Switch"] = switch
    pols["SoftGate"] = softgate
    pols["Simplex"] = make_simplex(D, bc, r0)
    pols["Filter"] = Predictive(D, bc, r0)
    pols["Backup"] = BackupFilter(D, bc, r0)

    def switch_dart(o):
        out = dart(o)
        far = D.obs_error(o) >= r0
        out[far] = D.labeler(o[far])
        return out
    pols["Switch-DART"] = switch_dart
    pols["Switch-support"] = support_switch(D, bc, O, X, r0)
    res = A - D.labeler(O)
    Sres = D.obs(D.box_states(n_aug, rng, r0))
    rnet = train(np.concatenate([O, Sres]), np.concatenate([res, np.zeros((len(Sres), D.nu))]), O,
                 mseed(dname, seed, "Residual-box"))
    rdemo = train(O, res, O, mseed(dname, seed, "Residual"))
    pols["Residual-box"] = lambda o: np.clip(D.labeler(o) + rnet(o), -1, 1)
    pols["Residual"] = lambda o: np.clip(D.labeler(o) + rdemo(o), -1, 1)
    pols["Cascade"] = Cascade(D, [bc, pols["Residual-box"]], r0)
    if os.environ.get("ONLY") and "Filter-learned" in os.environ.get("ONLY"):
        model = learn_model(D, X, A, np.random.default_rng(mseed("model-data", dname, seed)), seed)
        pols["Filter-learned"] = PredictiveLearned(D, bc, r0, model)
    pols["Labeler"] = D.labeler
    pols["Demonstrator"] = D.demonstrator
    only = os.environ.get("ONLY")
    if only:   # evaluate only these methods (plus the references), into a separate result tree
        keep = set(only.split(",")) | {"Labeler", "Demonstrator"}
        pols = {m: p for m, p in pols.items() if m in keep}
        tag = ("dec_" if os.environ.get("DECISION") else "only_") + only.replace(",", "_")
        path = os.path.join(OUT, tag, dname, f"eps{D.demo_eps}", f"seed{seed}.csv")
        if os.path.exists(path):
            return path

    prng = np.random.default_rng(mseed("probe", dname, seed))
    probes = {r: D.probe(r, N_PROBE, prng) for r in D.radii}
    # DECISION=1: a fresh, independent set of nominal starts (for deciding the rule apart from scoring)
    x_nom = D.nominal_starts(N_PROBE, np.random.default_rng(mseed("dec" if os.environ.get("DECISION") else "nom", dname, seed)))
    ref_err, _, ref_traj, ref_dv = rollout(D, D.demonstrator, x_nom)
    ref_kept = (ref_err < D.r_nominal).all(0) & ~ref_dv
    rows = []
    for m, p in pols.items():
        fid = fidelity(D, p, x_nom, ref_traj, ref_kept)
        if os.environ.get("DECISION"):     # decision starts only: no probes
            rows.append(dict(domain=dname, seed=seed, demo_eps=D.demo_eps, method=m, **fid))
            continue
        for r, x0 in probes.items():
            err, U, _, dv = rollout(D, p, x0)
            met = probe_metrics(D, err, U, dv)
            rows.append(dict(domain=dname, seed=seed, demo_eps=D.demo_eps, method=m, radius=r,
                             **{k: float(np.nanmean(v)) if k != "t_enter" else float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
                                for k, v in met.items()},
                             final_median=float(np.median(met["final"])), **fid))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="quad3d")
    ap.add_argument("--seeds", default="0-14")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--eps", default="", help="comma-separated demonstration budgets (episodes)")
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    budgets = [int(e) for e in a.eps.split(",")] if a.eps else [0]
    jobs = [(a.domain, s, e) for e in budgets for s in range(lo, hi + 1)]
    with ProcessPoolExecutor(a.workers) as ex:
        for p in ex.map(run_seed, jobs):
            print("done", p, flush=True)


if __name__ == "__main__":
    main()
