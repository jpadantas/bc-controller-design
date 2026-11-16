"""An invariance check at the interface (Theorem 3.1, quadrotor).

V(x) = x' P x, with P solving the discrete Lyapunov equation of the labeler's closed loop
linearized at hover (A' P A - P = -I). The nominal level c0 is the 95th percentile of V
over the demonstrations, N = {V <= c0}. For each policy we start rollouts on the
boundary band of N (demonstration states with 0.8 c0 <= V <= c0) and measure the
fraction that leave N within 1 s (V > c0): an empirical violation rate of (b), which
the theorem says must be zero for N to be invariant. We also report each policy's
nominal retention from the main experiment for comparison.

    python theory_check.py --seeds 0-4
"""
import argparse
import os
import sys

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import domains as Dm  # noqa: E402
import run as R  # noqa: E402


def lyapunov(D):
    x0 = np.zeros((1, D.nx))
    h = 1e-4
    A = np.zeros((D.nx, D.nx))
    f0 = D.step(x0, D.labeler(D.obs(x0)))[0]
    for j in range(D.nx):
        x = x0.copy()
        x[0, j] += h
        A[:, j] = (D.step(x, D.labeler(D.obs(x)))[0] - f0) / h
    P = solve_discrete_lyapunov(A.T, np.eye(D.nx))
    rho = max(abs(np.linalg.eigvals(A)))
    return P, rho


def V(P, x):
    xx = x.copy()
    xx[:, 8] = Dm.wrap(xx[:, 8])
    return np.einsum("ni,ij,nj->n", xx, P, xx)


def exit_rate(D, pol, xb, P, c0, steps=50):
    x = xb.copy()
    if isinstance(pol, R.Stateful):
        pol.reset(len(x))
        f = pol.fn
    else:
        f = pol
    out = np.zeros(len(x), dtype=bool)
    for _ in range(steps):
        x = D.step(x, np.clip(f(D.obs(x)), -1, 1))
        out |= V(P, x) > c0
    return out.mean()


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--eps", default="3,5,10,30")
    a = ap.parse_args()
    D = Dm.Quad3D()
    P, rho = lyapunov(D)
    print(f"linearized labeler loop: spectral radius {rho:.4f} per step")
    rows = []
    lo, hi = map(int, a.seeds.split("-"))
    for eps in [int(e) for e in a.eps.split(",")]:
        D.demo_eps = eps
        for seed in range(lo, hi + 1):
            X, O, A = R.demos(D, seed)
            D30 = Dm.Quad3D()
            X30, _, _ = R.demos(D30, seed)
            v30 = V(P, X30)
            c0 = np.percentile(v30, 95)
            band = X30[(v30 >= 0.8 * c0) & (v30 <= c0)]
            band = band[np.random.default_rng(seed).integers(0, len(band), 256)]
            bc = R.train(O, A, O, R.mseed("quad3d", seed, "BC"))
            n_aug = int(R.AUG_FRAC * D.demo_T * 30)
            rng = np.random.default_rng(R.mseed("aug", "quad3d", seed))
            S = D.box_states(n_aug, rng, D.r_nominal)
            So = D.obs(S)
            blend = R.train(np.concatenate([O, So]), np.concatenate([A, D.labeler(So)]), O, R.mseed("quad3d", seed, "box1.0"))

            S_agg, pi = S.copy(), blend
            for it in range(3):
                x = D.box_states(64, rng, D.r_nominal)
                vis = []
                for _ in range(D.horizon):
                    x = D.step(x, np.clip(pi(D.obs(x)), -1, 1))
                    vis.append(x[D.error(x) >= D.r_nominal])
                Vv = np.concatenate(vis)
                S_agg = np.concatenate([S_agg, Vv[rng.integers(0, len(Vv), n_aug // 3)]])
                Sa = D.obs(S_agg)
                pi = R.train(np.concatenate([O, Sa]), np.concatenate([A, D.labeler(Sa)]), O,
                             R.mseed("quad3d", seed, f"dagger{it}"))
            dagger = pi

            def switch(o):
                out = bc(o)
                far = D.obs_error(o) >= D.r_nominal
                out[far] = D.labeler(o[far])
                return out
            res = A - D.labeler(O)
            rnet = R.train(np.concatenate([O, So]), np.concatenate([res, np.zeros((len(So), D.nu))]), O,
                           R.mseed("quad3d", seed, "Residual-box"))
            pols = {"BC": bc, "Switch": switch, "Simplex": R.make_simplex(D, bc, D.r_nominal),
                    "Switch-support": R.support_switch(D, bc, O, X, D.r_nominal),
                    "Residual-box": (lambda o, rnet=rnet: np.clip(D.labeler(o) + rnet(o), -1, 1)),
                    "Blend-box": blend, "DAgger-L": dagger, "Labeler": D.labeler, "Demonstrator": D.demonstrator}
            for m, p in pols.items():
                rows.append(dict(demo_eps=eps, seed=seed, method=m, exit=exit_rate(D, p, band, P, c0)))
            print(eps, seed, {r["method"]: round(r["exit"], 3) for r in rows[-len(pols):]}, flush=True)
    d = pd.DataFrame(rows)
    out = os.path.join(R.OUT, "theory_check_quad.csv")
    d.to_csv(out, index=False)
    print(d.groupby(["demo_eps", "method"]).exit.mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
