"""P1 in the v3 domains: does the policy's off-nominal recovery follow its labeler's?

Deterministic labeler variants only (a stochastic labeler run as a policy injects
fresh noise each step, while the cloned policy regresses its mean):
  correct   the domain's labeler
  weak      labeler output scaled by 0.4
  rot30/60/90/120/150  labeler applied to the task error rotated by that angle
            (quad: horizontal position and velocity error; pendulum: the angle error is
            replaced by the angle error scaled by cos(angle) and the velocity term rotated
            into it, see rotated())
  linear    least-squares linear fit of the labeler on the demonstration states,
            extrapolated
  flip      sign of the labeler's output flipped
Policy: demonstrations (30 episodes) + DAgger-L with the variant as labeler.
References: the variant run alone, and the GLUED reference of the theorem: the
demonstrator inside the nominal set and the variant outside it.
For each (variant, radius, seed): labeler-as-policy final error and policy final
error, plus label bias = mean squared difference between the variant's and the
correct labeler's labels on the same synthetic states.

    python run_inherit.py --domain quad3d --seeds 0-9 --workers 12
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import domains as Dm  # noqa: E402
import run as R  # noqa: E402

VARIANTS = ["correct", "weak", "weak7", "rot30", "rot60", "rot90", "rot120", "rot150", "linear", "flip"]


def rot2(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.column_stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]])


def variant(D, name, O_demo, A_demo):
    base = D.labeler
    if name == "correct":
        return base
    if name == "weak":
        return lambda o: 0.4 * base(o)
    if name == "weak7":
        return lambda o: 0.7 * base(o)
    if name == "flip":
        return lambda o: -base(o)
    if name == "linear":
        Xa = np.column_stack([O_demo, np.ones(len(O_demo))])
        W, *_ = np.linalg.lstsq(Xa, base(O_demo), rcond=None)
        return lambda o: np.clip(np.column_stack([o, np.ones(len(o))]) @ W, -1, 1)
    ang = np.radians(float(name[3:]))
    if D.name == "quad3d":
        def f(o):
            o2 = o.copy()
            o2[:, 0:2] = rot2(o[:, 0:2], ang)
            o2[:, 3:5] = rot2(o[:, 3:5], ang)
            return base(o2)
        return f
    if D.name == "reacher":
        def h(o):
            o2 = o.copy()
            o2[:, 8:10] = rot2(o[:, 8:10], ang)
            return base(o2)
        return h

    def g(o):  # pendulum: rotate the (angle error, angular velocity) pair seen by the law
        th = np.arctan2(o[:, 1], o[:, 0])
        v = rot2(np.column_stack([th, o[:, 2] * 0.3]), ang)
        th2, w2 = v[:, 0], v[:, 1] / 0.3
        return base(np.column_stack([np.cos(th2), np.sin(th2), w2]))
    return g


def dagger(D, O, A, lab, seed, rng, n_aug, tag):
    r0 = D.r_nominal

    def fit(S):
        So = D.obs(S)
        return R.train(np.concatenate([O, So]), np.concatenate([A, lab(So)]), O, R.mseed(D.name, seed, tag, len(S)))
    S = D.box_states(n_aug, rng, r0)
    pi = fit(S)
    for _ in range(3):
        x = D.box_states(64, rng, r0)
        vis = []
        for _ in range(D.horizon):
            x = D.step(x, np.clip(pi(D.obs(x)), -1, 1))
            vis.append(x[D.error(x) >= r0])
        V = np.concatenate(vis)
        S = np.concatenate([S, V[rng.integers(0, len(V), n_aug // 3)]])
        pi = fit(S)
    return pi, S


def run_seed(args):
    dname, seed = args
    import pandas as pd
    D = Dm.DOMAINS[dname]()
    out = os.path.join(R.OUT, "inherit", dname, f"seed{seed}.csv")
    if os.path.exists(out):
        return out
    X, O, A = R.demos(D, seed)
    n_aug = int(R.AUG_FRAC * len(O))
    prng = np.random.default_rng(R.mseed("probe", dname, seed))
    probes = {r: D.probe(r, R.N_PROBE, prng) for r in D.radii if r > 0}
    S_common = D.box_states(n_aug, np.random.default_rng(R.mseed("common", dname, seed)), D.r_nominal)
    rows = []
    for v in VARIANTS:
        lab = variant(D, v, O, A)
        rng = np.random.default_rng(R.mseed("inh", dname, seed, v))
        pi, S = dagger(D, O, A, lab, seed, rng, n_aug, v)
        So = D.obs(S_common)                       # the same states for every variant
        bias = float(((lab(So) - D.labeler(So)) ** 2).mean())
        def glued(o, lab=lab):
            out = D.demonstrator(o)
            far = D.obs_error(o) >= D.r_nominal
            out[far] = lab(o[far])
            return out
        for r, x0 in probes.items():
            el, _, _, dvl = R.rollout(D, lab, x0)
            ep, _, _, dvp = R.rollout(D, pi, x0)
            eg, _, _, dvg = R.rollout(D, glued, x0)
            rows.append(dict(domain=dname, seed=seed, variant=v, radius=r, label_bias=bias,
                             labeler_final=float(np.mean(el[-1])), policy_final=float(np.mean(ep[-1])),
                             labeler_success=float(np.mean((el[-1] < D.success) & ~dvl)),
                             policy_success=float(np.mean((ep[-1] < D.success) & ~dvp)),
                             glued_final=float(np.mean(eg[-1])),
                             glued_success=float(np.mean((eg[-1] < D.success) & ~dvg))))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="quad3d")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    with ProcessPoolExecutor(a.workers) as ex:
        for p in ex.map(run_seed, [(a.domain, s) for s in range(lo, hi + 1)]):
            print("done", p, flush=True)


if __name__ == "__main__":
    main()
