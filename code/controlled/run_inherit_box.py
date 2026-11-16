"""P1 without interaction: the same labeler variants as run_inherit.py, but the policy is
cloned non-interactively, (i) as a box blend (demonstrations + variant labels on box states)
and (ii) as a residual blend (variant + correction fitted to demonstrator - variant on the
demonstrations and to zero on box states). Same probes and common label-error states.
    python run_inherit_box.py --domain quad3d --seeds 0-9 --workers 8
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import domains as Dm  # noqa: E402
import run as R  # noqa: E402
import run_inherit as I  # noqa: E402


def run_seed(args):
    dname, seed = args
    import pandas as pd
    D = Dm.DOMAINS[dname]()
    out = os.path.join(R.OUT, "inherit_box", dname, f"seed{seed}.csv")
    if os.path.exists(out):
        return out
    X, O, A = R.demos(D, seed)
    n_aug = int(R.AUG_FRAC * len(O))
    prng = np.random.default_rng(R.mseed("probe", dname, seed))
    probes = {r: D.probe(r, R.N_PROBE, prng) for r in D.radii if r > 0}
    S_common = D.box_states(n_aug, np.random.default_rng(R.mseed("common", dname, seed)), D.r_nominal)
    rows = []
    for v in I.VARIANTS:
        lab = I.variant(D, v, O, A)
        rng = np.random.default_rng(R.mseed("inhbox", dname, seed, v))
        So = D.obs(D.box_states(n_aug, rng, D.r_nominal))
        box = R.train(np.concatenate([O, So]), np.concatenate([A, lab(So)]), O, R.mseed(dname, seed, "ibox", v))
        rnet = R.train(np.concatenate([O, So]), np.concatenate([A - lab(O), np.zeros((len(So), D.nu))]), O,
                       R.mseed(dname, seed, "ires", v))
        res = (lambda o, lab=lab, rnet=rnet: np.clip(lab(o) + rnet(o), -1, 1))
        Sc = D.obs(S_common)
        bias = float(((lab(Sc) - D.labeler(Sc)) ** 2).mean())

        def glued(o, lab=lab):
            out = D.demonstrator(o)
            far = D.obs_error(o) >= D.r_nominal
            out[far] = lab(o[far])
            return out
        for r, x0 in probes.items():
            row = dict(domain=dname, seed=seed, variant=v, radius=r, label_bias=bias)
            for name, pol in [("labeler", lab), ("glued", glued), ("box", box), ("resid", res)]:
                e, _, _, dv = R.rollout(D, pol, x0)
                row[f"{name}_final"] = float(np.mean(e[-1]))
                row[f"{name}_success"] = float(np.mean((e[-1] < D.success) & ~dv))
            rows.append(row)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="quad3d")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    with ProcessPoolExecutor(a.workers) as ex:
        for p in ex.map(run_seed, [(a.domain, s) for s in range(lo, hi + 1)]):
            print("done", p, flush=True)


if __name__ == "__main__":
    main()
