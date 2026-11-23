"""bcbench: score a new way of combining a cloned policy with an engineered controller under the paper's protocol.

A method is a function that receives a Task and returns a policy. The Task holds the domain (simulator, labeler,
nominal set), the demonstrations of one seed, and helpers to train networks with the paper's budget. The policy is
a callable mapping a batch of observations to a batch of actions in [-1, 1] (or a run.Stateful / run.Predictive
object, for methods that keep state or need the simulator). Every method is scored on the same probes, nominal starts
and demonstrator reference as the paper's methods, so its numbers sit next to the released per-seed results.

    import bcbench
    def my_method(task):
        bc = task.bc()                                   # the paper's BC for this seed (same data, steps, seed)
        return lambda o: bc(o)                           # ...or anything else
    df = bcbench.evaluate(my_method, split="dev")        # settings x seeds, one row each
    print(bcbench.compare(df))                           # next to the paper's methods, with paired tests

Splits: "dev" (the 8 settings of the main results), "stress" (9 settings with a labeler unlike the demonstrator)
and "test" (the 12 settings held out to test the frozen rule; use them once, at the end). Metrics per seed:
recovery (off-nominal probes that end in the nominal set), retention (nominal starts well inside the set that never
leave it), distance (RMS distance to the demonstrator from common nominal starts, so leaving the set is penalized)
and fidelity gain 1 - distance / labeler's distance.

    python bcbench.py --method mymodule:my_method --split dev --workers 8
    python bcbench.py --method bcbench:example_support_switch --split dev --seeds 0-2     # a self-check
"""
import argparse
import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import domains as Dm  # noqa: E402
import run as R  # noqa: E402

SPLITS = {
    "dev": [("quad3d", 3), ("quad3d", 5), ("quad3d", 10), ("quad3d", 30), ("reacher", 3), ("reacher", 10),
            ("reacher", 30), ("pendulum", 30)],
    "stress": [("quad3d_hg4", 3), ("quad3d_hg4", 5), ("quad3d_hg4", 10), ("quad3d_hg16", 3), ("quad3d_hg16", 5),
               ("quad3d_hg16", 10), ("reacher_hg4", 3), ("reacher_hg4", 10), ("reacher_hg4", 30)],
    "test": [("pendulum", 10), ("pendulum_hg4", 10), ("pendulum_hg4", 30), ("pendulum_q10", 10), ("pendulum_q10", 30),
             ("quad3d_q05", 3), ("quad3d_q05", 5), ("quad3d_q05", 10), ("quad3d_hg4", 30), ("reacher_hg2", 3),
             ("reacher_hg2", 10), ("reacher_hg2", 30)],
}
SEEDS = range(15)
REFERENCE = ["BC", "Switch", "Switch-support", "Simplex", "Filter", "Backup", "Residual-box", "Blend-box", "DAgger-L",
             "Labeler"]
NAMES = {"Switch": "error switch", "Switch-support": "support switch", "Filter": "look-ahead switch",
         "Backup": "backup filter", "Residual-box": "residual blend", "Blend-box": "box blend"}


class Task:
    """One (setting, budget, seed) unit. Attributes: D (domain object: obs, step, labeler, obs_error, box_states,
    r_nominal, horizon, ...), X / O / A (demonstration states, observations, actions), seed, setting, budget."""

    def __init__(self, setting, budget, seed):
        self.setting, self.budget, self.seed = setting, budget, seed
        self.D = Dm.DOMAINS[setting]()
        self.D.demo_eps = budget
        self.X, self.O, self.A = R.demos(self.D, seed)
        self._bc = None

    def rng(self, tag):
        """A random generator private to this unit and tag."""
        return np.random.default_rng(R.mseed(tag, self.setting, self.seed))

    def train(self, O, A, tag, **kw):
        """The paper's MLP regression with the paper's gradient-step budget, inputs normalized on the demonstrations."""
        return R.train(O, A, self.O, R.mseed(self.D.name, self.seed, tag), **kw)

    def bc(self):
        """The paper's BC policy for this unit (same data, seed and steps as in the released results)."""
        if self._bc is None:
            self._bc = R.train(self.O, self.A, self.O, R.mseed(self.D.name, self.seed, "BC"))
        return self._bc

    def synthetic_states(self, n, tag="aug"):
        """States drawn over the domain's physical range outside the nominal set (the box states of the paper)."""
        return self.D.box_states(n, self.rng(tag), self.D.r_nominal)


def score(task, policy):
    """Recovery, retention and distance on the paper's probes and nominal starts for this unit."""
    D, seed, dname = task.D, task.seed, task.D.name
    prng = np.random.default_rng(R.mseed("probe", dname, seed))
    probes = {r: D.probe(r, R.N_PROBE, prng) for r in D.radii}
    x_nom = D.nominal_starts(R.N_PROBE, np.random.default_rng(R.mseed("nom", dname, seed)))
    ref_err, _, ref_traj, ref_dv = R.rollout(D, D.demonstrator, x_nom)
    ref_kept = (ref_err < D.r_nominal).all(0) & ~ref_dv
    fid = R.fidelity(D, policy, x_nom, ref_traj, ref_kept)
    succ = [R.probe_metrics(D, *[v for i, v in enumerate(R.rollout(D, policy, x0)) if i != 2])["success"].mean()
            for r, x0 in probes.items() if r > 0]
    lab = R.fidelity(D, D.labeler, x_nom, ref_traj, ref_kept)["fid_common"]
    return dict(setting=task.setting, budget=task.budget, seed=seed, recovery=float(np.mean(succ)),
                retention=fid["kept_inner"], distance=fid["fid_common"], fidelity_gain=1 - fid["fid_common"] / lab)


def _job(args):
    spec, setting, budget, seed = args
    mod, fn = spec.split(":")
    method = getattr(importlib.import_module(mod), fn)
    task = Task(setting, budget, seed)
    return score(task, method(task))


def evaluate(method, split="dev", seeds=SEEDS, workers=1):
    """Score `method` (a function Task -> policy, or a 'module:function' string for parallel runs) on a split."""
    import pandas as pd
    units = [(s, b, k) for s, b in SPLITS[split] for k in seeds]
    if workers > 1:
        assert isinstance(method, str), "pass 'module:function' to run in parallel"
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_job, [(method,) + u for u in units]))
    else:
        if isinstance(method, str):
            mod, fn = method.split(":")
            method = getattr(importlib.import_module(mod), fn)
        rows = []
        for s, b, k in units:
            t = Task(s, b, k)
            rows.append(score(t, method(t)))
    return pd.DataFrame(rows)


def reference(settings, seeds=SEEDS):
    """The paper's methods on the same units, from the released per-seed results (outputs/v3)."""
    import pandas as pd
    rows = []
    for s, b in settings:
        for k in seeds:
            p = os.path.join(R.OUT, s, f"eps{b}", f"seed{k}.csv")
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p)
            d = d[d.method.isin(REFERENCE)]
            rec = d[d.radius > 0].groupby("method").success.mean()
            one = d.groupby("method")[["kept_inner", "fid_common"]].first()
            lab = one.loc["Labeler", "fid_common"]
            for m in one.index:
                rows.append(dict(setting=s, budget=b, seed=k, method=NAMES.get(m, m), recovery=rec[m],
                                 retention=one.loc[m, "kept_inner"], distance=one.loc[m, "fid_common"],
                                 fidelity_gain=1 - one.loc[m, "fid_common"] / lab))
    return pd.DataFrame(rows)


def compare(df, name="new method"):
    """Means per setting for the new method and the paper's methods, and paired Wilcoxon tests over all units of
    the new method against each (two-sided; positive difference = new method better)."""
    import pandas as pd
    from scipy.stats import wilcoxon
    settings = sorted(set(zip(df.setting, df.budget)))
    ref = reference(settings, sorted(df.seed.unique()))
    allm = pd.concat([ref, df.assign(method=name)])
    table = allm.groupby(["setting", "budget", "method"])[["recovery", "retention", "distance", "fidelity_gain"]] \
        .mean().round(3)
    tests = []
    key = ["setting", "budget", "seed"]
    for m in ref.method.unique():
        j = df.merge(ref[ref.method == m], on=key, suffixes=("", "_ref"))
        row = dict(versus=m, units=len(j))
        for c, sign in (("recovery", 1), ("retention", 1), ("distance", -1)):
            diff = sign * (j[c] - j[c + "_ref"]).dropna()
            row[c + "_diff"] = round(float(diff.mean()), 4)
            row[c + "_p"] = float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0
        tests.append(row)
    return table, pd.DataFrame(tests)


def example_support_switch(task):
    """Self-check: the paper's support switch rebuilt through the Task interface; it should match the released
    'support switch' rows up to floating point."""
    return R.support_switch(task.D, task.bc(), task.O, task.X, task.D.r_nominal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="module:function returning a policy for a Task")
    ap.add_argument("--split", default="dev", choices=list(SPLITS))
    ap.add_argument("--seeds", default="0-14")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None, help="CSV for the per-seed scores")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-")) if "-" in a.seeds else (int(a.seeds), int(a.seeds))
    df = evaluate(a.method, a.split, range(lo, hi + 1), a.workers)
    if a.out:
        df.to_csv(a.out, index=False)
    table, tests = compare(df, a.method.split(":")[1])
    import pandas as pd
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table)
        print(tests)


if __name__ == "__main__":
    main()
