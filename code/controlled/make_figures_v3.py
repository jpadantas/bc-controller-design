"""Figures for the v3 paper.

fig_budget.pdf  -- quadrotor, demonstration budget sweep: recovery, nominal retention,
                   closed-loop distance to the demonstrator; pendulum and quadrotor gap sweep.
    python make_figures_v3.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402


def logx_plain(ax, axis="x"):
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402

FIG = os.path.join(A.ROOT, "paper", "figures")
INK, MUTED, GRID = "#1f1f1e", "#b9b8b2", "#e6e5e0"
# one color per method, used in every figure of the paper
STYLE = {
    "BC": ("BC", "#eb6834", "-", "o"),
    "Switch": ("Error switch", "#1baf7a", "-", "s"),
    "SoftGate": ("Soft gate", "#e87ba4", "-", "D"),
    "Simplex": ("Simplex", "#6250d6", "-", "v"),
    "Switch-support": ("Support switch", "#1baf7a", "--", "s"),
    "Filter": ("Look-ahead switch", "#9c5b2e", "--", "h"),
    "Backup": ("Backup filter", "#9c5b2e", "-", "h"),
    "Blend-box": ("Blend (box)", "#86b6ef", "-", "^"),
    "DAgger-L": ("Blend (DAgger-L)", "#c2185b", "-.", "P"),
    "Residual-box": ("Residual blend", "#eda100", "-", "X"),
    "Labeler": ("Labeler", "#52514e", "--", None),
    "Demonstrator": ("Demonstrator", "#52514e", ":", None),
}
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6,
                     "axes.edgecolor": "#6b6a64", "xtick.color": "#3d3d3a", "ytick.color": "#3d3d3a"})


def style(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def ci(x):
    from scipy import stats
    x = np.asarray(x, float)
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def fig_budget():
    doms = [("quad3d", "Quadrotor", 1.0, "m"), ("reacher", "Reacher", 100.0, "cm")]
    fig, axes = plt.subplots(2, 3, figsize=(6.75, 3.9))
    for row, (dn, dname, fs, unit) in enumerate(doms):
        q = A.per_seed(A.load(dn))
        budgets = sorted(q.demo_eps.unique())
        panels = [("recover", "recovered (fraction)"), ("kept_inner", "never left $\mathcal{N}$ (fraction)"),
                  ("fid_common", f"distance ({unit})")]
        for col, (k, yl) in enumerate(panels):
            ax = axes[row, col]
            style(ax)
            for m in ["BC", "Switch", "Switch-support", "Simplex", "Backup", "Blend-box", "DAgger-L", "Residual-box"]:
                name, c, ls, mk = STYLE[m]
                mu, hw = zip(*[ci(q[(q.method == m) & (q.demo_eps == b)][k].dropna() * (fs if k == "fid_common" else 1.0))
                               for b in budgets])
                mu, hw = np.array(mu), np.array(hw)
                ax.fill_between(budgets, mu - hw, mu + hw, color=c, alpha=0.05 if m == "DAgger-L" else 0.12, lw=0)
                ax.plot(budgets, mu, color=c, ls=ls, marker=mk, ms=3.2, lw=1.2, label=name)
            if k == "kept_inner":
                lo = 0.9 if dn == "quad3d" else None
                ax.set_ylim(bottom=lo, top=1.01)
            lab = q[q.method == "Labeler"][k].mean() * (fs if k == "fid_common" else 1.0)
            ax.axhline(lab, color=STYLE["Labeler"][1], ls=":", lw=0.9)
            if k == "fid_common":
                ax.text(budgets[-1], lab * 0.86, "labeler alone", color="#52514e", fontsize=6, ha="right", va="top")
                ax.set_yscale("log")
                logx_plain(ax, "y")
            ax.set_xscale("log")
            ax.set_xticks(budgets)
            ax.set_xticklabels([str(b) for b in budgets])
            ax.minorticks_off()
            ax.set_ylabel(yl, color="#3d3d3a")
            if row == 1:
                ax.set_xlabel("demonstration episodes", color="#3d3d3a")
            ax.set_title(f"{dname}: " + ["recovery", "retention", "distance to demonstrator"][col],
                         fontsize=7.5, color=INK, pad=3)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.05), columnspacing=1.0)
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    fig.savefig(os.path.join(FIG, "fig_budget.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig_budget.png"), dpi=300, bbox_inches="tight")

def fig_inherit():
    import glob
    import pandas as pd
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.75))
    doms = [("quad3d", "Quadrotor", "#eb6834", "s"), ("reacher", "Reacher", "#1baf7a", "^"),
            ("pendulum", "Pendulum", "#2a78d6", "o")]
    for dn, name, col, mk in doms:
        d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(A.OUT, "inherit", dn, "seed*.csv"))])
        g = d.groupby(["variant", "radius"]).mean(numeric_only=True).reset_index()
        axes[0].scatter(g.glued_success, g.policy_success, s=10, color=col, marker=mk, edgecolor="white",
                        linewidth=0.4, label=name, zorder=3)
        axes[1].scatter(np.maximum(g.label_bias, 1e-3), g.policy_success, s=10, color=col, marker=mk, edgecolor="white",
                        linewidth=0.4, zorder=3)
    for ax in axes:
        style(ax)
        ax.set_ylim(-0.05, 1.05)
    axes[0].plot([0, 1], [0, 1], color=MUTED, lw=0.8, zorder=1)
    axes[0].set_xlabel("glued reference: recovered", color="#3d3d3a")
    axes[0].set_ylabel("cloned policy: recovered", color="#3d3d3a")
    axes[0].set_title("(a) Closed loop", fontsize=7.5, color=INK, pad=3)
    axes[1].set_xscale("log")
    axes[1].set_xticks([1e-3, 1e-2, 1e-1, 1e0])
    axes[1].set_xticklabels(["0", "$10^{-2}$", "$10^{-1}$", "$10^{0}$"])
    axes[1].minorticks_off()
    axes[1].set_xlabel("label MSE to correct labeler", color="#3d3d3a")
    axes[1].set_title("(b) Label accuracy", fontsize=7.5, color=INK, pad=3)
    axes[1].set_yticklabels([])
    axes[0].legend(frameon=False, fontsize=6, loc="upper left", handletextpad=0.2)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(FIG, "fig_inherit.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig_inherit.png"), dpi=300, bbox_inches="tight")



def fig_frontier():
    """Recovery vs. closed-loop distance to the demonstrator as each method's knob varies
    (seeds 0-9): switching radius x{0.7,1,1.4} for Switch and Simplex, synthetic budget
    {0.1,0.3,1.0} of the 30-episode set for the residual and box blends."""
    import glob
    import pandas as pd
    fig, AX = plt.subplots(2, 4, figsize=(6.75, 3.4), sharex="col")
    cfg = [("quad3d", 3, "Quadrotor, 3 ep.", 1.0, "m"), ("quad3d", 30, "Quadrotor, 30 ep.", 1.0, "m"),
           ("reacher", 3, "Reacher, 3 ep.", 100.0, "cm"), ("reacher", 30, "Reacher, 30 ep.", 100.0, "cm")]
    for col, (dn, eps, title, sc, unit) in enumerate(cfg):
      for row, metric in enumerate(["rec", "ret"]):
        ax = AX[row, col]
        style(ax)
        main = A.load(dn)
        main = main[(main.demo_eps == eps) & (main.seed <= 9)]
        sens = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(A.OUT, "sens", dn, f"eps{eps}_seed*.csv"))])

        def point(d, m, metric=metric):
            g = d[d.method == m]
            y = g[g.radius > 0].success.mean() if metric == "rec" else g.groupby("seed").kept_inner.first().mean()
            return y, g.groupby("seed").fid_common.first().mean() * sc
        series = [("Switch", ["Switch@0.5", "Switch@0.7", "Switch@0.85", "Switch", "Switch@1.4"]),
                  ("Simplex", ["Simplex@0.5", "Simplex@0.7", "Simplex@0.85", "Simplex", "Simplex@1.4"]),
                  ("Blend-box", ["Blend-box@0.1", "Blend-box", "Blend-box@1.0"]),
                  ("Residual-box", ["Residual-box@0.1", "Residual-box", "Residual-box@1.0"])]
        for key, ms in series:
            pts = [point(sens if "@" in m else main, m) for m in ms]
            y, x = zip(*pts)
            name, c, ls, mk = STYLE[key]
            ax.plot(x, y, color=c, lw=1.1, marker=mk, ms=3.0, label=name)
            dflt = ms.index(key) if key in ms else 1
            ax.plot([x[dflt]], [y[dflt]], color=c, marker=mk, ms=6.5, markeredgecolor="black", markeredgewidth=0.6)
        for key in ["Backup", "DAgger-L", "Labeler", "BC"]:
            y, x = point(main, key)
            name, c, ls, mk = STYLE[key]
            ax.plot([x], [y], color=c, marker=mk or "*", ms=5 if mk else 7, ls="none", label=name)
        ax.set_xscale("log")
        logx_plain(ax)
        if row == 0:
            ax.set_title(title, fontsize=7.5, color=INK, pad=3)
        else:
            ax.set_xlabel(f"distance to demonstrator ({unit})", color="#3d3d3a", fontsize=6.5)
        ax.set_ylim(-0.03, 1.03)
    AX[0, 0].set_ylabel("recovered (fraction)", color="#3d3d3a")
    AX[1, 0].set_ylabel("never left $\mathcal{N}$ (fraction)", color="#3d3d3a")
    h, l = AX[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.06), columnspacing=1.0)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(FIG, "fig_frontier.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig_frontier.png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    fig_frontier()
    fig_budget()
    fig_inherit()
