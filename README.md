# Design Considerations for Combining Behavior Cloning with Engineered Controllers

Code and per-seed results for an evaluation of how to combine a policy cloned from nominal human demonstrations
with an engineered controller: switches, monitors and safety filters, action mixtures, learned blends and
residual blends, compared under one protocol in three controlled domains, on the robomimic multi-human Lift
benchmark, and on a formation-flight case study with fighter pilots.

## Layout

```
code/controlled/   controlled domains (quadrotor, MuJoCo Reacher, pendulum): methods, runs, statistics, tables, figures
code/robo/         robomimic Lift with six human operators (separate environment, see below)
code/*.py          FlightGear formation-flight pipeline (needs FlightGear and the pilots' data; see below)
outputs/v3/        per-seed results of the controlled domains (one CSV per domain, budget and seed)
outputs/robo/      per-seed results of the robomimic study
notes/             predictions written down before each run, with their dated outcomes
paper/tables/      generated LaTeX tables (created by the scripts)
paper/figures/     generated figures (created by the scripts)
```

## Environments

* Controlled domains and flight: Python 3.12, `pip install -r requirements.txt`.
* robomimic study: a separate environment, `pip install -r requirements-robo.txt` (robosuite 1.4.1 with
  mujoco 3.2.7). On Windows, robosuite 1.4.1 needs two local fixes: remove the `ctypes.WinDLL(...mujoco.dll)` line
  in `robosuite/utils/binding_utils.py`, and add `robosuite/macros_private.py` with `MUJOCO_GPU_RENDERING = False`;
  MuJoCo also needs ASCII-only paths. Run with `MUJOCO_GL=disable`.

## Reproducing the paper

Tables and figures from the released per-seed results (no training needed):

```
cd code/controlled
python make_tables_v3.py        # Table 1 (main), smoothness and appendix tables
python make_figures_v3.py       # Figure 1 (budget), P1 and frontier figures
python make_mismatch_table.py   # Table 2 and the mismatched-labeler appendix tables
python make_median_table.py     # medians and interquartile ranges of distance (appendix)
python stats_v3.py              # every in-text statistic, the pairwise table, the rule evaluations
cd ../robo
python lift_table.py            # Table 4 (six human operators); between-operator reference from lift_between.py
python lift_rule.py             # the frozen rule on Lift
```

Rerunning the experiments (15 seeds each; CPU only):

```
cd code/controlled
python run.py --domain quad3d   --eps 3,5,10,30 --seeds 0-14
python run.py --domain reacher  --eps 3,10,30   --seeds 0-14
python run.py --domain pendulum --eps 30        --seeds 0-14
python run.py --domain quad3d_hg4 --eps 3,5,10,30 --seeds 0-14      # mismatched labelers: quad3d_hg16, reacher_hg4,
                                                                     # reacher_hg16; new settings: pendulum (10),
                                                                     # pendulum_hg4, pendulum_q10, quad3d_q05, reacher_hg2
DECISION=1 ONLY=BC,Residual-box python run.py --domain ... # fresh decision starts for the selection rule
ONLY=Filter-learned,Filter python run.py --domain ...       # look-ahead switch with a learned model
python run_inherit.py --domain quad3d --seeds 0-9           # labeler variants (P1), also run_inherit_box.py
python run_sens.py --domain quad3d --eps 3,30 --seeds 0-9   # switch radii and a larger blend network
python theory_check.py --seeds 0-14                         # condition (b) in the quadrotor
python fitcheck.py ... ; python cvselect.py ...             # open-loop fit check; cross-validated selection

cd ../robo                                                  # in the robosuite environment
ROBO_DATA=<path>/lift_mh_low_dim_v141.hdf5 MUJOCO_GL=disable LIFT_OUT=lift_big LIFT_STEPS=60000 LIFT_HIDDEN=1024 \
    python lift_study.py --workers 6
... same environment variables ... python lift_lookahead.py --workers 6   # look-ahead switch (simulator look-ahead)
... same environment variables ... python lift_chunk.py --workers 6       # action-chunking BC and its switches
ROBO_DATA=<path> python lift_between.py                                    # human reference distances
```

The robomimic Lift multi-human low-dimensional dataset is distributed by its authors
(robomimic v0.3, `lift/mh/low_dim_v141.hdf5`); it is not included here.

## Flight case study

`code/fg_*.py` drive FlightGear 2020.3.19 with the F-16CJ Block 52 aircraft through its generic UDP protocol
(`code/Protocol/*.xml`; copy them to a folder with an ASCII path and set `FG_EXTRA_DATA` to it, and `FG_HOME`
to the FlightGear installation). The pilots' recordings are not released, so these scripts document the protocol
but cannot be run without comparable data:

```
python fg_replay.py                                          # open-loop replay check
FG_SEEDS=0,1,2,3,4,5,6,7,8,9 python fg_experiment_v4.py --workers 6
python fg_extra_v4.py --workers 5                            # probe starts
FLIGHT_EXP=experiment_v4 FLIGHT_TAB=v4_flight.tex python fg_flight_table.py
```

The earlier flight run of the appendix is `fg_experiment_v3.py`, `fg_dagger_v3.py` and `fg_extra_v3.py`.

## License

MIT (see `LICENSE`).
