# A controller fitted to the demonstrations: protocol and predictions, written 2026-11-28 before the run

Purpose (reviewer request): the engineered labeler is given for free. What if a practitioner instead fits a simple
controller to the demonstrations? Code: code/v3/run_fitted.py. In each seed the labeler is replaced by an affine
state-feedback law u = clip(W o + b) fitted by ridge regression (lambda = 1e-3) to that seed's demonstrations, and
the combinations are rebuilt around it exactly as in run.py (same demonstrations, synthetic states, seeds, steps,
probes and metrics): BC, error switch, support switch, residual blend, box blend, DAgger-L, and the fitted law alone.
Settings: quad3d 3 and 10 episodes, reacher 3 and 10, pendulum 30; seeds 0-9. The engineered-labeler numbers are the
main run's on the same seeds. Action error is measured on states of an independent demonstration set (seed + 1000).

Predictions (mean over seeds):
 F1. The fitted law's action error against the demonstrator is lower than the engineered labeler's in each of the
     five settings.
 F2. The fitted law alone recovers from fewer off-nominal probes than the engineered labeler in each setting.
 F3. Averaged over the five combinations (error switch, support switch, residual blend, box blend, DAgger-L), recovery
     with the fitted law as labeler is lower than with the engineered labeler in at least four of five settings.
 F4. The fitted law alone is closer to the demonstrator from nominal starts (fid_common) than the engineered labeler
     in at least four of five settings.

Outcome (2026-11-28; quad3d 3 / quad3d 10 / reacher 3 / reacher 10 / pendulum 30, means over ten seeds):
 F1 holds. Action error, fitted vs engineered: 0.031 vs 0.716, 0.024 vs 0.717, 0.036 vs 0.100, 0.013 vs 0.101,
    0.044 vs 0.219.
 F2 holds. Recovery of the fitted law alone: 4, 24, 2, 1, 0% (engineered 100, 100, 95, 95, 100%).
 F3 holds (5/5). Mean recovery of the five combinations, fitted vs engineered: 5 vs 100, 25 vs 100, 3 vs 80,
    4 vs 85, 3 vs 100%.
 F4 fails (1/5). Distance from nominal starts, fitted law alone vs engineered (x100): 106.9 vs 62.5, 18.7 vs 62.5,
    8.5 vs 1.1, 1.5 vs 1.1, 207.9 vs 8.2. Its closed loop leaves the nominal set from many nominal starts
    (retention 64, 98, 13, 81, 0%), so it loses even on fidelity.
