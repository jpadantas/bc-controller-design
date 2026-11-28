# Fidelity during recovery: protocol and predictions, written 2026-11-28 before the run

Purpose (reviewer request): in the quadrotor and Reacher the demonstrator itself recovers from the probes, so we can
ask whether a combination recovers the way the demonstrator does. Code: code/v3/run_offfid.py. Policies as in run.py;
for each off-nominal probe (radius > 0) the RMS distance between the method's trajectory and the demonstrator's from
the same start, on probes from which the demonstrator recovers. Settings: quad3d 3 and 10, reacher 3 and 10, seeds 0-9.

Predictions (means over seeds):
 O1. Among methods that recover from at least 90% of probes in a setting, none is closer to the demonstrator during
     recovery than the labeler alone by more than 10%, in any of the four settings.
 O2. The error switch, the support switch and Simplex are within 10% of the labeler alone in every setting.

Outcome (2026-11-28; quad3d 3 / quad3d 10 / reacher 3 / reacher 10; distance x100, m or m):
 Labeler alone: 153.9, 153.9, 2.86, 2.86. Closest method recovering >= 90%: residual blend 148.9, 144.2, 2.76, 2.63.
 O1 holds. No method that recovers from at least 90% is closer than the labeler by more than 10% (at most 8%,
    the residual blend in Reacher with 10 episodes).
 O2 fails in Reacher. Support switch within 3% everywhere (153.8, 153.9, 2.89, 2.78). Error switch 154.5, 146.6,
    3.57, 3.35 and Simplex 149.8, 146.6, 3.38, 3.12, i.e. 17-25% farther in Reacher, where they recover less
    (62-74% and 84-92%) because BC acts outside its support.
