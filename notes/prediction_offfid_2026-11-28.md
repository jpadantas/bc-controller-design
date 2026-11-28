# Fidelity during recovery: protocol and predictions, written 2026-11-28 before the run

Purpose (reviewer request): in the quadrotor and Reacher the demonstrator itself recovers from the probes, so we can
ask whether a combination recovers the way the demonstrator does. Code: code/v3/run_offfid.py. Policies as in run.py;
for each off-nominal probe (radius > 0) the RMS distance between the method's trajectory and the demonstrator's from
the same start, on probes from which the demonstrator recovers. Settings: quad3d 3 and 10, reacher 3 and 10, seeds 0-9.

Predictions (means over seeds):
 O1. Among methods that recover from at least 90% of probes in a setting, none is closer to the demonstrator during
     recovery than the labeler alone by more than 10%, in any of the four settings.
 O2. The error switch, the support switch and Simplex are within 10% of the labeler alone in every setting.

Outcome: (to be filled in after the run)
