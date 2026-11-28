# The ensemble switch in the controlled domains, scored with bcbench: predictions, written 2026-11-28 before the run

Purpose (reviewer request): the ensemble switch, the gate that fixed Can and Square, was only run on the arm tasks.
Code: code/v3/bench_ensemble.py (support switch plus a monitor on the disagreement of five cloned networks, 95th
percentile on training frames), scored with code/v3/bcbench.py on the dev split (8 settings, seeds 0-14), against the
released support-switch results on the same units.

Predictions (means over 15 seeds per setting):
 K1. Recovery is within 2 points of the support switch's, or higher, in every setting.
 K2. Retention is at least the support switch's in at least six of eight settings.
 K3. Distance to the demonstrator from common starts is larger than the support switch's in at least six of eight
     settings (the labeler acts more often near nominal).

Outcome (2026-11-28; quad3d 3/5/10/30, reacher 3/10/30, pendulum 30; %, distance x100):
 K1 holds (8/8). Recovery equals the support switch's in every setting (100, 100, 100, 100, 95.7, 95.7, 95.7, 100).
 K2 holds (8/8). Retention 100 in every setting (support switch 98.6, 100, 100, 100, 97.6, 96.1, 98.8, 100).
 K3 holds (8/8). Distance 62.7, 62.7, 62.7, 62.5, 1.08, 1.08, 1.08, 2.9 vs support switch 48.9, 46.4, 41.6, 43.1,
    1.07, 0.91, 0.53, 2.2.
Not predicted: in the quadrotor and Reacher the ensemble switch is as far from the demonstrator as the labeler alone
(62.7 and 1.08), so the labeler must act on nearly every step. Only
in the pendulum does it keep some fidelity (2.9 vs labeler 8.2).
