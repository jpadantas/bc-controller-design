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

Outcome: (to be filled in after the run)
