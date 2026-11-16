# Closed-loop cross-validation as the selection rule (written 2026-11-13 ~08:20, before running it)
Motivation: on the first 105 held-out seeds, the three-outcome rule (decided on retention) was worse than
always using the look-ahead switch (excess 0.047 vs 0.007 of the labeler's distance), because a residual
blend can stay nominal and still be far from the demonstrator. Retention cannot see fidelity.
Rule CV (frozen now, code/v3/cvselect.py): K-fold over demonstration episodes (K = min(episodes, 5)); per
fold train BC and the residual blend on the other episodes; run the error switch, the look-ahead switch and
the residual blend from the held-out episodes' initial states; choose the candidate with the smallest RMS
distance to the recorded held-out trajectories, among candidates that never leave N from those starts
(if none, the look-ahead switch). Scored with the full-data policies of run.py (distance on common starts),
same tolerance (5% of the labeler's distance) and excess measures as rule3, per group (design / Reacher
registered / held-out).
Prediction: rule CV's mean excess is at most that of the best fixed choice in each of the three groups.

## Outcome (seen 2026-11-13 ~09:45): prediction FAILED
Mean excess over the closest of three (/labeler's distance): design 0.082 (rule3 0.026, look-ahead 0.080);
Reacher registered 0.020 (look-ahead 0.000); held-out 0.118 (rule3 0.026, look-ahead 0.025).
With few episodes, folds train on even fewer, and closed-loop validation on a handful of held-out starts is
too noisy to rank the candidates; the retention test (rule3) does better. CV is reported as a failed
alternative in the appendix.
