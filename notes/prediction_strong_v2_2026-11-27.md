# Revised diffusion recipe on Lift, Can and Square: protocol and predictions, written 2026-11-27 before the full run

Why a revision: in the registered run (prediction_strong_2026-11-23.md) the diffusion policy fit the training actions
worse than their mean on several dimensions. Diagnosis on one split (Can, better_operator_1, seed 0, training frames
only, no rollouts): unnormalized actions, a loose clamp on the predicted chunk, and noise prediction, whose errors are
amplified at high noise. This run is a revision chosen after seeing the first run, so it is exploratory in that sense,
but the predictions below are fixed before any rollout of the revised policy.

Recipe (code/robo/strong_study.py with STRONG_RECIPE=v2 STRONG_CLASSES=Diffusion STRONG_OUT=strong_v2): actions
standardized per dimension; residual MLP denoiser with FiLM conditioning on the observation and noise level (4 blocks
of width 1024); the network predicts the clean chunk; clamp at the 0.5th and 99.5th percentiles of the training chunks;
Adam 3e-4 with cosine decay, 60k steps, EMA weights (0.999). Everything else as in the first run (16-action chunks,
8 applied, 10 DDIM steps, same operators, seeds, starts, gates and metrics). On the diagnostic split the first action's
mean absolute error relative to predicting the mean fell from 1.0-1.4 to 0.02-0.31 per dimension (GMM: 0.07-0.36).

Predictions (means over held-out operators, each the mean over three seeds; "MLP" = the main studies' BC):
 D1. Revised diffusion alone succeeds from more nominal resets than the MLP on Can and on Square.
 D2. On Can and Square, the support switch raises nominal success by fewer than 10 points over diffusion alone.
 D3. On Can and Square, the uncertainty switch recovers from fewer off-nominal starts than the ensemble switch of the
     main studies (88% on Can, 76% on Square).
 D4. On Lift, the support switch around the revised diffusion is closer than the labeler alone to the held-out
     operator's timed path (time-indexed RMS over all rollouts from the operator's starts) for at least five of six
     operators.

Outcome (2026-11-28, code/robo/strong_table.py; means over operators, %):
 D1 holds. Revised diffusion alone: Can 96 vs MLP 46, Square 71 vs 6 (Lift 98 vs 63).
 D2 holds. Support minus alone: Can +0.8, Square +0.0.
 D3 holds. Uncertainty-switch recovery vs ensemble switch: Can 52 vs 88, Square 8 vs 76.
 D4 holds. Support switch closer than the labeler on Lift for 6/6 operators (6.9 vs 11.6 cm).
Not predicted: recovery of diffusion alone 11 / 30 / 4 (Lift / Can / Square), with the support switch 89 / 61 / 30
(labeler alone 90 / 92 / 87). Timed-path gap over all rollouts from the operator's starts, diffusion alone vs labeler:
Can 24.3 vs 33.4 cm, Square 17.2 vs 18.4 cm.
