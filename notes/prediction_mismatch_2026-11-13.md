# Prediction registered before running the mismatched-labeler experiment
Written 2026-11-13 ~05:10 (local), before any policy is trained on the mismatched labelers.

Setup: quadrotor (and Reacher if time allows), labeler replaced by tanh(G * labeler(o)), G in {4, 16};
demonstrations identical to the main experiment. Labeler alone still recovers 100% (checked),
its distance to the demonstrator is unchanged (0.63 m); var(d - kappa) on demonstration states is
8x (G=4) and 35x (G=16) var(d), against 0.76x for the original labeler.

Predictions (3, 5, 10 quadrotor episodes; 15 seeds):
1. Residual check: the residual blend fits held-out demonstrations WORSE than BC for G=4 and G=16
   (it fits them at least as well for the original labeler).
2. Where BC leaves N (3 episodes), the residual blend is no longer the closest-or-tied combination;
   full-state monitors that apply BC inside N (support switch, look-ahead switch) are closer.
3. The extended rule (switch if BC never leaves; else residual if the check passes; else a
   full-state monitor) picks a combination at least as close as the residual blend on most seeds.
4. Recovery of the residual blend and of the full-state monitors stays high (>= 95%).

## Note added 2026-11-13 ~05:40, after the fit checks and before any closed-loop mismatch result
The parenthetical in prediction 1 is FALSE: with the ORIGINAL labeler, the residual blend also fits
held-out quadrotor demonstrations worse than BC on every seed and budget (MSE 0.0030 vs 0.0002 at
3 episodes; 0.00015 vs 0.00001 at 30), yet it is as faithful as the switches in closed loop.
Open-loop held-out action error therefore does not discriminate when the residual helps, and the
"residual check" proposed in the Discussion is withdrawn. Predictions 2-4 stand as written.

## Quadrotor result (seen 2026-11-13 ~06:15)
Prediction 2 confirmed strongly: with tanh(4 k) and tanh(16 k), the residual blend (and the other learned
blends) lose retention (residual 76-94% from inner starts) and fidelity (0.68-1.83 m, worse than the
labeler alone, 0.63 m), while switches and filters, which use the labeler only outside N, are unchanged
(look-ahead / backup 100% retention, 0.32 m at 3 episodes). Prediction 4 holds for recovery (95-99%).

## Extended rule and confirmatory prediction for Reacher (registered 2026-11-13 ~06:20, before running Reacher
## with the mismatched labelers)
Extended rule (no demonstrator, no threshold):
 1. run BC from inner nominal starts; if it never leaves N -> error switch;
 2. else run the residual blend from the same starts; if it never leaves N -> residual blend;
 3. else -> a full-state monitor on BC (look-ahead switch; support switch where no model).
Predictions for reacher_hg4 / reacher_hg16 (3, 10, 30 episodes; 15 seeds):
 A. The residual blend leaves N from inner nominal starts on most seeds (retention < labeler's).
 B. The look-ahead switch is then closer to the demonstrator than the residual blend.
 C. Over all settings (original + mismatched, quadrotor + Reacher + pendulum), the extended rule's choice
    is the closest of {error switch, residual blend, look-ahead switch} on more seeds than any fixed choice.

## Reacher outcome (seen 2026-11-13 ~07:30)
- tanh(16 k) destabilizes the Reacher labeler itself (46% recovery, 14% inner retention): the premise
  (a labeler that recovers) fails, so reacher_hg16 is reported in the appendix and excluded from the rule.
- reacher_hg4: A confirmed (residual leaves N on 15, 13, 15 of 15 seeds at 3, 10, 30 episodes);
  B confirmed (look-ahead closer by 1.0, 0.8, 1.1 cm; Wilcoxon p = 1e-4 each);
  C confirmed: over 255 seeds (8 original + 9 mismatched settings) the extended rule picks the closest
  of {error switch, residual blend, look-ahead switch} on 209; fixed choices: look-ahead 166,
  switch 124, residual 70; two-way rule 164. Mean excess distance / labeler's: 0.025 vs 0.066 (look-ahead),
  0.21 (two-way rule), 0.28 (switch), 0.55 (residual).
