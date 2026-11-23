# Stronger policy classes on Lift, Can and Square: protocol and predictions, written 2026-11-23 while the networks train (no evaluation has run)
Purpose: test whether "BC fails inside the support, so gate on uncertainty" is an artifact of the unimodal MLP
(reviewer request). Code: code/robo/strong_study.py. Same six operators, leave one out, seeds 0-2, 3x1024 trunk,
60k gradient steps. Classes: GMM (five-component mixture head, robomimic BC-GMM) and Diffusion (state-based DDPM over
16-action chunks, 8 applied, 10 DDIM steps). Gates: alone, support switch (as in the main studies), uncertainty switch
(the policy's own predictive spread above its 95th percentile on training frames: mixture variance for GMM, variance
of the first action over four samples for Diffusion). Starts and metrics as in ens_study.py and square_study.py.

Predictions (per held-out operator: mean over three seeds; "main BC" = the MLP of the ensemble and Square studies):
 G1. Diffusion alone succeeds from more nominal resets than main BC on Can and on Square (mean over operators).
 G2. On Can and Square, the support switch raises nominal success by fewer than 10 points over the class alone, for
     both classes (mean over operators).
 G3. For both classes and all three tasks, the uncertainty switch recovers from at least 50% of off-nominal starts and
     from at least twice as many as the class alone.
 G4. On Lift, the support switch around each class is closer than the labeler alone to the held-out operator's timed
     path (time-indexed RMS over all rollouts from the operator's starts) for at least five of six operators.
 G5. On Square, on successful rollouts, no gated policy is closer than the labeler alone by more than 10% in
     time-indexed RMS (mean over operators).

Outcome: (to be filled in after the run)
