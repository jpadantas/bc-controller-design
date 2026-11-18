# Square (robomimic NutAssemblySquare, multi-human): protocol and predictions, written 2026-11-18 before any policy is trained
Purpose: a confirmatory test, on a task not used to shape anything, of the recommendation the paper arrived at after
Lift, Can and the ensemble study: clone the operators, keep the labeler behind a monitor that sees the full state, and
where the cloned policy fails inside the support gate also on an ensemble's disagreement; use a residual blend only if
the labeler resembles the operators. Raised by every reviewer of round 17.

Code: code/robo/square_common.py (labeler, rollout), code/robo/square_study.py (study). Same six operators
(50 demonstrations each), leave one out, training seeds 0-2. Networks, steps, monitors and metrics as in the Can and
ensemble studies (3x1024 MLP, 60k steps; workspace = end effector and nut within the training range + 2 cm; support
threshold 95th percentile of leave-one-operator-out nearest-neighbour distances; ensemble of five networks, threshold
95th percentile of the summed action variance on the training frames). Residual blend: synthetic states with the arm's
joints perturbed by N(0, 0.5 rad) and the nut anywhere in x in [-0.20, -0.02], y in [0.02, 0.30], any yaw (30% of the
frames). Evaluation: 20 nominal resets, 20 off-nominal starts (joints perturbed by 0.4 rad, nut anywhere in that
region), every other initial state of the held-out operator; per rollout success, time-indexed RMS gap and DTW gap to
the operator's end-effector path.

Labeler (scripted, written and checked before this note): grasps the handle or a wall of the nut on the robot's side,
turns the nut to the peg's orientation, carries it over the peg and drops it from just above. Checked before any policy
was trained (final version): 40 of 40 nominal resets (simulator seeds 0-39, which were also used to develop it),
17 of 20 off-nominal starts and 19 of 20 operator starts (not used in development).

Predictions (per held-out operator: mean over three seeds):
 S1. BC succeeds from fewer than 50% of nominal resets (as on Can, it fails inside the support).
 S2. The support switch raises nominal success by fewer than 10 points over BC.
 S3. The ensemble switch raises nominal success over BC by at least 20 points for at least five of six operators,
     and recovers from at least twice as many off-nominal starts as the support switch.
 S4. On the held-out operator's successful rollouts, the ensemble switch is closer than the labeler in time-indexed
     RMS gap for at least four of six operators, and its mean DTW gap is within 10% of the labeler's (pace, not shape).
 S5. The residual blend recovers from fewer off-nominal starts than the ensemble switch for at least five of six
     operators.
 S6. The frozen rule (error switch if BC never leaves the workspace from the nominal resets, else the residual blend
     if it never leaves, else the full-state monitor) picks the full-state monitor for at least 15 of 18 units.

Outcome: (to be filled in after the run)
