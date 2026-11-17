# Can (robomimic multi-human PickPlaceCan): protocol and predictions, written 2026-11-17 before any run
Six human operators (2 better, 2 okay, 2 worse; 50 demos each), leave one operator out, five training seeds,
3x1024 networks, 60k gradient steps each (as the Lift main run). Code: code/robo/can_study.py, can_common.py.
Labeler: scripted pick-and-place law (align above the can, descend, close, lift to 1.05 m, travel over the bin,
release once the can is over the bin, move up); checked before any policy was trained: 10/10 from nominal
resets, 10/10 with arm joints perturbed (sd 0.4 rad), 10/10 from an operator's starts.
Policies: BC, BC with action chunking (K=8, temporal ensembling m=0.1), each alone, behind the error switch
(workspace monitor: end-effector and can position within the demonstrations' range + 2 cm) and behind the support
switch (also full-observation kNN distance <= 95th percentile of cross-operator distances); box blend and
residual blend (synthetic states: joints sd 0.5 rad, can anywhere in its bin; 30% of the training frames);
labeler alone. Starts: 20 nominal resets; 20 off-nominal (joints sd 0.4 rad, can anywhere in its bin);
every other initial state of the held-out operator's demonstrations (rollouts capped at 600 steps).
The three-outcome rule as frozen for Lift (error switch if BC keeps every nominal start in the workspace, else
the residual blend if it does, else the support switch), applied to the feed-forward BC.

Predictions:
 C1. Recovery: BC and chunked BC recover from fewer than 30% of off-nominal starts; the support switch recovers
     more often than the error switch built on the same BC, for at least five of six operators (both BCs).
 C2. The labeler is unlike the operators (action error several times BC's), so neither blend is closer to the
     held-out operator than the support switch (mean distance), and the residual blend recovers less often
     than the support switch.
 C3. The support switch (either BC) is closer to the held-out operator than the labeler alone for at least five
     of six operators.
 C4. Chunked BC succeeds from nominal starts more often than feed-forward BC.
 C5. The rule's choice is within 5% of the labeler's distance of the closest of {error switch, residual blend,
     support switch} on at least 80% of the units.

Addendum (2026-11-17, before any policy result): the environment now resets without rebuilding the model
(env.hard_reset = False; same physics, ~600x faster resets). With it the labeler recovered 9/10 perturbed starts
in a second check (different random starts).

Outcome (2026-11-17, all 30 units; 24 ran in one batch, the other 6 ran out of memory at start-up and were rerun
with the same seeds and starts, networks trained first and policies evaluated in parallel; table: paper/tables/can.tex):
 C1. HELD. Off-nominal recovery: BC 2.5%, chunked BC 15%. Support switch above the error switch on the same BC for
     5 of 6 operators with either BC (feed-forward 12% vs 8%; chunked 34% vs 29%).
 C2. FAILED. The residual blend is closer than the support switch (35.4 vs 36.2 cm; 4 of 6 operators) and recovers
     MORE often (25% vs 12%). The box blend is not closer (37.4 cm).
 C3. FAILED. The support switch is closer than the labeler alone for 3 of 6 operators (feed-forward; 36.2 vs 33.2 cm
     on average) and 4 of 6 (chunked; 28.3 vs 33.2 cm).
 C4. HELD. Nominal success: chunked BC 79% vs BC 42%, higher for all 6 operators.
 C5. FAILED. The rule (support 25, error switch 4, residual 1) is correct for 16 of 30 units (53%).
What differs from Lift: BC fails inside the demonstrated workspace. It fails from 58% of nominal starts but leaves the
workspace from 11%, so at least four fifths of its failures happen where every gate lets it act (the support switch
runs it 97% of the time). The labeler alone recovers from 91% and succeeds from 99%. Distances are large for every
policy (28-38 cm; two operators' demonstrations: 26.5 cm, two of one operator's: 14.7 cm).
