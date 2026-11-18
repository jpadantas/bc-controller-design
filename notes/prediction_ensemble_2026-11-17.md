# Ensemble (uncertainty) monitor on Lift and Can: protocol and predictions, written 2026-11-17 before any run
Raised by reviewers (round 15): the "full-state monitor" of the paper is a nearest-neighbour support test; would
an uncertainty monitor, as in ThriftyDAgger or EnsembleDAgger, change the conclusions, and would a better monitor
rescue Can, where BC fails inside the demonstrated workspace?
Code: code/robo/ens_study.py. Per (task, held-out operator, seed), tasks Lift and Can, six operators, seeds 0-2:
BC exactly as in the main studies (3x1024, 60k steps, same seed) plus four networks trained on the same data with
other seeds. Ensemble switch: BC acts unless the state leaves the demonstrated workspace or the ensemble's
disagreement (summed variance of the five predicted actions) exceeds its 95th percentile on the training frames;
otherwise the labeler. Compared on the same networks and starts with BC, the support switch and the labeler alone.
Starts: 20 nominal, 20 off-nominal (as in each main study, drawn anew for this run), every other initial state of
the held-out operator.

Predictions:
 E1. Lift: the ensemble switch recovers from at least 70% of off-nominal starts and its mean distance to the
     held-out operator is within 1 cm of the support switch's.
 E2. Can: neither monitor raises nominal success by more than 10 points over BC, and the ensemble switch recovers
     from fewer than 30% of off-nominal starts (failures inside the support are not flagged as novel).
 E3. On Lift, the ensemble switch is closer to the held-out operator than the labeler alone for at least five of
     six operators.
Reported without a prediction: the fraction of steps in which BC acts under each monitor.

Outcome (filled in 2026-11-18 after the run; means over six held-out operators and three seeds, labeler once per
operator; outputs/robo/ens/ens_results.csv, paper/tables/ens.tex):
 E1. Failed in part. The ensemble switch recovers from 85% of off-nominal starts on Lift (support switch 81%), but
     its distance is 9.7 cm against the support switch's 7.1 cm (support switch closer for five of six operators):
     it hands control to the labeler in 70% of steps (BC acts 30%; support switch 92%).
 E2. Failed. On Can the support switch leaves nominal success unchanged (46% vs. BC 46%), but the ensemble switch
     raises it to 94% (higher for all six operators) and recovers from 88% (support switch 13%, labeler 92%). The
     networks disagree where BC fails, inside the support. Distance 27.8 cm (support 35.7, labeler 33.4, BC 36.9;
     two operators 26.5 in the Can study); closer than the labeler for five of six operators, than the support
     switch for three.
 E3. Held: on Lift the ensemble switch is closer than the labeler for six of six operators (9.7 vs. 11.6 cm).
BC fraction: Lift ensemble 30%, support 92%; Can ensemble 53%, support 96%.
