# Lift (robomimic multi-human) study: protocol and predictions, written 2026-11-13 ~16:00 before the full run
Six human operators (2 better, 2 okay, 2 worse; 50 demos each), leave one operator out, 3 training seeds.
Labeler: scripted pick-and-lift law (code/robo/lift_common.py), unlike any operator near nominal.
Nominal set N for the rule: the demonstrated workspace (end-effector and cube positions within the training
demonstrations' bounding box + 2 cm). The three-outcome rule as frozen: error switch (monitor on positions)
if BC never leaves N from the nominal starts as often as the labeler does not; else the residual blend if it
never leaves; else the full-state monitor without a model (support switch).
Predictions (from the controlled domains and flight):
 L1. Off-nominal starts: BC recovers rarely; the position-only switch recovers less than the support switch;
     the labeler, the support switch and the residual blend recover most often.
 L2. The labeler is a mismatched case (its action error on held-out operators is several times BC's), so the
     residual blend and the box blend are no closer to the held-out operator than the support switch.
 L3. The rule's choice is within 5% of the labeler's distance of the closest of {switch, residual, support}
     on most (operator, seed) units.

## Outcome (seen 2026-11-13 18:10; 6 operators x 3 seeds)
Recovery: BC 0.3%, box blend 10%, residual 63%, error switch 71%, support switch 85%, labeler 91%.
L1: BC recovers rarely (yes); position-only switch < support switch (yes, 71 vs 85); residual among the best
    recoverers (NO: 63%). -> partly confirmed.
L2: residual (9.9 cm) and box blend (8.2) no closer than the support switch (7.4): CONFIRMED (6/6 operators).
L3: rule picks support on 16/18 units, residual 1, switch 1; within 5% of the labeler's distance of the closest of
    three on 17/18 (always support 17, always residual 7, always switch 4); excess 0.7%: CONFIRMED.
Also: support switch is closer to the held-out operator than the labeler alone (7.4 vs 11.6 cm, 6/6 operators),
but completes the task from nominal starts less often (72% vs 99%; BC alone 54%).
