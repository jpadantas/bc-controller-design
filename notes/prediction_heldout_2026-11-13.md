# Prospective test of the three-outcome rule on held-out settings
Written 2026-11-13 ~07:05 (local), BEFORE running any of the settings below. The rule is frozen as in
code/v3/stats_v3.py::rule3 (kept_inner of BC >= labeler's -> error switch; else kept_inner of the residual
blend >= labeler's -> residual blend; else look-ahead switch). Nothing in the rule, the methods, the
training or the metrics will be changed after this point.

Held-out settings (none used to design the rule):
- pendulum, original labeler, 10 episodes (new budget)
- pendulum_hg4: tanh(4 k), 10 and 30 episodes (mismatch in a new domain)
- pendulum_q10: three-level labeler sign(k)*[|k|>0.1], 10 and 30 episodes (new family, new domain)
- quad3d_q05: three-level labeler sign(k)*[|k|>0.05], 3, 5, 10 episodes (new family)
- quad3d_hg4, 30 episodes (new budget)
- reacher_hg2: tanh(2 k), 3, 10, 30 episodes (new gain)
All labelers checked to recover from every probe radius before training anything (reacher with the
three-level labeler does not, and is not included).

Scoring (fixed now): per seed, distance on common starts of {error switch, residual blend, look-ahead switch};
a choice is "within tolerance" if its distance exceeds the closest of the three by at most 5% of the
labeler-alone distance in that setting; also mean excess distance / labeler's distance.
Predictions:
 P-a. The rule is within tolerance on more seeds than each fixed choice (always switch, always residual,
      always look-ahead) and than the two-outcome rule.
 P-b. Its mean excess distance is lower than each fixed choice's and the two-outcome rule's.

## Outcome (seen 2026-11-13 ~09:00, scored as registered: decision on inner-start retention)
180 held-out seeds (12 settings). Within tolerance: rule3 163, look-ahead 158, two-outcome rule 154,
switch 135, residual 53 -> P-a CONFIRMED.
Mean excess / labeler's distance: rule3 0.026, look-ahead 0.025, two-outcome 0.095, switch 0.173,
residual 1.26 -> P-b FAILED narrowly (tied with always-look-ahead).
