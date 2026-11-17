# Lift follow-ups (2026-11-16): action-chunking BC and look-ahead switch

## Action-chunking BC (lift_chunk.py) -- exploratory, no prediction recorded

Run to answer "is the feed-forward BC too weak?". No prediction was written before the run; the result
(seeds 0-2: chunked BC 90% nominal success, 6% recovery; support switch on it 95% / 86%) is reported as
exploratory. Seeds 3-4 were started after seeing seeds 0-2.

## Look-ahead switch (lift_lookahead.py) -- prediction written before the corrected run

History: a first run (outputs/robo/lift_big_la_gripbug) restored the MuJoCo state after each look-ahead but not
robosuite's gripper command state (`gripper.current_action`) or the env time counters, so the real step was
taken with a gripper command advanced by the look-ahead. That run (success 68%, recovery 80%) is discarded.
The fix restores both; a check applying the same 8 actions with and without 10-step look-aheads in between
differs by 4e-4 in the next observation, the same as two repeats with no look-ahead (4.5e-4).

This prediction is written after seeing the discarded run but before any result of the corrected one.

- LA1. Recovery within 10 points of the support switch (86%), and above the error switch (71%).
- LA2. Nominal success within 10 points of the support switch (83%).
- LA3. Distance to the held-out operator within 0.5 cm of the support switch (7.0 cm).

Outcome (seeds 0-2, 18 units; written 2026-11-16 after the run): recovery 85% (support switch 86%, error switch
71%; above the error switch for all six operators) -- LA1 held. Nominal success 76% (support switch 83%; lower for
five of six operators) -- LA2 held (within 10 points), though the support switch is better. Distance 7.1 cm (support
switch 7.0) -- LA3 held. BC acted on 83% of the steps from the operators' starts.

Five seeds (30 units, 2026-11-17; seeds 3-4 run with the same code after seeds 0-2 were seen): look-ahead switch
recovery 85% (support 86%, error switch 72%; above the error switch for all six operators), nominal success 74%
(support 80%; lower for five of six), distance 7.1 cm (support 7.0; support closer for three of six). LA1-LA3 hold.
Action chunking, five seeds: chunked BC 92% nominal success, 5% recovery; support switch on it 96% / 85% / 6.9 cm.
