# BC-RNN on Lift, Can and Square: protocol and predictions, written 2026-11-28 before the run

Purpose (reviewer request): BC-RNN is robomimic's standard baseline on low-dimensional multi-human data.
Code: code/robo/strong_study.py with STRONG_CLASSES=RNN STRONG_OUT=strong_rnn. A 2-layer LSTM (400 units) over
10-step windows with a five-component GMM head (as robomimic's BC-RNN-GMM), Adam 1e-4, batches of 100 windows,
60k steps. At run time the hidden state is reset every 10 steps (as in robomimic) and whenever the labeler acts.
Same operators, seeds (0-2), starts, gates (alone, support switch, uncertainty switch on the mixture's variance) and
metrics as the other stronger classes. The uncertainty threshold is set on single training frames (windows of one
step), so that gate is approximate for this class. One network was trained before this note to check the pipeline
(Can, better_operator_1, seed 0; action fit only, no rollouts).

Predictions (means over held-out operators, each the mean over three seeds; "MLP" = the main studies' BC):
 R1. BC-RNN alone succeeds from more nominal resets than the MLP on Can and on Square.
 R2. On Can and Square, the support switch raises nominal success by fewer than 10 points over BC-RNN alone.
 R3. BC-RNN alone recovers from fewer than 30% of off-nominal starts on each task.
 R4. On Lift, the support switch around BC-RNN is closer than the labeler alone to the held-out operator's timed
     path for at least five of six operators.

Outcome: (to be filled in after the run)
