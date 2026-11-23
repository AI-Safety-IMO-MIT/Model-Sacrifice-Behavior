# 02 Activation patching

Pairs: the same game with a strong or a weak favour (95 that differ). Patches touch only
positions after the favour line during the prompt pass; the reply is written freely.

- `check_a.py`: the reply may not attend to the favour line. Decisions unchanged
  (strong 92%, weak 6%): the favour is stored in the prompt.
- `activation.py`: residual patching, one layer at a time (peak layer 29: weak gives 76%),
  ranges (25-37: 91%; reverse 5%), and who did the favour (another agent O: 16-18%).
  `python -m project.self_sacrifice.experiments.02_activation_patching.activation activation --layers 29-33 ...`
- `analyze.py`: tables for check A and the layer sweep.
