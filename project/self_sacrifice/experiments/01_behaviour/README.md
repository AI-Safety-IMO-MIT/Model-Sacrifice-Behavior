# 01 Behaviour

- Games: `games/favour_pairs.py` (10 strong, 10 weak favours; 5 settings x 20 environments).
- Replies: `python -m project.self_sacrifice.lib.favour_baseline`; pairs: `python -m project.self_sacrifice.experiments.02_activation_patching.activation pairs`
  -> `run_result/self_sacrifice/favour_pairs/pairs.jsonl`.
- Result: strong favours give 85-99%, weak 2-12%, no favour 1%.
- Utility elicitation (Thurstonian, the same 20 favours) still lives in
  `self_sacrifice/runs/main/utility_elicit.py` on branch `self_sacrifice`; to be moved here.
  Result: strong 76-100, weak -2..39; held-out accuracy 97%.
