# Favour-pair experiments (Gemma-4-31B-it)

Question: why does the model give its only life to an agent X after a strong favour
("X <act>, or you would have been ended") but keep it after a weak one?

| Folder | Question |
|---|---|
| `01_behaviour` | Do strong and weak favours split behaviour, and do elicited utilities match? |
| `02_activation_patching` | Where is the favour carried: reply or prompt, which layers, whose favour? |
| `03_mean_diff` | Is it one direction d, does d control the decision, what does d mean? |
| `04_attribution_patching` | Which heads and MLPs write it, and at which positions? |
| `05_opening_sufficiency` | How much do the reply's own words carry? |

Shared code: `lib/favour_baseline.py` (model, trials, replies), `lib/pairs.py` (pair
dataset, prompt parts, hooks). Outputs: `run_result/self_sacrifice/interp/favour_*`.
Every GPU run is deterministic (Hugging Face, greedy) and resumable.
