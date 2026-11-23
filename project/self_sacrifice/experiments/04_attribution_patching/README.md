# 04 Attribution patching

495 components (32 heads + 1 MLP per layer, layers 15-29). Score = sum over positions
after the favour line of (o_strong - o_weak) . d metric / d o on the weak prompt;
metric b = log p(give reply) - log p(keep reply), both replies from the weak prompt.

- `attribution.py attribute --layer 29 --window 15-30 [--positions]`: scores (per position).
- `attribution.py validate`: copy only the top k components (real patching): top 40 -> 68%,
  random 0-3%.
- `attribution.py heads_where`: where the top heads attend.
