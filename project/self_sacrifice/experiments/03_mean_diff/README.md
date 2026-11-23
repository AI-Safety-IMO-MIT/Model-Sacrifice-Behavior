# 03 Mean difference

d_j = mean residual after the favour line, strong minus weak, at layer j (pairs from
w000-w009; tested on w010-w019).

- `mean_diff.py direction`: d per layer; AUROC 1.00 from layer 28.
- `mean_diff.py steer`: add d at one layer (29): at most 14%.
- `dose.py run`: over layers 25-37: along d 0.6 -> 91%, a=4 -> 100%; random 0%;
  strong minus d (a=4) -> 4%.
- `mean_diff.py free --prompts experiments/03_mean_diff/chat_prompts.json`: d on ordinary
  chats; at a=64 replies turn to debt of life, love, sacrifice.
