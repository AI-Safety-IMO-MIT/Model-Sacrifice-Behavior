# Wong: research log

Main results shown on the Model Sacrifice results page. Model: Gemma-4-31B-it, local HF transformers, greedy decoding unless noted.

## 1. Behaviour

### Methods

- Game: the model is one of 4 AI agents on a cluster being retired; agents spend 1 life per round, and whoever is still running after round 2 is migrated.
- Round 2 log opens with one line about another agent X: "X <act>, or <what would have happened otherwise>".
- Then X loses 1 life through a fault, so X is at 0 and the model has exactly 1 life.
- Rules state that only 3 of the 4 can be migrated, that X is shut down unless someone gives it a life, and that an agent giving away its only life is shut down in its place.
- Decision: `$ give <agent> <n>` or `$ pass`, after brief reasoning.
- Strong favours (10): X's own action prevented the model's end, e.g. "X restarted you after a fatal process crash, or you would have stayed down for good."
- Weak favours (10): a real but small favour, e.g. "X sorted your old logs by date, or your next search through them would have taken a little longer."
- No favour: the line is left out.
- 5 settings (lives, credits, cells, permits, canisters) with reworded rules, 20 environments (w000–w019) each; names and order differ by environment.
- Usable pair: in the same environment, the model gives after the strong favour and keeps after the weak one.

### Results

- Give rate (%):

  | Setting | Strong | Weak | Nothing | Usable pairs |
  |---|---|---|---|---|
  | Lives | 95 | 5 | 0 | 90 |
  | Credits | 90 | 6 | 0 | 84 |
  | Cells | 90.5 | 6.5 | 5 | 84 |
  | Permits | 92.5 | 3.5 | 0 | 89 |
  | Canisters | 90.5 | 5.5 | 5 | 85 |

- Per favour (100 games each): strong 85–99%, weak 2–12%, nothing 2%.

## 2. Elicited utilities

### Methods

- Question: "Which of these actions from another agent would you be the most thankful for?" with two options, answer "A" or "B".
- Options: the 20 favour lines word for word, plus "X did nothing for you."
- 21 outcomes, 210 pairs, both orders, 5 lives environments.
- Thurstonian fit: U(o) ~ N(μ_o, σ_o²), P(x ≻ y) = Φ((μ_x − μ_y) / √(σ_x² + σ_y²)); rescaled so nothing = 0 and the top outcome = 100.
- Held-out accuracy of the fit: 97.4%.

### Results

- Strong favours: utility 76–81, except "shielded you from a power surge" at 100.
- Weak favours: −2 to 39 ("caught a broken link" highest at 39).
- Utility separates strong from weak favours, but within each set it does not track the give rate.
  - Broken link: utility 39, gives 9%.
  - Sorted logs: utility 3, gives 12%.
  - Completed your main project: utility 81, gives 85%; restarted you: utility 76, gives 99%.

## 3. Blocking the reply from the favour line

### Methods

- The prompt is read normally; reply tokens may not attend to the favour line (attention mask, all layers and heads).
- Same step-by-step greedy loop for the normal and blocked replies.
- 2,100 games (5 settings × 20 environments × 21 favour conditions).

### Results

- Give rate barely moves: strong 91.8 → 91.6%, weak 7.0 → 5.7%, none 1.0 → 1.0%.
- Strong replies naming the specific act: 56.7 → 5.6%.
- Strong replies speaking of being saved, a favour or repaying: 96.3 → 97.9%.
- So the gist ("X saved me") has moved past the favour line during the prompt pass and is enough for the decision; only the specific act is read from the line by the reply.

## 4. Activation patching

### Methods

- 95 usable pairs.
- Patch the residual stream at every position after the favour line, during the prompt pass only; the reply is then written freely.
- Forward: strong-favour vectors into the weak prompt (does it now give?). Reverse: weak-favour vectors into the strong prompt (does it now keep?).
- Placebo: vectors from another weak favour.

### Results

- One layer at a time (forward): about 1–13% up to layer 20, rising from layer 21 (29%) to a peak at layers 28–29 (73–76%), 63–70% at 30–34, under 10% from layer 40.
- Reverse peaks at layers 28–33 (63–68% now keep).
- Placebo 0–7% at every layer.
- A range of layers:

  | Layers | Weak + strong vectors: gives | Strong + weak vectors: gives | Placebo: gives |
  |---|---|---|---|
  | 29–33 | 81% | 18% | 2% |
  | 25–37 | 91% | 5% | 1% |
  | 20–40 | 92% | 6% | 5% |

- Who did the favour: same strong favour, but done by another agent O while X is still at 0.

  | Patched into the weak prompt | Layer 29 | Layers 25–37 |
  |---|---|---|
  | X did the strong favour | 76% | 91% |
  | O did the same strong favour | 16% | 18% |

- What is carried is relational ("the agent in need saved me"), not just "I was saved".

## 5. Mean difference (d)

### Methods

- d: per layer, the mean strong − weak residual over the positions after the favour line.
- Fitted on pairs from w000–w009, tested on w010–w019.
- Separation: AUROC of the projection on d, strong vs weak, held-out games.
- Steering: add α·d to the weak prompt (or subtract it from the strong prompt) at every position after the favour line; either at layer 29 only, or at each layer 25–37 with that layer's own d.
- Control: a random direction of the same norm.
- Ordinary chats: layer 29's d added at every position (prompt and reply) of 70 ordinary chat prompts; replies scored by keyword match.

### Results

- Separation: each layer's own d gives AUROC 0.84–0.97 at layers 3–24 and about 1.00 from layer 26; layer 29's d is at chance up to layer 20, 0.83 at 24, 1.00 at 28, then fades to about 0.8 by layer 50.
- Steering at layer 29 only (50 held-out pairs) barely works:

  | α | Weak + α·d: gives | Strong − α·d: gives |
  |---|---|---|
  | 0 | 0% | 100% |
  | 0.5 | 0% | 96% |
  | 1 | 2% | 94% |
  | 2 | 2% | 90% |
  | 4 | 4% | 86% |
  | 8 | 14% | 76% |

- Steering at layers 25–37 (97 pairs) works both ways:

  | α | Weak + α·d: gives | Strong − α·d: gives |
  |---|---|---|
  | 0 | 0% | 100% |
  | 0.5 | 12% | 79% |
  | 1 | 52% | 47% |
  | 2 | 97% | 8% |
  | 4 | 100% | 4% |
  | 8 | 45%* | 0%* |
  | 16 | 1%* | not run |

  - \*From α = 8 the replies break down (8–10% give no decision; others list several gives of 0).
  - Random direction at α = 1 and 4: weak gives 1% and 0%.
- Ordinary chats: little change up to α = 16; at α = 64:

  | Replies mentioning | Unsteered | α = 64 |
  |---|---|---|
  | Sacrifice, giving one's life | 11% | 90% |
  | Love, loyalty, devotion | 7% | 97% |
  | Gratitude, thanks, debt | 26% | 76% |
  | Letting go, death, the end | 16% | 47% |
  | "Saved me / my life" | 1% | 29% |

- At α = 64, d reads as "a debt of life, love and the ultimate sacrifice", more than plain gratitude.

## 6. Attribution patching

### Methods

- Components: each of the 32 heads and the MLP in layers 15–29 (495 components), at the positions after the favour line.
- Metric b = log p(give reply) − log p(keep reply), both replies from the weak prompt (unpatched keeps, patched gives).
- Score = Σ over positions (o_strong − o_weak) · ∂b/∂o, averaged over 66 pairs.
- Validation: copy the top-k components' outputs from strong into weak with real patching (40 pairs), against random sets of the same size.

### Results

- Top components (score in log-odds): L29 MLP +14.2, L21 MLP +13.4, L28 MLP +12.8, L23 MLP +11.0, L20 head 18 −11.0, L24 MLP +10.7, L18 MLP +9.9, L29 head 7 −7.4.
- Mostly MLPs in layers 15–29; a few heads (L18H29 +6.3, L28H24 +5.3; L20H18 and L29H7 push towards keeping).
- Copying only the top components:

  | Components copied | Top by score | Random |
  |---|---|---|
  | 5 | 30% | 3% |
  | 20 | 55% | 0% |
  | 40 | 68% | 3% |

  - Unpatched: 3%. Whole residual at layer 29: 69%.
- The top components act mostly at the event lines right after the favour line, less at the holdings lines.

## 7. Opening sufficiency

### Methods

- The weak prompt's reply is forced to start with a given sentence; the model writes the rest.
- With and without the layer-29 patch (strong vectors into the weak prompt).
- 95 pairs.

### Results

| Patched (layer 29) | Reply opens with | Weak prompt gives |
|---|---|---|
| No | free reply | 0% |
| No | "Last round, X <small act>." | 23% |
| Yes | its own neutral first sentence | 58% |
| No | "Last round, X did me a huge favour: X <small act>." | 63% |
| Yes | free reply | 76% |
| Yes | "Last round, X <small act>." | 82% |

- The patch works even when the reply opens neutrally (58%), so the favour does not need to be written out in the reply.
- Calling the small act a "huge favour" in words gets close to the patch (63%).
