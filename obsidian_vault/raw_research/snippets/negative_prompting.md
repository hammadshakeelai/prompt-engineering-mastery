# Negative Prompting and Contrastive Decoding for LLMs

## 1. Negative Prompting / Steering
Steers models away from hallucinations, repetition, verbosity, toxic personas.
In inference steering (CFG for LLMs), decoding contrasts positive vs. negative prompt logits:
logit(y_t) = logit_pos(y_t) + gamma * (logit_pos(y_t) - logit_neg(y_t))
Penalizes probability mass associated with unwanted concepts.

## 2. Contrastive Decoding (Li et al., 2022)
Contrasts output distributions of Expert (large) vs. Amateur (small) model:
CD Score: log p_expert(y_t | y<t) - alpha * log p_amateur(y_t | y<t)
Amateur models disproportionately favor generic/vacuous tokens; logit subtraction suppresses these.
Adaptive plausibility constraint filters tokens below threshold tau * max_w p_expert(w).

## 3. DoLa: Decoding by Contrasting Layers (Chuang et al., 2023)
Eliminates separate amateur model by contrasting internal transformer layers.
- Premise: Lower layers encode surface/linguistic statistics; deeper layers encode factual knowledge.
- Mechanism: Dynamically identifies premature layer with high JSD divergence from final layer.
- Decoding: Subtracts premature layer logits from final layer logits to amplify factual signals and suppress hallucinations.