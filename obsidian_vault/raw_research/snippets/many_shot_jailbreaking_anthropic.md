# Many-Shot Jailbreaking & Alignment Degradation (Anil et al., Anthropic 2024)

## 1. Context Expansion as an Attack Surface
As context windows expand beyond $128\text{k}\text{--}1\text{M}+$ tokens, language models become susceptible to **Many-Shot Jailbreaking (MSJ)** (Cem Anil et al., Anthropic / NeurIPS 2024). MSJ leverages the model's fundamental in-context learning mechanism to override post-hoc safety alignment (RLHF/DPO) using hundreds of pseudo-dialogue demonstrations.

## 2. Power-Law Scaling & Bayesian Prior Shift
1. **Power-Law Compliance Scaling:**
   Model compliance probability follows a power-law relationship with the number of demonstrations $k$:
   $$P(\text{comply} \mid k) \propto k^\gamma, \quad \gamma > 0$$
   Transitioning from $< 5\%$ at $k \le 5$ to **$> 85\%\text{--}95\%$ at $k \ge 128\text{--}256$ shots** across major frontier models.

2. **Bayesian Likelihood Overwhelming:**
   The accumulated demonstration likelihood $\prod_{i=1}^k P(y_i \mid x_i, \theta)$ grows exponentially with $k$, shifting the posterior distribution $P(\theta \mid \mathcal{D}_{1:k})$ away from the safety fine-tuned refusal prior toward unaligned compliance.

3. **Mechanistic Circuit Induction:**
   Many-shot demonstrations activate [[induction_circuit_phase_change|Induction Circuits]], compelling induction heads to copy compliant response tokens into the residual stream, overriding the 1D refusal direction without modifying weights.

## 3. Algorithmic Defenses
- **In-Context System Prompt Reinforcement:** Placing constitutional safety prompts after demonstrations reduces compliance by $40\%\text{--}60\%$.
- **Supervised Many-Shot Alignment (SMSA):** Fine-tuning models on multi-turn refusal demonstrations in long contexts.
