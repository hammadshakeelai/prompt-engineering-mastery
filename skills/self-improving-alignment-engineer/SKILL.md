---
name: self-improving-alignment-engineer
description: Specialized directive for autonomous self-improving alignment, Self-Rewarding Language Models, Meta-Rewarding with LLM-as-a-Meta-Judge, Iterative DPO loops, and self-correction without external human supervision.
---

# Self-Improving Alignment Engineer Skill

Use this skill when designing, training, or evaluating autonomous alignment loops where large language models generate their own preference supervision, act as internal judges or meta-judges, and update their own policies via iterative preference optimization (Iterative DPO, SPPO, Meta-Rewarding).

## 1. Self-Rewarding & Meta-Rewarding Mechanics

1. **Integrated LLM-as-a-Judge (Self-Rewarding, Yuan et al., Meta / ICML 2024):**
   - Eliminate external static reward models by instruction-tuning the policy $\pi_\theta$ to evaluate candidate responses via an internal judge prompt template.
   - For prompt $x$, sample candidate completions $\{y_1, y_2\} \sim \pi_\theta(\cdot \mid x)$.
   - Query $\pi_\theta$ to score each candidate on a 1–5 scale. Construct preference pair $(y_w, y_l)$ based on score delta.
   - Update $\pi_\theta$ via DPO. Simultaneous improvement occurs across both generation and evaluation capability across iterations $M_1 \to M_2 \to M_3$.

2. **Mitigating Judgment Saturation via Meta-Judging (Wu et al., Meta / EMNLP 2024):**
   - *Problem:* In pure self-rewarding, reward modeling capability saturates early, causing the model to reinforce its own evaluation blindspots.
   - *Meta-Judge Protocol:* Train the model to evaluate its own judgment outputs:
     $$\text{Meta-Judge}(J_1, J_2 \mid x, y_1, y_2)$$
   - Uses meta-feedback to iteratively update judge calibration before scoring generation candidates, breaking the saturation plateau and lifting AlpacaEval 2 from $22.9\%$ to $39.4\%$.

3. **Length-Bias Calibration in Self-Generated Rewards:**
   - LLM judges possess intrinsic verbosity bias, systematically rewarding longer answers.
   - **Mandate:** Incorporate length-regularization tiers $\rho$:
     $$y_w \succ y_l \iff \operatorname{Score}(y_w) - \operatorname{Score}(y_l) > \tau + \beta \left( \operatorname{len}(y_w) - \operatorname{len}(y_l) \right)$$
   - Only select longer responses if the quality advantage exceeds the length penalty $\beta \cdot \Delta \text{len}$.

## 2. Iterative Self-Correction & Policy Loops

1. **Iterative DPO Multi-Stage Schedule:**
   - In each self-improvement cycle:
     - Generate $K=4$ candidates per prompt from diverse seeds.
     - Judge candidates with bidirectional permutation swapping.
     - Prune ambiguous ties ($|\Delta s| < 1.0$).
     - Execute 1 epoch of DPO with learning rate $\eta \in [5 \cdot 10^{-7}, 1 \cdot 10^{-6}]$.
   - Cap iterative cycles at $3\text{--}4$ rounds to prevent distribution collapse and mode collapse.

## 3. Evaluation & Validation Directives

- Track both **Instruction Following (Arena-Hard / AlpacaEval 2)** and **Reward Correlation (RewardBench)** across each self-training round.
- If RewardBench correlation drops while AlpacaEval length explodes, trigger SimPO length-normalized rewards immediately.
