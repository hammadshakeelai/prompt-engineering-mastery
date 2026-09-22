---
name: diffusion-lm-engineer
description: Specialized directive for non-autoregressive discrete diffusion language models (SEDD, Plaid, MDLM, LAVE), continuous-time Markov jump processes, bidirectional infilling, and non-causal grammar-constrained decoding.
---

# Diffusion LM Engineer Skill

Use this skill when developing, steering, or evaluating discrete diffusion language models (dLLMs), non-autoregressive text generators, continuous-time Markov jump score models, bidirectional infilling pipelines, or non-causal grammar-constrained decoders.

## 1. Discrete Diffusion & Score Entropy Formulation

1. **Continuous-Time Markov Jump Processes:**
   - In discrete token spaces $\mathcal{V}^L$, replace continuous Gaussian diffusion with Markov jump processes where state transitions occur via Poisson rates $R_t(x, y)$.
   - Define concrete transition probability ratios (concrete scores):
     $$s_\theta(x, t)_{i, y} \approx \frac{p_t(x \setminus \{x_i\} \cup \{y\})}{p_t(x)}$$
2. **Score Entropy Training (SEDD Protocol):**
   - Optimize discrete models using the score entropy objective (Lou et al., ICML 2024):
     $$\mathcal{L}_{\text{SEDD}}(\theta) = \mathbb{E}_{t, x_0, x_t}\left[ \sum_{i=1}^L \sum_{y \in \mathcal{V}} w_t(y) \left( s_\theta(x_t, t)_{i, y} - \log s_\theta(x_t, t)_{i, y} \right) \right]$$
   - This directly models transition likelihoods without requiring continuous softmax relaxation or Gumbel noise approximations.

## 2. Bidirectional Infilling & Flexible Prompting

1. **Non-Causal Conditioning:**
   - In contrast to autoregressive models bound to left-to-right prompt prefixes, dLLMs condition on arbitrary prompt masks $\mathcal{M}_{\text{prompt}}$:
     $$x_t = \text{Concat}(x_{\text{prefix}}, [\text{MASK}]^K, x_{\text{suffix}})$$
   - Iteratively denoise the interior masked slots while holding boundary tokens fixed, enabling native code editing, infilling, and insertion tasks without KV cache shifts.
2. **Dynamic Test-Time Compute Scaling:**
   - Decouple generation compute from sequence length. Scale sampling fidelity at inference time by adjusting reverse integration steps $N_{\text{steps}} \in [16, 1024]$ using predictor-corrector or Euler-step reverse jump samplers.

## 3. Non-Causal Grammar Constrained Decoding (LAVE Protocol)

1. **The Masked State Evaluation Dilemma:**
   - Conventional CFG parsers (Outlines, XGrammar) cannot evaluate partial strings with interior mask holes ($[\text{MASK}]$).
2. **Lookahead-then-Verify Architecture:**
   - Extract parallel marginal logit distributions across all masked slots: $p_\theta(x_i \mid x^{(t)}), \, \forall i \in \mathcal{M}$.
   - Propose candidate unmaskings $\hat{x}_U$ and sample $N$ complete lookahead completions $\tilde{x}^{(j)}$ for remaining masks.
   - Verify complete lookahead strings against a formal deterministic CFG parser $\mathcal{P}_{\text{CFG}}$ (e.g., Python AST, JSON validator).
   - Accept the candidate step if at least one lookahead branch satisfies the formal grammar:
     $$\bigvee_{j=1}^N \mathbf{1}[\mathcal{P}_{\text{CFG}}(\tilde{x}^{(j)}) \in \mathcal{L}(G)] == 1$$
   - This guarantees $100\%$ syntactic validity with high proposal acceptance ($\alpha > 97\%$).
