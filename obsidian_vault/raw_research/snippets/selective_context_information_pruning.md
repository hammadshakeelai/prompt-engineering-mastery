# Selective Context: Information-Theoretic Prompt Pruning (Li et al., EMNLP 2023)

## 1. Context Bloat & Attention Complexity
Long prompt contexts incur quadratic attention prefill compute $\mathcal{O}(T^2)$ and linear KV-cache memory consumption $\mathcal{O}(T)$. Naive window truncation risks discarding critical prompt dependencies.

## 2. Self-Information (Shannon Surprise) Scoring
Selective Context (Li et al., EMNLP 2023 / arXiv:2310.06201) scores lexical units $u_i$ (tokens, phrases, sentences) by their conditional self-information under a base language model $\mathcal{M}$:
$$I(u_i) = -\frac{1}{|u_i|} \sum_{t=1}^{|u_i|} \log P_{\mathcal{M}}(x_{i, t} \mid x_{<i}, x_{i, <t})$$
- **High Self-Information:** Dense, informative tokens (entities, constraints, propositions).
- **Low Self-Information:** Formulaic filler and redundant phraseology.

## 3. Pruning Protocol & Inference Gains
Given target retention ratio $\rho \in (0, 1)$, retain units exceeding quantile threshold:
$$\mathcal{S}_{\text{pruned}} = \{u_i \in x \mid I(u_i) \ge \tau_\rho\}$$
- **Efficiency:** Delivers **$50\%$ context length reduction**, **$36\%$ KV-cache VRAM reduction**, and **$32\%$ end-to-end latency reduction**.
- **Fidelity:** Preserves task performance with $<0.023$ drop in BERTscore across QA and summarization.
