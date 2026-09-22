# LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding (Samarin et al., 2026)

## 1. The Proxy Misalignment of KL Divergence
In speculative decoding (Leviathan et al., 2023), tokens drafted by $P_d$ are accepted by $P_t$ with expected probability:
$$\mathbb{E}_{x \sim P_d}[\alpha(x)] = 1 - \frac{1}{2} \|P_t - P_d\|_{\text{TV}}$$
Maximizing speculative acceptance rate is mathematically equivalent to minimizing the **Total Variation (TV) distance**.
- **The Mode-Covering Pathology:** Draft models are conventionally trained by minimizing forward KL divergence:
  $$\mathcal{D}_{\text{KL}}(P_t \parallel P_d) = \sum_{x \in \mathcal{V}} P_t(x) \log \frac{P_t(x)}{P_d(x)}$$
  Forward KL severely penalizes assigning zero mass to any token where $P_t(x) > 0$. Small draft models waste parameter capacity covering negligible tail tokens, starving probability mass from the top-$k$ tokens and depressing acceptance rates.

## 2. LK Loss Formulation & Hybrid Total Variation
Samarin et al. (arXiv:2602.23881, 2026) formulate training objectives that directly maximize the intersection of probability distributions:
1. **Total Variation Surrogate:**
   $$\mathcal{L}_{\text{TV}}(P_d, P_t) = \sum_{x \in \mathcal{V}} \max\left(0, P_t(x) - P_d(x)\right)$$
   Penalizes under-allocating mass on tokens where $P_t(x) > P_d(x)$, without penalizing the draft model for zeroing out irrelevant tail tokens.
2. **Hybrid Formulation (`lk_hybrid`):**
   $$\mathcal{L}_{\text{LK}}(z_d, P_t) = \sum_{x \in \mathcal{V}} \mathbb{I}[P_t(x) \ge \tau] \cdot \max\left(0, P_t(x) - P_d(x)\right)^\gamma + \lambda \mathcal{D}_{\text{KL}}(P_t \parallel P_d)$$
   Stabilizes subgradient backpropagation through logits $z_d$ while concentrating optimization on target mass peaks.

## 3. Systems Impact & Acceptance Length
- **Inference Speedup:** Integrated into SpecForge and vLLM, increases average accepted sequence length $\mathbb{E}[L]$ by **$+8\%\text{--}+10\%$** across 8B–685B target models.
- **Zero Runtime Overhead:** Enhances speculative draft accuracy strictly by altering the distillation loss, adding zero compute or memory cost during generation.
