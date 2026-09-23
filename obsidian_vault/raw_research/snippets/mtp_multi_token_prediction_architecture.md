# Multi-Token Prediction (MTP) Speculative Architecture (DeepSeek-V3 / Gloeckle et al., Meta 2024)

## 1. Limitations of Next-Token Prediction
Standard next-token cross-entropy $\mathcal{L}_{\text{NTP}} = -\sum \log P(x_i \mid x_{<i})$ promotes greedy token optimization without forcing representations to plan multi-token syntax or long-horizon dependencies, while capping autoregressive decoding throughput to 1 token per forward pass.

## 2. DeepSeek-V3 MTP Module Design
DeepSeek-V3 cascades $D$ sequential MTP modules ($D=1$ in primary production):
1. **Feature Projection & Fusion:** At step $i$, projects trunk hidden state $h_i^{(L)}$ concatenated with embedding of token $x_{i+1}$:
   $$u_i^{(1)} = \operatorname{RMSNorm}\left( W_{\text{proj}} \left[ h_i^{(L)} \,\|\, \operatorname{Embedding}(x_{i+1}) \right] \right)$$
2. **Dedicated Transformer Block:** Passes $u_i^{(1)}$ through an auxiliary transformer block (MLA + MoE FFN).
3. **Shared Unembedding Projection:** Computes logits using the main model's unembedding matrix $W_U$:
   $$P_{\text{MTP}}(x_{i+2} \mid x_{\le i}) = \operatorname{Softmax}(W_U h_i^{(1)})$$
4. **Training Objective:**
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NTP}} + \lambda \mathcal{L}_{\text{MTP}}, \quad \lambda = 0.3$$

## 3. Integrated Speculative Decoding Mechanics
Inference repurposes the MTP head as an internal speculative draft generator:
- Emits token $x_{t+1}$ from the main trunk and candidate token $\tilde{x}_{t+2}$ from MTP concurrently.
- In the subsequent forward step, executes dual-token verification. If verified, emits both tokens and drafts $\tilde{x}_{t+3}$.
- Achieves **$1.8\times$ decoding speedup** with zero external draft model hosting overhead and $<2\%$ parameter addition.
