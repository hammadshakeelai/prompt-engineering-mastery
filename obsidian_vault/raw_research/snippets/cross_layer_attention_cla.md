# Cross-Layer Attention (CLA): Inter-Layer KV Cache Sharing (Brandon et al., NeurIPS 2024)

## 1. Beyond Head-Dimension Compression (MQA/GQA)
Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) reduce KV cache footprint by reducing the number of attention heads.
- **The Unaddressed Layer Dimension:** Even with MQA ($H_{\text{KV}} = 1$), an $L$-layer model stores $L$ distinct Key-Value matrices across sequence length $T$:
  $$\mathcal{M}_{\text{KV}} = 2 \cdot B \cdot L \cdot T \cdot d_{\text{head}}$$
- Storing and retrieving $L$ separate caches per token from GPU HBM dominates autoregressive generation latency.

## 2. Cross-Layer Attention Mechanism
Brandon et al. (MIT / NeurIPS 2024) group adjacent transformer layers into sharing blocks of size $S=2$:
$$\mathcal{G}_k = \{ 2k-1, 2k \}$$
1. **Shared KV Projections:**
   Only the first layer in each block computes and stores Key and Value tensors:
   $$K_{\mathcal{G}_k} = x_{2k-1} W_K^{(k)}, \quad V_{\mathcal{G}_k} = x_{2k-1} W_V^{(k)}$$
2. **Independent Query Execution:**
   Both layers compute private query projections ($Q_{2k-1}, Q_{2k}$) and maintain independent MLP blocks, but attend to the shared $(K_{\mathcal{G}_k}, V_{\mathcal{G}_k})$:
   $$\text{Attn}_l(Q_l) = \text{Softmax}\left( \frac{Q_l K_{\mathcal{G}_k}^\top}{\sqrt{d}} \right) V_{\mathcal{G}_k}$$

## 3. Systems Impact & Empirical Performance
- **$2\times$ KV Cache Halving:** Delivers an additional $2\times$ memory reduction beyond MQA, and up to $16\times$ beyond standard MHA.
- **Perplexity Invariance:** Validated on 1B–3B models trained across 1T+ tokens: CLA-2 matches standard MQA perplexity within $<0.02$ cross-entropy loss.
- **Zero Architectural Disruptions:** Unlike bipartite models ([[yoco_you_only_cache_once|YOCO]]), CLA preserves standard transformer feedforward layers and pipeline parallelization schemes.
