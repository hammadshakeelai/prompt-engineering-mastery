# Infini-Attention: Bounded Memory Transformers (Munkhdalai et al., Google 2024)

## 1. The Quadratic KV Memory Ceiling
Standard scaled dot-product attention scales memory quadratically $\mathcal{O}(N^2)$ and requires storing all past Key-Value activations in GPU HBM, exhausting memory at million-token scale.

## 2. Compressive Memory Integration
Infini-attention (Munkhdalai et al., Google / arXiv:2404.07143) integrates compressive memory directly into masked dot-product attention:
1. **Compressive Memory Retrieval:**
   $$A_{\text{mem}} = \frac{\sigma(Q) M_{t-1}}{\sigma(Q) z_{t-1} + \epsilon}, \quad \sigma(x) = \operatorname{ELU}(x) + 1$$
2. **Compressive Memory Update:**
   $$M_t = M_{t-1} + \sigma(K)^\top V, \quad z_t = z_{t-1} + \sum_{s=1}^S \sigma(K_s)^\top$$
3. **Adaptive Sigmoid Gating:**
   $$A_{\text{total}} = \operatorname{sigmoid}(\beta) \odot A_{\text{mem}} + (1 - \operatorname{sigmoid}(\beta)) \odot A_{\text{dot}}$$
   where $A_{\text{dot}}$ is standard local masked dot-product attention.

## 3. Complexity & Scalability
Maintains a constant $\mathcal{O}(1)$ memory footprint across infinite contexts. Successfully performs $100\%$ passkey retrieval on **1,000,000 token sequences** with **$114\times$ memory reduction** over FlashAttention.
