# Multi-Head Latent Attention (MLA) & Decoupled RoPE

## 1. The Core Innovation of MLA (DeepSeek-V2/V3/R1)
In large-scale autoregressive serving, Multi-Head Attention (MHA) consumes prohibitive amounts of KV cache memory ($2 \times n_h \times d_h = 32,768$ values per token per layer), while Grouped-Query Attention (GQA) degrades expressive retrieval capacity. **Multi-Head Latent Attention (MLA)** compresses the Key-Value cache into a shared **low-rank latent vector** $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ ($d_c = 512$), achieving a **$56.8\times$ reduction vs MHA** and **$3.56\times$ reduction vs GQA-8** while exceeding MHA expressive accuracy.

```mermaid
flowchart LR
    Input["Input h_t"] --> DownKV["W^DKV: Low-Rank Latent c_t^KV (d_c=512)"]
    Input --> DecoupledRoPE["W^KR: Decoupled RoPE Key k_t^R (d_R=64)"]
    DownKV & DecoupledRoPE --> Cache["Cached KV: [c_t^KV, k_t^R] (576 elements = 1.15 KB/tok)"]
    Cache --> KernelAbsorption["Inference: Absorb W^UK into Query & W^UV into W_O (Zero Decompression)"]
```

## 2. Decoupled RoPE & Inference Weight Absorption
- **Decoupled RoPE**: Because Rotary Position Embeddings are non-commutative with low-rank projection matrices, RoPE cannot be directly absorbed if applied to compressed latents. MLA cleanly decouples positional keys ($\mathbf{k}_t^R \in \mathbb{R}^{64}$) from content latents ($\mathbf{c}_t^{KV} \in \mathbb{R}^{512}$), storing only $512 + 64 = 576$ scalars per token.
- **Inference Kernel Absorption**: During decoding, the up-projection matrices $W_i^{UK}$ and $W_i^{UV}$ are mathematically absorbed into the query vector $(\tilde{\mathbf{q}}_i = (W_i^{UK})^T \mathbf{q}_i)$ and output projection matrix $W_O$, eliminating the need to decompress multi-head keys and values in GPU memory.
