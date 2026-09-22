# Multi-Head Latent Attention & Matrix Absorption (DeepSeek 2024–2026)

## 1. Latent KV Compression Bottleneck
Serving long-context models is bounded by GPU HBM memory bandwidth. While Multi-Head Attention (MHA) caches separate Key and Value vectors for all $n_h$ heads, **Multi-Head Latent Attention (MLA)** compresses the Key and Value space into a single low-rank latent vector:
$$c_t^{KV} = W^{DKV} h_t \quad (c_t^{KV} \in \mathbb{R}^{d_c})$$
In DeepSeek-V3, $d_c = 512$, compressing the KV dimension by over $93\%$.

## 2. Decoupled RoPE Key
Because non-linear Rotary Positional Embeddings destroy matrix associativity:
$$R_{\Theta, t} (W^{UK} c_t^{KV}) \ne W^{UK} (R_{\Theta, t} c_t^{KV})$$
MLA factors positional information into a small decoupled key head:
$$k_t^R = \text{RoPE}(W^{KR} h_t) \quad (k_t^R \in \mathbb{R}^{d_R}, d_R = 64)$$
The KV cache per token per layer requires only storing $[c_t^{KV}, k_t^R]$ ($576$ scalars vs. $16,384$ in MHA).

## 3. Zero-Materialization Matrix Absorption
During generation, MLA bypasses key/value decompression entirely:
1. **Query Absorption:**
   Pre-multiplies up-projection matrices: $W^{Q\_\text{abs}} = W^{UQ \top} W^{UK}$. The query is projected into latent space $q^{\text{abs}} = c_t^Q W^{Q\_\text{abs}}$ and directly inner-producted with cached latents $c_j^{KV}$. Keys $K$ are never uncompressed into memory.
2. **Value Absorption:**
   Pre-multiplies $W^{O\_\text{abs}} = W^{UV \top} W^O$. Attention weights are summed directly over 512-dimensional $c_j^{KV}$ vectors and multiplied once by $W^{O\_\text{abs}}$. Values $V$ are never materialized in memory.
