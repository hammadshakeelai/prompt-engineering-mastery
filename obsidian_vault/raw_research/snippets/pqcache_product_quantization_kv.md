# PQCache: Product Quantization for KV Cache (Zhang et al., 2024)

## 1. The Long-Context Attention Bottleneck
In sequences exceeding $10^5\text{--}10^6$ tokens, linear attention scanning over full Key-Value caches becomes severely memory-bandwidth bound, while token eviction algorithms (e.g., SnapKV) permanently discard context.
PQCache (Zhang et al., arXiv:2407.12820) implements a vector-indexed KV cache using **Product Quantization (PQ)** and **Maximum Inner Product Search (MIPS)** to enable sub-linear attention retrieval without discarding tokens.

## 2. Architecture & Algorithmic Mechanics
1. **Key Subspace Decomposition:**
   Key activation vectors $k \in \mathbb{R}^d$ are partitioned into $m$ orthogonal sub-vectors $k = [k^{(1)}, \dots, k^{(m)}]$ and mapped to nearest centroids across learned codebooks $\mathcal{C}_1, \dots, \mathcal{C}_m$. Storing 8-bit cluster indices compresses Key storage by **$8\times\text{--}16\times$**.

2. **Asymmetric Inner Product Tables:**
   During decoding, the query vector $q_t$ computes an $m \times 256$ inner product lookup table $T_s[j] = \langle q_t^{(s)}, c_{s, j} \rangle$ in constant time $\mathcal{O}(m \cdot K)$, completely independent of context length.

3. **Sub-Linear MIPS Retrieval:**
   Attention logits $\langle q_t, k_i \rangle$ are evaluated via table lookups and additions. Top-$\kappa$ tokens ($< 10\%$ of context) are dynamically fetched for full attention evaluation, slashing decoding memory bandwidth by **$70\%\text{--}85\%$**.

## 3. Empirical Performance
- Achieves **$+4.60\%$ higher accuracy** on InfiniteBench compared to token eviction baselines.
- Sustains near constant-time attention decoding latency across $128\text{k}\text{--}1\text{M}$ token sequences.
