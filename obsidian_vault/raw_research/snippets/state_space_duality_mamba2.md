# Structured State Space Duality (SSD) & Mamba-2 (Dao & Gu, ICML 2024)

## 1. Unifying Attention and State Space Models
Standard Transformers require $\mathcal{O}(T^2)$ prefill compute and store $\mathcal{O}(T)$ KV caches per layer.
Dao & Gu (ICML 2024) proved that linear Attention and selective SSMs are two representations of the same mathematical object connected via **1-semiseparable matrices**.

## 2. Mathematical Duality: Recurrence vs. Semiseparable Matrix
1. **Recurrent Form:**
   $$h_t = a_t h_{t-1} + B_t x_t, \quad y_t = C_t h_t$$
   Scalar decay $a_t \in (0, 1)$ tracks exponential decay over time.
2. **Matrix Multiplicative Form:**
   $$Y = (M \circ (Q K^\top)) V$$
   where $M$ is a lower-triangular 1-semiseparable matrix:
   $$M_{j, i} = \prod_{k=i+1}^j a_k \quad (j \ge i)$$
   SSMs represent causal linear attention with exponential relative positional masking, eliminating Softmax while preserving causal expressivity.

## 3. Hardware-Aware Computation & $\mathcal{O}(1)$ Decoding
- **Block-Decomposed Prefill:** Partitions sequences into chunks of size $Q=64$. Intra-chunk attention executes via dense Tensor Core matrix multiplications; inter-chunk states transfer via recurrent updates.
- **$\mathcal{O}(1)$ Autoregressive Memory:** Generation runs recurrently:
  $$H_t = a_t H_{t-1} + K_t^\top V_t, \quad Y_t = Q_t H_t$$
  The KV cache is replaced by fixed state $H \in \mathbb{R}^{d \times d}$, enabling **constant memory and latency** across $1\text{M}+$ tokens.
- **Throughput:** Achieves $2\times\text{--}8\times$ speedups over FlashAttention-2 Transformers while matching perplexity.
