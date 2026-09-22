# YOCO: You Only Cache Once & Decoder-Decoder Architectures (Sun et al., Microsoft 2024)

## 1. The $L$-Layer KV Cache Bottleneck
In standard autoregressive Transformers, memory for caching Key-Value tensors scales linearly with depth $L$:
$$\mathcal{M}_{\text{KV}} = 2 \cdot B \cdot L \cdot T \cdot d$$
At long context lengths ($T \ge 128\text{k}$), retrieving $L$ distinct key-value matrices from GPU HBM saturates memory bandwidth, bottlenecking autoregressive token throughput.

## 2. Asymmetric Decoder-Decoder Topology
Sun et al. (Microsoft Research, 2024) split the model into two cascaded stages:
1. **Self-Decoder ($L/2$ Layers):**
   - Applies efficient sliding-window local self-attention with window size $W \ll T$.
   - Cache memory is strictly local ($\mathcal{O}(W)$), evicting tokens outside window $W$ dynamically.
2. **Global KV Interface:**
   - At the transition layer, a single global Key-Value pair is projected:
     $$K_{\text{global}} = X_{\text{mid}} W_K, \quad V_{\text{global}} = X_{\text{mid}} W_V$$
3. **Cross-Decoder ($L/2$ Layers):**
   - All subsequent upper layers share the identical global $(K_{\text{global}}, V_{\text{global}})$ via causal cross-attention:
     $$\text{Attn}_l(Q_l) = \text{Softmax}\left(\frac{Q_l K_{\text{global}}^\top}{\sqrt{d}} + M\right) V_{\text{global}}$$
   - Cross-decoder layers allocate **zero** private KV cache memory.

## 3. Systems Performance & Needle Retrieval
- **Memory Compression:** Reduces global KV cache storage by $16\times\text{--}40\times$, maintaining constant serving memory across context windows up to $1\text{,000,000}$ tokens.
- **Prefill Early-Exit:** Global cache generation finishes at layer $L/2$, doubling prefilling throughput.
- **Accuracy:** Achieves $100\%$ retrieval accuracy on needle-in-a-haystack tasks, matching standard full-cache Transformer quality.
