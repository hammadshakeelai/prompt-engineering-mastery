# Pyramidal Information Funneling & Layer-Adaptive KV Eviction (PyramidKV) (Cai et al., COLM 2024)

## 1. The Uniform Allocation Bottleneck
Conventional dynamic KV cache eviction algorithms (e.g., [[h2o_heavy_hitter_submodular_kv|H2O]], SnapKV, Scissorhands) assign an identical token capacity $C$ to every transformer layer $l \in [1, L]$.
However, attention entropy exhibits a pronounced vertical gradient across network depth:
- **Early Layers (Diffuse Context Integration):** Attention maps are broad and high-entropy, dispersing probability mass over long horizons to capture syntactic dependencies and token associations.
- **Deep Layers (Semantic Funneling):** Attention entropy collapses into extreme sparsity, concentrating almost exclusively on immediate task tokens, entity anchors, and initial [[streaming_llm_sinks|attention sinks]].
Imposing uniform capacity $C$ starves early layers of necessary syntactic memory while squandering high-bandwidth cache on redundant tokens in deeper layers.

## 2. PyramidKV Mechanics & Allocation Schedule
Cai et al. (*PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling*, arXiv:2406.02069 / COLM 2024) introduce layer-adaptive memory scaling:
1. **Pyramidal Budget Distribution:** Total budget $C_{\text{total}}$ is distributed monotonically decreasing across depth $l$:
   $$C_l = C_{\text{min}} + \Delta C \left(\frac{L - l}{L - 1}\right), \quad \sum_{l=1}^L C_l = C_{\text{total}}$$
   Lower layers retain expansive contexts ($3\times\text{--}4\times$ standard allocations), while top layers compress aggressively to minimal sink and local window subsets.
2. **Observation-Window Heavy Hitter Selection:**
   Within each layer's budget $C_l$, tokens are partitioned into:
   - $S_{\text{sink}}$ initial sequence tokens.
   - $W_{\text{local}}$ trailing sliding-window tokens.
   - Top $C_l - S_{\text{sink}} - W_{\text{local}}$ tokens selected by max-pooling attention weights over prompt observation window $W_{\text{obs}}$:
     $$I_l = \operatorname{argTopK}_j \left( \max_{t \in W_{\text{obs}}} A_{l, t, j} \right)$$

## 3. Empirical Results
- **8.3x Memory Reduction:** Maintains $99.2\%$ baseline performance across LongBench and L-Eval on LLaMA-3-8B and Mistral-7B while storing only **12%** of the full KV cache.
- **Extreme Compression Resilience:** In ultra-low cache regimes (retaining just **0.7%** KV cache), PyramidKV retains $>80\%$ retrieval on Needle-in-a-Haystack, whereas uniform baselines collapse completely ($<15\%$).
