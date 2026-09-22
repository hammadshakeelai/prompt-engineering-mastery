# KIVI & ThinK: Sub-Token Channel Pruning and Asymmetric 2-Bit Quantization (Liu et al. ICML 2024; ThinK 2024)

## 1. Limitations of Coarse Whole-Token Eviction
Whole-token KV eviction methods (e.g., [[h2o_heavy_hitter_submodular_kv|H2O]], SnapKV) permanently drop entire key-value representations when memory constraints bind.
- **The Information Destruction Trap:** In retrieval-augmented tasks, code synthesis, or numeric reasoning, evicting even a single token (e.g., a delimiter, register, or scalar digit) catastrophically breaks needle-in-a-haystack recall.
- **Sub-Token Retention Principle:** Retain 100% of sequence tokens by compressing along bit-width ($b$) and channel dimensions ($d$) rather than sequence length ($T$).

## 2. KIVI: Asymmetric 2-Bit Quantization Geometry
Liu et al. (ICML 2024) discovered fundamentally divergent outlier structures between Key and Value tensors in autoregressive transformers:
1. **Per-Channel Key Quantization:**
   Keys display cross-timestep **channel outliers** where specific coordinate indices exhibit extreme magnitudes persistently across time $T$. Quantizing along tokens washes out these channels. KIVI groups Keys per-channel across time:
   $$\tilde{K}_{c, t} = \text{round}\left( \frac{K_{c, t} - \min_{t'} K_{c, t'}}{\Delta_c} \right), \quad \Delta_c = \frac{\max_{t'} K_{c, t'} - \min_{t'} K_{c, t'}}{2^b - 1}$$
2. **Per-Token Value Quantization:**
   Values exhibit **token-level outliers** (high norm at specific generation steps), while individual channels within a token remain bounded. KIVI quantizes Values per-token across channel dimension $D$:
   $$\tilde{V}_{t, c} = \text{round}\left( \frac{V_{t, c} - \min_{c'} V_{t, c'}}{\Delta_t} \right), \quad \Delta_t = \frac{\max_{c'} V_{t, c'} - \min_{c'} V_{t, c'}}{2^b - 1}$$
3. **Efficiency:** Tuning-free true 2-bit quantization ($b=2$, effective $\sim 2.6$ bits with scale/bias) yielding up to $3.47\times$ peak throughput increase with $<0.1$ perplexity drop.

## 3. ThinK: Query-Driven Key Channel Pruning
ThinK (2024) demonstrates that $>30\%$ of Key channels contribute negligible variance to the dot-product attention score $q^\top k$.
- Runtime query-directed channel pruning removes low-utility Key channels on the fly, reducing Key cache memory by up to $50\%$.
- Combined with KIVI 2-bit compression, this achieves a $2.8\times$ compounding memory reduction over standard 2-bit baselines, unlocking million-token context windows on single consumer GPUs.
