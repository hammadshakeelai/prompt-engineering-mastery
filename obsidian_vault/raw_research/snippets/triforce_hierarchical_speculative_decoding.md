# TriForce: Hierarchical Speculative Decoding for Long Context (Sun et al., COLM 2024)

## 1. Long-Context Speculation Bottlenecks
In long sequences ($>32\text{k}\text{--}128\text{k}$), standard speculative decoding fails because loading the full 128k KV cache on every verification step saturates GPU memory bandwidth, collapsing acceleration.

## 2. Three-Tier Speculative Hierarchy
TriForce (Sun et al., CMU & Meta / COLM 2024 / arXiv:2404.11912) introduces a three-level speculative pipeline:
1. **Tier 1 (Streaming Draft Model):** Small base model (Llama-68M) using StreamingLLM cache eviction ($4$ sinks + $1024$ recent tokens) proposing candidate tokens in $\mathcal{O}(1)$ time.
2. **Tier 2 (Target Model with Retrieval-Augmented Sparse KV):** Target LLM with dynamic Key-Query retrieval cache filtering Tier 1 proposals.
3. **Tier 3 (Lossless Full KV Verification):** Exact target model parallel verification via standard speculative rejection sampling ($\alpha = \min(1, P_t / P_d)$).

## 3. Empirical Acceleration
Delivers up to **$4.86\times$ wall-clock speedup** on Llama2-7B-128K on a single consumer RTX 4090 GPU while cutting KV-cache memory traffic by $>75\%$.
