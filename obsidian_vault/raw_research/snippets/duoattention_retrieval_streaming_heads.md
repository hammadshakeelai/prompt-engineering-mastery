# DuoAttention: Retrieval vs. Streaming Heads & Dual KV Caching (Xiao et al., MIT / ICLR 2025)

## 1. Functional Specialization in Long-Context Heads
Conventional inference frameworks allocate full Key-Value memory buffers across all $H$ attention heads.
Xiao et al. (*DuoAttention*, MIT / ICLR 2025) demonstrate that heads naturally bifurcate into two distinct behavioral regimes during long-context processing:
- **Retrieval Heads ($\sim 25\%$):** Maintain expansive receptive fields, actively fetching dispersed tokens and factual needles across long contexts ($>32\text{k}\text{--}1\text{M}$ tokens).
- **Streaming Heads ($\sim 75\%$):** Receptive fields decay sharply outside of initial [[streaming_llm_sinks|attention sinks]] ($x_{1:4}$) and recent sliding windows ($W_{\text{local}} \approx 512$). Storing historical middle tokens for these heads wastes memory with zero utility.

## 2. Dual-Cache Architecture & Kernel Co-Design
DuoAttention optimizes head-level allocations offline:
1. **Automated Head Classification:**
   Learns a binary head assignment vector $\mathbf{m} \in \{0, 1\}^{L \times H}$ via continuous relaxation on validation loss:
   $$m_{l, h} = \begin{cases} 1 & \text{Retrieval Head (Allocated Full Cache)} \\ 0 & \text{Streaming Head (Allocated Constant Buffer)} \end{cases}$$
2. **Dual-Cache Allocation:**
   - **Retrieval Heads:** $\text{Cache}_{\text{retrieval}} \in \mathbb{R}^{T \times d_{\text{head}}}$.
   - **Streaming Heads:** Rolling circular buffer $\text{Cache}_{\text{streaming}} \in \mathbb{R}^{(S_{\text{sink}} + W_{\text{local}}) \times d_{\text{head}}}$, independent of sequence length $T$.
3. **Dense GPU Kernel Optimization:**
   Unlike token-eviction algorithms ([[h2o_heavy_hitter_submodular_kv|H2O]], [[pyramidkv_adaptive_compression|PyramidKV]]) that scatter-gather non-contiguous memory, head partitioning preserves dense contiguous tensor layouts, executing directly within optimized FlashAttention kernels without index-gathering overhead.

## 3. Empirical Scaling
- **$2.55\times$ Memory Reduction:** Halves KV memory footprint on MHA models and delivers $1.67\times$ reduction on GQA with $0\%$ accuracy loss on Needle-in-a-Haystack and LongBench.
- **Extreme Lengths:** Paired with 4-bit quantization, allows LLaMA-3-8B to serve **3.3 million tokens on a single 80GB A100 GPU**.
