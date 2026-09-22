# Structured State Space Models (Mamba) vs. Transformers for Prompting

## 1. Architectural Differences
- Transformers: O(L^2) time/memory via pairwise dot-product attention. Exact lossless KV cache.
- Mamba (Gu and Dao, 2023): Linear SSM (h'_t = A*h_{t-1} + B*x_t, y_t = C*h_t). Selectivity makes B, C, Delta functions of current input. O(L) compute, O(1) inference memory.
- Hardware-aware parallel scan replaces attention's matmul for efficient training.

## 2. Implications for Very Long Sequences
- Throughput: Transformers face quadratic prefill latency; Mamba scales linearly with constant generation latency.
- Memory: Mamba processes 100k-1M+ tokens with drastically reduced VRAM.
- Trade-off: Transformers preserve exact token identities (strong associative recall). Mamba compresses history into fixed-size state (lossy decay on verbatim long-range retrieval).

## 3. Divergent Prompting Strategies
- Recency Bias: Mamba has continuous recurrent state decay (no attention sinks). Place crucial context directly before generation boundary (end of prompt).
- Few-Shot Limits: Diminishing returns for Mamba vs. Transformers with 20+ examples due to state compression. Prefer zero-shot with rich instructions.
- Hybrid Architectures (Jamba, Zamba): Interleave attention layers for exact associative recall + Mamba layers for long-context efficiency.