# KV Cache and Flash Attention for LLM Efficiency

## 1. KV Cache Mechanics
Past tokens Key/Value states cached in GPU memory, eliminating redundant O(N^2) prefix recomputation.
Cache size = 2 x b x s x l x h x d, dominating VRAM in long contexts.

## 2. Cache Eviction
- StreamingLLM: Retains attention sink initial tokens + rolling local window for infinite-length generation.
- H2O (Heavy Hitter Oracle): Preserves top cumulative-attention-score tokens. ~80% memory reduction with negligible degradation.

## 3. Flash Attention 2/3
Computes attention in SRAM via tiling + online softmax, avoiding O(N^2) HBM materializations.
- FA2 (A100): 50-73% peak FLOP utilization, 2x faster than FA1.
- FA3 (H100): Async TMA, FP8, warp specialization. 1.5-2x over FA2 (~75-85% utilization).

## 4. Prompt Caching APIs
- Anthropic: Explicit cache_control:ephemeral marker. Min 1024 tokens. 90% cost reduction, 80% TTFT latency cut.
- OpenAI: Automatic caching for prefixes >= 1024 tokens. 50% discount on cached tokens.