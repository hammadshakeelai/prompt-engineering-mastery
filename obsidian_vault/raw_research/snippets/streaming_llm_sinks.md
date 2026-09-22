# StreamingLLM: Attention Sinks & Sliding Windows

- **Attention Sinks**: LLMs dedicate disproportionately large attention weights to initial tokens (token 0) to absorb excess attention mass, regardless of semantic content.
- **Mechanism**: Preserves initial attention sink tokens (first 4 tokens) alongside the sliding window of latest tokens in KV cache.
- **Impact**: Restores softmax stability, avoids perplexity spikes, and enables infinite-length text streaming with fixed memory footprint without fine-tuning.