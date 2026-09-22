# PagedAttention & vLLM Prefix Caching

- **Virtual Memory Architecture**: Partitions KV cache into non-contiguous fixed-size blocks (pages), eliminating external fragmentation.
- **Prefix Caching**: Identical prefix tokens (system prompts, few-shot examples) map to shared physical memory blocks through block tables and reference counting.
- **Copy-on-Write**: Requests reuse existing physical pages, allocating new pages only when tokens diverge during generation, saving TTFT latency and memory.