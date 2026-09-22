# SGLang RadixAttention & Jump-Forward Decoding

- **RadixAttention**: Organizes cached KV tokens into a radix tree, automatically reusing KV cache across prompts sharing prefixes (system instructions, multi-turn dialogue, few-shot examples) via LRU eviction.
- **Jump-Forward Decoding**: Identifies deterministic tokens in structured grammars (JSON syntax, keys) and jumps forward by emitting parallel token batches, bypassing redundant sequential forward passes.