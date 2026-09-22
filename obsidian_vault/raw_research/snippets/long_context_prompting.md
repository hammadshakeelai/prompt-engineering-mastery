# Long Context Prompting Techniques (2024-2025)

## 1. Positional Encoding Extrapolation
- RoPE: Rotary position embeddings via rotation matrices. NTK-aware base-frequency scaling needed for extrapolation.
- YaRN: Interpolates high-frequency RoPE features, extrapolates low-frequency. Enables 128k+ context with <0.1% pre-training data.
- ALiBi: Static linear distance penalty in attention logits (-m * |i-j|). Zero-shot extrapolation but degrades beyond 64k vs. scaled RoPE.

## 2. API Context Caching
- Reuses server-side KV attention states across calls.
- Cuts prefill latency (TTFT) by 85%, input costs by 50-90%.
- Requires byte-identical prompt prefixes. Immutable context at START, dynamic user turns at END.

## 3. Structuring 100k+ Token Prompts
- Prompt Sandwiching: Core directives first, retrieved docs in middle, restate query + output constraints at end.
- Hierarchical Delimiters: XML tags prevent attention drift and prompt injection.
- Explicit Scratchpads: Instruct model to cite doc IDs in <thinking> before generating answers.