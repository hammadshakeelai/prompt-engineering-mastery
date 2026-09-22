# Medusa: Draft-Free Multi-Head Speculative Decoding

- **Multi-Head Drafting**: Appends multiple lightweight MLP heads to the target LLM's final hidden state, predicting future tokens (t+1, t+2) concurrently without a draft model.
- **Tree Verification**: Candidates structured into a tree verified in a single forward pass via tree-structured attention, emitting multiple tokens per step for 2x-3x speedup.