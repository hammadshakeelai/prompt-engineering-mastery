# Context Extrapolation: RoPE vs. YaRN vs. ALiBi

- **RoPE**: Encodes position by rotating Q and K vectors in 2D subspaces. Struggles on unseen lengths due to out-of-distribution rotation angles.
- **ALiBi**: Discards vector position embeddings; injects linear distance penalties ($-m \cdot |i - j|$) directly into attention logits, providing zero-shot length extrapolation.
- **YaRN**: Non-uniform frequency scaling for RoPE: interpolates low frequencies, keeps high frequencies unscaled, and rescales attention logits to prevent entropy dilution.