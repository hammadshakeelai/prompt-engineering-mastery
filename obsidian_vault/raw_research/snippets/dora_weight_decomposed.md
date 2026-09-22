# DoRA: Weight-Decomposed Low-Rank Adaptation

- **Disentangling Magnitude & Direction**: Decomposes weights into magnitude $m$ and directional matrix $V$:
  $$W = m \frac{V}{\|V\|_c}$$
- **Optimization**: Freezes base weights, applies LoRA to update direction $V = W_0 + BA$, and trains magnitude $m$ independently, matching full fine-tuning dynamics with zero inference overhead.