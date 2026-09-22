# EAGLE: Feature-Level Speculative Decoding

- **Feature-Space Speculation**: Uses a single-layer Transformer decoder to autoregressively predict second-to-top hidden features rather than discrete tokens.
- **Mitigating Compounding Error**: Continuous feature prediction reduces distribution shift, boosting draft acceptance rates (>80%) and yielding 2-3x lossless wall-clock acceleration.