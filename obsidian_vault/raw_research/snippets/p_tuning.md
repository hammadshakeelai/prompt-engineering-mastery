# Soft Prompts and P-Tuning (Liu et al.)

## Discrete vs. Continuous Soft Prompts
Discrete prompts constrained to vocabulary tokens, highly sensitive to phrasing.
Soft prompts: virtual continuous embedding vectors optimized directly via gradient descent.
Unconstrained by dictionary, discover latent representations tailored to downstream objectives.

## P-Tuning v1 (Liu et al., 2021 - "GPT Understands, Too")
- Injects continuous virtual prompt tokens into input embedding layer.
- Uses BiLSTM + MLP prompt encoder to capture dependencies and stabilize optimization.
- Primarily boosts NLU and few-shot knowledge probing on autoregressive models.

## P-Tuning v2 (Liu et al., 2022 - Deep Prompt Tuning)
- Inserts learnable prefix vectors into Key and Value matrices at EVERY self-attention layer.
- Deep injection dramatically increases representational capacity.
- Matches full fine-tuning performance across model scales 0.3B-100B+ and complex tasks (NER).

## Computational Advantages
- Trains only 0.1-3% task-specific parameters; freezes >99% of backbone.
- Single frozen model serves hundreds of tasks by swapping lightweight prefix weights.
- Pre-computed prefix embeddings cacheable in KV-cache for rapid inference.