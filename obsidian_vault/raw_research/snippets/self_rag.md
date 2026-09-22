# Self-RAG: Self-Reflective Retrieval-Augmented Generation (Asai et al., 2023)

## Reflection Tokens
1. **[Retrieve] / [No Retrieve]:** Model learns when retrieval is actually needed.
2. **[IsREL]:** Is retrieved passage relevant?
3. **[IsSUP]:** Is generation grounded/supported by evidence?
4. **[IsUSE]:** Is the answer useful?

## Training Pipeline
- **Critic Model:** Trained on GPT-4-annotated corpus to insert reflection tokens.
- **Generator:** Fine-tuned to jointly predict text tokens and reflection tokens in one autoregressive pass.

## Inference
- Segment-level beam search across multiple retrieved passages.
- Reflection token probabilities score branches.
- Test-time controllability: tune thresholds without retraining.