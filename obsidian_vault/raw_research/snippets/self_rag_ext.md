# Self-RAG: Self-Reflective Retrieval-Augmented Generation

Enhances factuality by training an LLM to output special reflection tokens:
- `[Retrieve]`: Dynamically decides whether external knowledge retrieval is necessary (`yes`, `no`, `continue`).
- `[IsRel]`: Evaluates whether a retrieved passage contains relevant context to address the prompt.
- `[Critique]` (`[IsSup]`, `[IsUse]`): Assesses generated responses. `[IsSup]` verifies whether claims are supported by evidence; `[IsUse]` rates overall usefulness.

Reflection token probabilities guide beam search during inference, filtering irrelevant docs and selecting the most grounded text.