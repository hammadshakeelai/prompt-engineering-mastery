# G-Eval Framework: NLG Evaluation with LLM-as-a-Judge (Liu et al., 2023)

- **Form-Filling Paradigm**: Structures evaluation as filling out a rubric containing task descriptions, input context, candidate text, and explicit evaluation criteria.
- **Chain-of-Thought (CoT)**: Prompts the LLM to generate explicit evaluation steps and reason sequentially through criteria prior to scoring.
- **Probability-Weighted Scoring**: Calculates expected score using output token probabilities of score integers (1-5) rather than greedy token numbers, yielding continuous fine-grained ratings and reducing scoring bias.