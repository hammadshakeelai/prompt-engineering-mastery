# Least-to-Most Prompting (Zhou et al., 2022)

1. **Sub-problem Decomposition**: Prompts direct the model to break down a complex target problem into an ordered list of simpler, intermediate subproblems.
2. **Sequential Solving with Accumulated Context**: Subproblems are solved sequentially in ascending complexity. Each question and generated answer are appended to context for subsequent steps.
3. **Compositionality Generalization**: Enables solving problems requiring more steps than seen in exemplars (e.g., 99.7% on SCAN length splits vs. 16.2% standard CoT).