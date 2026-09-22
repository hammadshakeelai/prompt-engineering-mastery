# Self-Refine (Madaan et al., NeurIPS 2023)

**Citation:** arXiv:2303.17651

## Architecture
Closed loop with three modular roles, all executed by the SAME LLM:
1. **Generate:** Initial candidate output from input prompt.
2. **Feedback:** Multi-aspect natural language critique (code efficiency, readability, logic).
3. **Refine:** Rewrite conditioned on prompt + previous draft + feedback.
Repeats for max k iterations (typically k=2-4).

## Results
- Evaluated on 7 tasks: dialogue, code optimization, code readability, constrained gen, sentiment reversal, GSM8K, explanations.
- ~20% absolute improvement across tasks.
- No external verifier needed.

## Limitations
- Self-feedback can hallucinate non-existent issues.
- Quality plateaus or degrades after 2-3 rounds.