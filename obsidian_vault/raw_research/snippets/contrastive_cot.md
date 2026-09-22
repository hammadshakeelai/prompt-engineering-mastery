# Contrastive Chain-of-Thought (CCoT) - Chia et al. (2023)

**Citation:** arXiv:2311.09277

## Core Concept
Pairs valid AND invalid reasoning demonstrations to teach models to avoid logical pitfalls.

## Prompt Structure
- Question: Target reasoning problem
- Explanation (INCORRECT): Plausible but flawed logic to avoid
- Explanation (CORRECT): Valid step-by-step reasoning
- Answer: Ground truth

## Negative Generation Methods
1. **Targeted perturbation:** Mutate numbers, operations, or entities in valid chains.
2. **Model sampling:** Filter model rollouts for incorrect reasoning paths.

## Results
- **+4-13%** over standard CoT on arithmetic (GSM8K, SVAMP) and commonsense (StrategyQA, CSQA).
- Synergistic with self-consistency.