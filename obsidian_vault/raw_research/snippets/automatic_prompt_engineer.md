# Automatic Prompt Engineer (APE) - Zhou et al. (ICLR 2023)

**Citation:** arXiv:2211.01910 - *Large Language Models Are Human-Level Prompt Engineers*

## Core Paradigm
Frames prompt design as black-box optimization: max_p E[(x,y)][f(p,x,y)]

## Mechanism
1. **Candidate Generation:** LLM conditioned on input-output demos to propose candidate instructions (forward generation, reverse generation, meta-prompting).
2. **Scoring:** Execution accuracy or likelihood scoring: sum log P(y|p,x).
3. **Refinement:** Generates semantic variants around top-performing candidates.

## Key Findings
- Outperformed human heuristics on 19/24 Instruction Induction tasks.
- Discovered zero-shot CoT trigger: *"Let's work this out step by step to be sure we have the right answer."*
- Surpassed Kojima et al.'s "Let's think step by step" on MultiArith (82.0% vs. 78.7%).