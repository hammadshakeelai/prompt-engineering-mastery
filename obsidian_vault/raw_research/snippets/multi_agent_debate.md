# Multi-Agent Debate and Society of Mind Prompting

Citations: Du et al. (2023, arXiv:2305.14325); Chan et al. ChatEval (2023, arXiv:2308.07201)

## Du et al. (2023) Framework
Inspired by Marvin Minsky's Society of Mind. Multiple independent LLM instances:
1. Independently generate initial reasoning trajectories and solutions.
2. Exchange and critique each other's outputs over multiple rounds.
3. Iteratively incorporate peer arguments to correct logical fallacies and reach consensus.

## Key Architectures
- ChatEval: Multi-agent referee team with diverse personas to evaluate text quality. One-on-one or simultaneous-talk topologies.
- Ensemble Refinement: Multiple reasoning paths pooled as student reasoning; LLM iteratively refines and synthesizes a final response.

## When Debate Helps
- Complex multi-step reasoning (math, code, logic) where intermediate validation is required.
- Mitigating hallucinations through cross-examination and diverse perspectives.

## When Debate Hurts
- Agreement Bias: Models converge prematurely on plausible but incorrect peer solutions.
- Debate Hacking: Agents optimize for rhetoric over truthfulness; overconfident agents dominate.
- Efficiency: Exponential token/latency overhead often fails to outperform simple self-consistency.