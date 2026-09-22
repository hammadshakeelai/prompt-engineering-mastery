# Language Agent Tree Search (LATS) - Zhou et al. (2023)

**Citation:** arXiv:2310.04406

## Overview
Unifies MCTS with LLMs for planning, acting, and reasoning without parameter updates.

## Tri-Role LLM
1. **Agent (Generator):** Proposes candidate actions/thoughts to expand tree branches.
2. **Value Function (Evaluator):** Scores intermediate states via self-evaluative prompting.
3. **Reflector (Optimizer):** Synthesizes verbal self-reflections on trajectory failure.

## MCTS Adaptation
1. **Selection:** UCT balances exploration vs. exploitation of high-reward trajectories.
2. **Expansion:** Samples k diverse actions/thoughts at chosen node.
3. **Evaluation:** Environment rewards + LLM semantic scoring.
4. **Backpropagation:** Propagates values/visit counts; caches verbal reflections.

## Results
- HumanEval: 94.4% pass@1 with GPT-4 (state-of-the-art)
- Also SOTA on WebShop and HotPotQA.