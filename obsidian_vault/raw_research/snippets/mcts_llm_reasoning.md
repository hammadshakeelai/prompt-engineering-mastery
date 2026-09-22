# Monte Carlo Tree Search (MCTS) for LLM Reasoning

- **Search Topology**: Structures test-time reasoning into a search tree balancing exploration and exploitation (UCT).
- **Core Loop**: Selection -> Expansion (candidate reasoning branches) -> Evaluation (PRM or rollout simulation) -> Backpropagation.
- **Lookahead**: Enables systematic error backtracking and lookahead verification for complex mathematics and code generation.