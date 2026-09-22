# Algorithm of Thoughts (AoT) - Sel et al. (ICML 2024)

**Citation:** arXiv:2308.10379

## Core Motivation
ToT achieves deliberate search but requires 100+ API calls. AoT internalizes algorithmic search into a single LLM context window.

## Mechanism
- **In-Context Search Exemplars:** Few-shot demos show DFS/BFS: propose candidates, evaluate states, prune branches, backtrack.
- **Hybrid Intuition-Search:** LLM fuses semantic prior with search heuristics, pruning suboptimal subtrees early.
- **Single-Context:** Transformer's working memory maintains state across exploration steps.

## Results (vs. ToT on Game of 24)
- Matches ToT performance.
- Reduces query count by 1-2 orders of magnitude (100+ → 1-2 queries).
- Also tested on Mini Crosswords and TSP.