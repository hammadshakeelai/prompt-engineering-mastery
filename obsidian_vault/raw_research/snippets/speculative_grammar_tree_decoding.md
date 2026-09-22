# Grammar-Pruned Speculative Tree Verification & Adaptive Constraints (2025–2026)

## 1. Tree-Structured Speculative Drafting
Linear speculative decoding evaluates a single token sequence, which fails when structured schemas feature branching options (enums, optional fields).
- **Candidate Trees:** Draft models/heads emit a tree $\mathcal{T}$ of candidate token branches.
- **Ancestor Attention Masking:** Target verification checks all tree nodes in one forward pass via an ancestor-causal mask:
  $$M_{i, j} = 0 \text{ if } j \in \text{Ancestors}(i) \text{ else } -\infty$$
  allowing the target model to verify dozens of branching hypotheses in parallel.

## 2. Zero-Compute Pre-Verification Branch Pruning
Modern engines (XGrammar / SGLang co-design) evaluate draft trees against compiled bitmask automata before GPU verification:
- **Pruning Inadmissible Subtrees:** Any candidate node that violates schema syntax is evicted prior to verification, eliminating wasted target model compute.
- **Leaf-to-Root Traversal:** Traverses from valid leaves backward, retaining valid suffix sequences that top-down pruning would prematurely reject.

## 3. Adaptive Constraint Propagation (MetaJuLS, 2025)
Addresses context-sensitive constraints (cross-field dependencies where `field_B` depends on `field_A`'s emitted value):
- Dynamically swaps and updates sub-grammars at runtime based on preceding token outputs.
- Guarantees relational consistency across complex nested schemas natively during inference without retry loops.
