# PromptBreeder (Fernando et al., Google DeepMind, 2023)

**Citation:** arXiv:2309.16797

## Core Concept
Evolutionary algorithm that co-evolves task-prompts AND the mutation-prompts that modify them (self-referential self-improvement).

## Evolutionary Units
1. **Task-Prompt (P):** Guides LLM to solve target task.
2. **Mutation-Prompt (M):** Meta-instruction specifying how to mutate P.
3. **Thinking-Style:** Cognitive heuristic for reasoning patterns.

## Mutation Operators
1. **Direct Mutation:** Zero-shot mutation of P without M.
2. **First-Order:** Applies M to P to yield improved P'.
3. **Hyper-Mutation (Self-Referential):** Mutates the mutation prompts themselves.
4. **Lamarckian Mutation:** Reverse-engineers prompts from correct solution traces.
5. **EDA Shuffling:** Maintains diversity via semantic embeddings.

## Results
Outperforms CoT and Plan-and-Solve on GSM8K, SVAMP, hate speech detection.