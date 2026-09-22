# LLM Agent Memory Taxonomy (CoALA Framework)

Draws from cognitive psychology to organize agent memory into four functional tiers.

## 1. Working Memory (Short-Term Workspace)
- Cognitive Basis: Baddeley and Hitch's Working Memory Model (1974) - active workspace for information under cognitive load.
- Implementation: LLM context window and runtime scratchpads. Holds conversational history, active CoT reasoning steps, sub-goals, tool call inputs/outputs.

## 2. Episodic Memory (Autobiographical Experience)
- Cognitive Basis: Endel Tulving (1972) - explicit memory of specific past events and agent-user interactions.
- Implementation: Vector Databases (Chroma, Qdrant) with dense embeddings + temporal metadata. Retrieved via k-NN semantic similarity, recency decay, importance scoring.

## 3. Semantic Memory (Factual and Conceptual Knowledge)
- Cognitive Basis: Tulving's declarative taxonomy - generalized world knowledge decoupled from specific episodes.
- Implementation: Knowledge Graphs, Graph RAG, relational entity stores. Deterministic entity-relation traversal and structured fact retrieval.

## 4. Procedural Memory (Implicit Skills and Policies)
- Cognitive Basis: Larry Squire's non-declarative memory (1987) - automated how-to knowledge.
- Implementation: Encoded into model weights via SFT/LoRA, RLHF/DPO. Operationalized via system prompt instructions, few-shot tool exemplars, deterministic execution workflows.