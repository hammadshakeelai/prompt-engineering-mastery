# Decomposed Prompting (DecomP) - Khot et al. (ICLR 2023)

Citation: arXiv:2210.02406

## Core Concept
Solves complex reasoning by decomposing into structured sub-tasks orchestrated by a central decomposer.
Generates a program-like execution trace where sub-task inputs/outputs are passed as variables.

## Sub-Task Handlers
- Specialized handlers: distinct few-shot LLM prompts, fine-tuned models, or symbolic tools/APIs.
- Modularity: Handlers independently optimized, debugged, or replaced without altering the main architecture.

## Recursive Decomposition
- When a handler encounters inputs exceeding capability, it recursively invokes the decomposer.
- Base case reduction: Hierarchical sub-problems unfold until manageable.

## DecomP vs. Least-to-Most Prompting
1. Architecture: Least-to-Most uses ONE LLM throughout. DecomP routes sub-tasks to HETEROGENEOUS handlers.
2. Control Flow: Least-to-Most is strictly flat and linear. DecomP supports non-linear control flow, variable binding, recursive trees.