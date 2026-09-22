# Program-Aided Language Models (PAL) - Gao et al. (ICML 2023)

**Citation:** arXiv:2211.10435

## Core Concept
Decouples reasoning from computation:
- **Reasoning (LLM):** Translates problem into Python code with natural language comments.
- **Computation (Interpreter):** Deterministic Python runtime executes the code.

## Mechanism
1. Few-shot exemplars show Python code as intermediate reasoning steps.
2. Generated code executed in Python runtime.
3. Runtime variables preserve intermediate states without hallucination.

## Results
- SOTA on GSM8K using Codex, surpassing PaLM-540B.
- Tested across 13 tasks: math, symbolic, algorithmic reasoning.
- Code generation pretraining generalizes to non-coding reasoning tasks.