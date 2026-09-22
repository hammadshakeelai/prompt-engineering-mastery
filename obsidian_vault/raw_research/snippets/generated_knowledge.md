# Generated Knowledge Prompting (Liu et al., ACL 2022)

**Citation:** arXiv:2110.08387

## Two-Stage Architecture
1. **Knowledge Generation (M_gen):** LLM generates K discrete knowledge statements from query x using few-shot demos.
2. **Knowledge Integration (M_eval):** Each statement k_i prepended to question, predictions aggregated via majority vote or probability-weighted ensemble.

## Key Properties
- No external retrieval corpus needed; uses parametric memory.
- M_gen and M_eval can be different model sizes.
- Low-quality knowledge degrades reasoning ("Knowledge Poisoning").

## Performance
- SOTA on CommonsenseQA, CommonsenseQA 2.0, NumerSense, QASC.
- +4-10% over standard few-shot baselines.

## Significance
Architectural bridge between RAG (external retrieval) and CoT (internal deduction).