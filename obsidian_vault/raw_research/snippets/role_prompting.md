# Role Prompting: Expert Personas

## Mechanism
Conditions LLM with expert persona (e.g., "You are an expert bioinformatician"). Steers generation toward stylistic/lexical subspaces from training but does NOT inject new parametric knowledge.

## Empirical Gains
- **Reasoning Trigger:** Kong et al. (NAACL 2024): AQuA 53.5% → 63.8%; Last Letter 23.8% → 84.2%.
- **Stylistic Alignment:** Significantly improves tone, domain terminology, audience adaptation.
- **Rule-Based Personas:** Integrating operational guidelines improves multi-turn agent consistency.

## Empirical Limits
- **Knowledge Benchmarks:** Generic personas ("world-class physicist") yield negligible MMLU/GSM8K gains.
- **Hallucination Risk:** Models optimize for persona conviction over truthfulness.
- **Reasoning Interference:** Mismatched personas can reduce task performance by ~26%.
- **Best Practice:** Pair with explicit constraints and few-shot/RAG exemplars.