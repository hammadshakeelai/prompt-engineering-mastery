---
name: agent-eval-benchmarker
description: Specialized directive for zero-hallucination benchmark protocols, LLM-as-a-judge calibration (G-Eval, MT-Bench, Arena-Hard), Semantic Entropy epistemic uncertainty quantification, verifiable execution harnesses (SWE-bench, LiveCodeBench), and contamination detection.
---

# Agent Eval Benchmarker Skill

Use this skill when auditing model performance, validating prompt variations, designing automated evaluation rubrics, quantifying epistemic uncertainty, or executing verifiable code/reasoning benchmarks.

## 1. Zero-Hallucination & Verifiable Execution Directives

1. **Verifiable Ground-Truth Over Synthetic Heuristics:**
   - In mathematical, algorithmic, and coding domains, prioritize deterministic execution harnesses (unit test pass rates, formal proof verifiers like Lean 4 / Isabelle, compiler checks) over raw string matching or ungrounded model judgment.
   - For SWE-bench and LiveCodeBench pipelines, enforce isolated Docker sandbox execution with strict timeouts and state rollbacks.

2. **LLM-as-a-Judge Calibration & Bias Mitigation:**
   - **Position Bias:** Always run pairwise judgments twice with swapped candidate order ($A \text{ vs } B$ and $B \text{ vs } A$). Discard non-transitive or self-contradictory ties unless explicitly modeled.
   - **Verbosity Bias:** Normalize candidate lengths or instruct the judge specifically to penalize hollow verbosity and prioritize concise correctness.
   - **Self-Enhancement Bias:** Avoid using the evaluated model family as the judge (e.g., using GPT-4 to judge GPT-3.5 vs Claude). Use independent, high-capability judges calibrated against human Elo rankings (Arena-Hard-Auto).

## 2. Epistemic Uncertainty & Semantic Entropy

1. **Semantic Equivalence Clustering:**
   - Sample $N$ independent generations at temperature $T \approx 0.7\text{--}1.0$.
   - Cluster outputs into equivalence classes $[s]$ using bidirectional NLI entailment:
     $$\mathcal{H}_{\text{sem}}(x) = -\sum_{[s]} p([s] \mid x) \log p([s] \mid x)$$
   - Flag predictions with high semantic entropy ($\mathcal{H}_{\text{sem}} > \tau$) as uncertain; trigger fallback to test-time search (MCTS/Best-of-$N$) or request external tool verification.

## 3. Benchmark Integrity & Contamination Audits

1. **Synthetic & Dynamic Benchmarking:**
   - Continuously refresh test cases (e.g., LiveCodeBench time-windowed problems) to prevent pre-training contamination and memorization.
   - Perform n-gram overlap and perplexity anomaly checks against public training corpora to detect contaminated evaluation sets.
