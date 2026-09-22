# Benchmark Contamination and Data Leakage in LLM Evaluation

## Impact: Score Inflation and Memorization
Evaluation benchmarks entering pre-training corpora cause models to exploit verbatim recall over true reasoning.
Inflates performance metrics (MMLU, GSM8K, HumanEval), masking brittle real-world generalization.

## Detection Methods
1. Exact Deduplication: Cryptographic hashing (MD5/SHA-256) matches byte-for-byte identical sequences. Fails against minor formatting tweaks.
2. N-Gram Overlap: Lexical overlap across 8-13-gram sequences with MinHash/LSH. Identifies near-duplicate and rephrased prompts.
3. Membership Inference Attacks: Statistical black-box probing of loss/perplexity discrepancies - models assign anomalously low cross-entropy to seen prompts vs. syntactically matched counterfactuals.

## Contamination-Resistant Evaluation
1. Dynamic Benchmarks: Continuously refresh test suites with newly authored, time-gated tasks.
2. LiveBench: Updated monthly with fresh data strictly after training cutoffs (arXiv papers, competitive programming, math olympiads). Objective automated scoring without LLM judges.
3. Synthetic Mutation: Procedural variations (perturbed logic, variable renaming, altered numerical values) break verbatim memorization while preserving conceptual complexity.