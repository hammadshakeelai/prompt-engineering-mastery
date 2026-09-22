import os

base = r'c:\Users\HP\Documents\antigravity\vibrant-kepler\obsidian_vault\raw_research\snippets'
os.makedirs(base, exist_ok=True)

files = {
'code_evals.md': """# LLM Code Evaluation: HumanEval, MBPP, SWE-Bench, LiveCodeBench

## HumanEval and pass@k (Chen et al., 2021)
164 hand-written Python programming challenges with docstrings, function signatures, and unit tests.
pass@k metric: unbiased estimation where n >= k candidates are sampled, c pass all unit tests.
pass@1 = single-turn reliability; pass@100 = capability ceiling.

## MBPP (Austin et al., 2021 - Google)
974 crowd-sourced entry-level programming tasks for few-shot evaluation.
Covers core algorithmic primitives, math, string operations, and data structures.
Each task includes English problem statement, reference solution, and 3 automated test assertions.

## SWE-Bench (Jimenez et al., 2024)
Repository-level software engineering: 2,294 real-world GitHub issues from 12 Python repos (Django, SymPy, scikit-learn).
Models must parse issue descriptions, locate buggy files, generate unified git patches passing regression suites.
Much harder than isolated function synthesis - requires codebase-level navigation.

## LiveCodeBench (Jain et al., 2024)
Contamination-free benchmark continuously refreshed with newly released competitive programming problems.
Sources: LeetCode, AtCoder, Codeforces problems strictly after model training cutoffs.
Tests beyond generation: code execution, self-repair, test output prediction.""",

'adaptive_computation.md': """# Adaptive Computation and Early Exit in LLMs

## 1. Shallow-Deep Architectures
Auxiliary classification heads or exit modules at intermediate transformer layers.
Easy tokens (syntax-bound, repetitive) exit early; complex tokens propagate to deeper layers.
Mitigates overthinking on trivial tokens, preserves representations for challenging steps.

## 2. CALM: Confident Adaptive Language Modeling (Schuster et al., 2022)
Dynamic early exit for autoregressive generation.
- Confidence Metrics: Softmax probability, entropy across top-k tokens, or hidden-state similarity between adjacent layers.
- Calibrated Thresholds: If confidence metric exceeds threshold, token emitted immediately, bypassing remaining layers.
- Performance Guarantees: Distribution-free risk control bounds sequence-level quality degradation.
- Achieves up to 3x wall-clock speedups without significant accuracy loss.

## 3. Token-Level Compute Budgeting
Standard LLM inference: fixed O(L) compute per token.
Adaptive computation: dynamic per-token budgeting.
- Punctuation and boilerplate: minimal FLOPs.
- Key entities, multi-hop reasoning, rare vocabulary: maximal capacity.
- Systems can throttle confidence thresholds to meet hard latency budgets.""",

'prompt_compression_ext.md': """# Prompt Compression Techniques

## LLMLingua-2 (Pan et al., 2024)
Reframes prompt compression as binary token classification (preserve vs. discard).
Distills compression labels from GPT-4 to train a small bidirectional Transformer encoder (XLM-RoBERTa).
Full bidirectional context for faithful, low-latency, task-agnostic extractive pruning.

## RECOMP Abstractive Compression (Xu et al., ICLR 2024)
Fine-tuned T5-large trained via distillation to synthesize multiple retrieved RAG passages into concise query-focused summary.
Mitigates lost-in-the-middle degradation.
Can output empty string to filter completely irrelevant passages.

## AutoCompressor Soft Tokens (Chevalier et al., 2023)
Extends vocabulary with <Sum> summary tokens that recursively compress text chunks into dense summary vectors.
Cached vectors preserve long-horizon context without storing raw token sequences.

## Gist Tokens (Mu et al., NeurIPS 2023)
Condenses entire natural language instructions into small fixed set of virtual gist tokens.
Modified attention masks during training force model to attend through bottleneck gist tokens.
Enables up to 26x context compression with reusable, cached activation prefixes.""",

'benchmark_contamination.md': """# Benchmark Contamination and Data Leakage in LLM Evaluation

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
3. Synthetic Mutation: Procedural variations (perturbed logic, variable renaming, altered numerical values) break verbatim memorization while preserving conceptual complexity.""",

'instruction_tuning.md': """# Instruction Following vs. Instruction Tuning

## Key Paradigms

### FLAN (Wei et al., 2021 - Google)
Multi-task instruction tuning across 62 NLP datasets grouped into 12 task clusters on LaMDA-PT (137B).
Proved instruction tuning significantly improves zero-shot performance on held-out task clusters.
Cross-task generalization is an emergent capability at scale.

### T0 and PromptSource (Sanh et al., 2021 - BigScience)
Trained T5 (11B) on extensive prompt diversity across varied task formatting templates.
T0 matches or surpasses zero-shot performance of GPT-3 (175B) on unseen tasks.
Key: prompt diversity > instance count per task.

### Super-NaturalInstructions (Wang et al., 2022 - Allen AI)
1,616 diverse NLP tasks across 76 task types in 55 languages with structured declarative instructions.
Scaling task and instruction diversity yields better OOD generalization than scaling instance count.

### InstructGPT (Ouyang et al., 2022 - OpenAI)
SFT -> Reward Model -> PPO on open-ended human intent alignment.
Human annotators prefer 1.3B aligned model over 175B unaligned base model.

## Key Principles
- Instruction Diversity: Forces meta-task of instruction execution, prevents mode collapse.
- Task Formatting Templates: Verbalize structured datasets into multiple phrasing variants, making representations invariant to prompt syntax.""",

'mamba_ssm.md': """# Structured State Space Models (Mamba) vs. Transformers for Prompting

## 1. Architectural Differences
- Transformers: O(L^2) time/memory via pairwise dot-product attention. Exact lossless KV cache.
- Mamba (Gu and Dao, 2023): Linear SSM (h'_t = A*h_{t-1} + B*x_t, y_t = C*h_t). Selectivity makes B, C, Delta functions of current input. O(L) compute, O(1) inference memory.
- Hardware-aware parallel scan replaces attention's matmul for efficient training.

## 2. Implications for Very Long Sequences
- Throughput: Transformers face quadratic prefill latency; Mamba scales linearly with constant generation latency.
- Memory: Mamba processes 100k-1M+ tokens with drastically reduced VRAM.
- Trade-off: Transformers preserve exact token identities (strong associative recall). Mamba compresses history into fixed-size state (lossy decay on verbatim long-range retrieval).

## 3. Divergent Prompting Strategies
- Recency Bias: Mamba has continuous recurrent state decay (no attention sinks). Place crucial context directly before generation boundary (end of prompt).
- Few-Shot Limits: Diminishing returns for Mamba vs. Transformers with 20+ examples due to state compression. Prefer zero-shot with rich instructions.
- Hybrid Architectures (Jamba, Zamba): Interleave attention layers for exact associative recall + Mamba layers for long-context efficiency.""",

'chain_of_symbol.md': """# Chain-of-Symbol (CoS) Prompting - Hu et al. (2023)

Citation: arXiv:2305.10276

## Core Motivation
LLMs struggle with spatial planning in natural language due to semantic ambiguity and compounding hallucinations.
Standard CoT uses verbose natural language scratchpads, introducing spatial confusion during intermediate state tracking.

## Methodology
Replaces natural language scratchpads with condensed symbolic representations:
- Symbolic State Representation: Entities, locations, and spatial relations encoded as concise symbols/tuples (coordinates, connection symbols, state tokens) rather than prose narratives.
- Intermediate reasoning steps update state transitions symbolically, providing an explicit mental map.
- Plug-and-play: Pure prompt engineering, no fine-tuning required.

## Empirical Gains
- Brick World (Spatial Manipulation): ChatGPT 31.8% (CoT) -> 92.6% (CoS), +60.8% absolute gain.
- Natural Language Navigation: Outperforms CoT by preserving spatial coherence and path validity.
- Spatial QA (SPARTUN): Consistently outperforms CoT, reducing relational errors.
- Efficiency: Reduces intermediate reasoning tokens by up to 65.8%, lowering inference latency.""",
}

written = 0
for fname, content in files.items():
    path = os.path.join(base, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    written += 1

print(f'Wrote {written} files.')
