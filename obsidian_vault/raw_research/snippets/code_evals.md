# LLM Code Evaluation: HumanEval, MBPP, SWE-Bench, LiveCodeBench

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
Tests beyond generation: code execution, self-repair, test output prediction.