# Complexity-Based CoT (Fu et al., ICLR 2023)

**Citation:** arXiv:2210.00720

## Core Hypothesis
Demonstrating higher reasoning complexity in few-shot exemplars elicits longer, more thorough reasoning chains.

## Mechanisms
1. **Complexity Metric:** Step count (newline splits) or total chain length.
2. **Prompt Selection:** Automatically select exemplars with most reasoning steps.
3. **Complexity-Based Consistency:** Restrict majority voting to top-K most complex sampled paths.

## Results
- Average **+5.3%** accuracy gain across multi-step benchmarks.
- Up to **+18%** on challenging subsets.
- Outperforms standard CoT on GSM8K, MathQA, MultiArith, BIG-Bench Hard.