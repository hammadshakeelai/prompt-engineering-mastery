# Think Step by Step vs. Direct Answering: When CoT Helps and Hurts

## When CoT Helps
- **Complex Multi-Step & Math Reasoning:** Multi-hop logical deductions, symbolic manipulation, and algorithmic math (GSM8K, MultiArith).
- **Working Memory & Decomposition:** Emitting intermediate tokens acts as an external computational scratchpad, breaking compound state transitions into tractable single steps.

## When CoT Hurts
- **Simple & Intuitive Tasks:** Straightforward factual recall, sentiment analysis, or pattern-matching suffer from "reasoning inflation" and spurious failure surfaces.
- **Error Cascades:** An early hallucination or arithmetic error compounds down the chain, derailing an otherwise obvious answer.
- **Latency & Cost Overhead:** High prompt instability and token inflation with zero or negative accuracy return on non-reasoning tasks.

## Key Empirical Studies
- **Kojima et al. (2022):** Appending *"Let's think step by step"* surged MultiArith accuracy from 17.7% to 78.7% and GSM8K from 10.4% to 40.7%.
- **Fu et al. (2022):** Reasoning performance scales with chain complexity; selecting exemplars with more reasoning steps outperformed standard few-shot CoT.
- **Sprague et al. (2024):** Survey showing CoT benefits are predominantly confined to mathematical and symbolic domains; on commonsense tasks, direct answering matches or exceeds CoT.