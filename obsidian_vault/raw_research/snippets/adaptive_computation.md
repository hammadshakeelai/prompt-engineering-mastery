# Adaptive Computation and Early Exit in LLMs

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
- Systems can throttle confidence thresholds to meet hard latency budgets.