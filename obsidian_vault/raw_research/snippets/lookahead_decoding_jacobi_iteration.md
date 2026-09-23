# Lookahead Decoding: Jacobi Parallel Autoregressive Decoding (Fu et al., ICML 2024)

## 1. Limitations of Draft-Model Speculation
Traditional speculative decoding requires maintaining and synchronizing a secondary draft model in memory, which introduces deployment complexity, memory overhead, and distribution mismatch on custom domains.

## 2. Jacobi Fixed-Point Formulation
Lookahead Decoding (Fu et al., UC Berkeley / ICML 2024 / arXiv:2402.02057) treats autoregressive generation as solving a non-linear fixed-point equation system:
$$x_i^{(k+1)} = \operatorname{argmax}_{v \in \mathcal{V}} P(v \mid x_{<t}, x_1^{(k)}, \dots, x_{i-1}^{(k)})$$
All token positions in an $N$-token window are updated concurrently in parallel using Jacobi iteration until fixed-point convergence ($\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)}$).

## 3. Dual-Branch Architecture & Speedup
Operates two concurrent branches in a single forward pass:
- **Lookahead Branch:** Generates and refines $n$-gram candidates.
- **Verification Branch:** Validates candidate $n$-grams against exact causal contexts.
Emits multiple verified tokens per pass without draft models, achieving **$1.8\times\text{--}4.0\times$ speedup** across MT-Bench, GSM8K, and HumanEval.
