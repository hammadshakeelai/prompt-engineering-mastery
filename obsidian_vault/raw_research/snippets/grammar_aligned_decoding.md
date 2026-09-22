# Grammar-Aligned Decoding (GAD) & Speculative Structural Verification (2024–2026)

## 1. The Distributional Bias of Naive Logit Masking
Grammar-Constrained Decoding (GCD) enforces syntax by masking illegal tokens before softmax. However, local normalization:
$$p_{\text{GCD}}(x_t \mid x_{<t}) = \frac{p(x_t \mid x_{<t}) \cdot \mathbb{I}(x_t \in \mathcal{V}_{\text{valid}})}{\sum_{v \in \mathcal{V}_{\text{valid}}} p(v \mid x_{<t})}$$
distorts the joint conditional distribution $p(x_{1:T} \mid x \in \mathcal{L}(\mathcal{G}))$. Local greedy choices frequently steer generation into low-likelihood paths or syntactic traps, causing reasoning degradation on structured benchmarks.

## 2. Grammar-Aligned Decoding (GAD) & ASAp Algorithm
Introduced to eliminate distributional drift without sacrificing grammar guarantees:
- **Lookahead Expectation:** Evaluates the approximate future probability mass of valid continuations.
- **ASAp (Adaptive Sampling with Approximate Expected Futures):** Re-weights candidate token logits based on the density of reachable successful completions under grammar $\mathcal{G}$, ensuring generated samples mirror the unconstrained model's true conditional distribution.

## 3. Speculative Verification in LMS Decoders
In Leviathan-type Local-Mask Speculative (LMS) architectures:
- **100% Acceptance on Fixed Grammar States:** Fixed structural syntax (JSON brackets, property quotes, delimiters) produces deterministic transitions where draft tokens achieve 100% speculative acceptance.
- **Parallel Mask Verification:** Verification steps check language model logit acceptance and automaton state validity concurrently in unified GPU kernels, achieving 2.5x–4.5x end-to-end acceleration.
