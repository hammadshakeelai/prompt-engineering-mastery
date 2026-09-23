# Outlines: DFA Logit Masking & Regex-Constrained Decoding (Willard & Louf, 2023)

## 1. Syntactic Fragility in Autoregressive Generation
Autoregressive token generation lacks formal grammar guarantees. When generating structured syntax (JSON, SQL, Regex), error probabilities compound exponentially over sequence length: $P(\text{error}) = 1 - \prod_{t=1}^T P(\text{valid}_t)$, causing frequent syntax crashes.

## 2. Deterministic Finite Automaton (DFA) Compilation
Outlines (Willard & Louf, 2023) compiles regex/JSON schemas into a minimal DFA $\mathcal{M} = (Q, \Sigma, \delta, q_0, F)$. To resolve the subword tokenization mismatch (multi-character subwords $w \in \mathcal{V}$ vs. character transitions $c \in \Sigma$):
1. Compute extended transition $\delta^*(q, w) = \delta(\dots \delta(q, c_1) \dots, c_k)$.
2. For each state $q \in Q$, precompute the set of valid vocabulary tokens:
   $$V(q) = \left\{ w \in \mathcal{V} \;\middle|\; \delta^*(q, w) \text{ is valid and leads to } F \right\}$$
3. Materialize $V(q)$ as precomputed Boolean bitmasks $\mathbf{M}_q \in \{0, 1\}^{|\mathcal{V}|}$.

## 3. Runtime Logit Masking & Zero Overhead
At generation step $t$ in state $q_t$:
- Mask logits: $\tilde{z}_{t, v} = z_{t, v}$ if $\mathbf{M}_{q_t}[v]=1$, else $-\infty$.
- Sample token $x_t \sim \operatorname{Softmax}(\tilde{z}_t)$ and advance state $q_{t+1} = \delta^*(q_t, x_t)$.
- Guarantees $P(\text{syntax error}) \equiv 0$ with execution latency $<15\,\mu\text{s}$, widely deployed in vLLM and SGLang.
