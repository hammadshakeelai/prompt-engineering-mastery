# Lookahead-then-Verify (LAVE) CFG Constrained Decoding for Diffusion LLMs (Zhang et al., 2026)

## 1. The Prefix Breakdown in Non-Autoregressive dLLMs
Conventional grammar-constrained decoding engines (e.g., [[xgrammar_structured_generation|XGrammar]], [[llguidance_earley_cfg|LLGuidance]], Outlines) presuppose continuous left-to-right token prefixes $x_{<t}$. At each step, a deterministic finite automaton (DFA) or pushdown automaton (PDA) filters the vocabulary to valid next-token follow-sets $\mathcal{V}_{\text{valid}}(x_{<t})$.
In contrast, Diffusion Large Language Models (dLLMs such as Dream-v0 and LLaDA) denoise sequences non-autoregressively:
$$x^{(T)} = [\text{MASK}, \dots, \text{MASK}] \to \dots \to x^{(0)}$$
Intermediate states $x^{(t)}$ feature concrete tokens at arbitrary, discontinuous indices separated by unresolved `[MASK]` tokens. Standard LL/LR parsers fail catastrophically because grammar rules cannot directly evaluate sequences with interior non-terminal gaps.

## 2. The Lookahead-then-Verify (LAVE) Framework
Yitong Zhang et al. (*Lookahead-then-Verify: Reliable Constrained Decoding for Diffusion LLMs under Context-Free Grammars*, arXiv:2602.00612 / ISSTA 2026) resolve this mismatch through parallel distribution lookahead:

1. **Parallel Marginal Logits:**
   In every diffusion step, the dLLM forward pass generates full vocabulary distributions across all masked slots $\mathcal{M}^{(t)} = \{ i \mid x_i^{(t)} = [\text{MASK}] \}$ simultaneously: $p_\theta(x_i \mid x^{(t)})$.
2. **Lookahead Sampling:**
   When proposing token updates $\hat{x}_U$ for candidate slots $U \subseteq \mathcal{M}^{(t)}$, LAVE samples $N$ complete sequences for remaining masked positions from the predicted marginals:
   $$\tilde{x}^{(j)}_{\mathcal{M} \setminus U} \sim \prod_{k \in \mathcal{M}^{(t)} \setminus U} p_\theta(x_k \mid x^{(t)}), \quad j \in \{1, \dots, N\}$$
   Each candidate $\tilde{x}^{(j)}$ is a concrete sequence free of `[MASK]` tokens.
3. **Formal CFG Verification:**
   Candidates are evaluated via formal CFG parsers $\mathcal{P}_{\text{CFG}}$ (e.g., Python AST or JSON lexers):
   $$\text{Valid}(\tilde{x}^{(j)}) = \mathbf{1}[\mathcal{P}_{\text{CFG}}(\tilde{x}^{(j)}) \in \mathcal{L}(G)]$$
   The proposal $\hat{x}_U$ is accepted if $\sum_{j=1}^N \text{Valid}(\tilde{x}^{(j)}) \ge 1$; otherwise, it is rejected and re-sampled, preventing the model from traversing dead-end syntactic trajectories.

## 3. Empirical Performance
- **100% Syntax Validity:** LAVE eliminates the $18\%\text{--}34\%$ syntax failures of unconstrained dLLMs on structured generation.
- **High Acceptance:** With $N = 10$, proposal acceptance reaches **98.1%** (Dream-v0-Instruct-7B) and **97.3%** (LLaDA-8B-Instruct).
- **Inference Efficiency:** Imposes $<12\%$ wall-clock latency overhead due to parallel token sampling and compiled C++ AST parsing.
