# Subword-Aligned Grammar Constraints & Subterminal Trees (DOMINO) (Beurer-Kellner et al., 2024)

## 1. The Subword Misalignment Dilemma
Autoregressive LLMs generate tokens over statistical Byte-Pair Encoded (BPE) subwords, whereas formal grammar specifications (CFG, JSON Schema, ASTs) define terminals (e.g., strings, integers, identifiers) over character alphabets.
Naive grammar-constrained decoders often restrict valid next characters in ways that break multi-character subwords:
- **Token Misalignment Tax:** For example, when completing a JSON string value, the engine may disallow valid multi-token completions because the grammar parser enforces intermediate character bounds, forcing the model into inefficient single-byte emissions (` `, `Y`, `o`, `r`, `k` instead of ` York`).
- **Distribution Distortion:** Forcing sub-optimal tokenization shifts internal representations away from the pretraining manifold, reducing reasoning accuracy and code correctness despite syntactic compliance.

## 2. DOMINO Architecture & Subterminal Trees
Beurer-Kellner, Fischer, and Vechev (*Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation*, ETH Zurich / arXiv:2403.01895, 2024) introduce **DOMINO**:
1. **Subterminal Tri-Partition:**
   Every vocabulary token $t \in \mathcal{V}$ is classified with respect to grammar scanner automata:
   - **Start Subterminals ($\mathcal{S}$):** Tokens initiating a terminal from an idle scanner state.
   - **Continuation Subterminals ($\mathcal{C}$):** Tokens continuing an ongoing terminal without triggering an automata boundary.
   - **End Subterminals ($\mathcal{E}$):** Tokens finalizing a terminal and prompting a PDA state reduction.
2. **Offline Subterminal Trees:**
   DOMINO precomputes a tree whose nodes are scanner states and edges are subword vocabulary tokens. Admissible subword continuations are retrieved in $\mathcal{O}(1)$ time without running dynamic string lexing during decoding.
3. **Speculative Jump-Forward Bypass:**
   Grammar-mandated deterministic tokens (delimiters, brackets, syntax boilerplate) are inserted without invoking the transformer forward pass, synergizing with [[cfsm_jump_forward_decoding|cFSM Jump-Forward]] mechanics.

## 3. Empirical Efficiency
- **Distribution Preservation:** Preserves natural high-probability subword paths, eliminating accuracy drops caused by [[token_healing_boundary_alignment|token boundary distortion]].
- **Zero Overhead & Speedup:** Eliminates runtime masking latency, delivering $1.5\times\text{--}2.1\times$ speedups over unconstrained decoding on JSON and code benchmarks.
