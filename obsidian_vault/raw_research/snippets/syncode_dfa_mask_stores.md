# SynCode: DFA Mask Stores & Incremental Lookahead Parsing (Ugare et al., TMLR 2024)

## 1. The Dynamic Parsing Bottleneck in CFG Constrained Decoding
Enforcing Context-Free Grammars (CFGs) during autoregressive token generation requires determining legal next-token subsets $\mathcal{V}_{\text{valid}} \subseteq \mathcal{V}$ at step $t$.
Existing frameworks face severe architectural trade-offs:
- **Regex Union Combinatorial Explosion:** Regex/FSM-based engines (e.g., Outlines) compile dynamic union automatons $\bigcup_{\tau \in \text{Follow}} r_\tau$, which explode exponentially on recursive grammars (e.g., Python ASTs).
- **Runtime Trie Traversal Overhead:** Dynamic Earley/LL parsers (e.g., [[llguidance_cfg_earley_trie|LLGuidance]]) traverse prefix tries over $|\mathcal{V}|$ on every subword step, introducing $0.5\text{--}5\,\text{ms}$ CPU serialization latency.

## 2. SynCode Architecture & DFA Mask Stores
Shubham Ugare et al. (*SynCode: Grammar-Guided Generation via Context-Free Parsing with Lookahead*, UIUC / TMLR 2024) eliminate runtime parsing latency by decoupling lexical DFA tracking from grammatical LR derivation:
1. **Offline DFA Mask Store:**
   - Compiles terminal patterns into a unified character DFA $\mathcal{A} = (Q, \Sigma_{\text{char}}, \delta, q_0, F)$.
   - Precomputes a Boolean bitmask for every DFA state $q \in Q$ and terminal $\tau \in \Sigma$:
     $$\mathbf{M}(q, \tau) \in \{0, 1\}^{|\mathcal{V}|}$$
     marking whether subword token $v \in \mathcal{V}$ is an admissible prefix leading to terminal $\tau$.
2. **Incremental Lookahead LR Parsing:**
   - Maintains an incremental LR state stack $\mathcal{S}_t$.
   - Computes allowed follow-set terminals $\mathcal{T}_{\text{valid}} = \text{Follow}(\mathcal{S}_t)$ in $\mathcal{O}(1)$ via table lookup.
   - Computes the active step mask through parallel bitwise OR operations across precomputed bit vectors:
     $$\mathbf{M}_{\text{step}} = \bigvee_{\tau \in \mathcal{T}_{\text{valid}}} \mathbf{M}(q_{\text{curr}}, \tau)$$
   - Executes in $\sim 12\,\mu\text{s}$ via SIMD/AVX-512 vector instructions without dynamic regex construction or trie scans.

## 3. Theoretical & Empirical Guarantees
- **Soundness & Completeness:** Formally guarantees that 100% of generated outputs satisfy the target CFG $\mathcal{L}(G)$, with zero premature pruning of valid syntax paths.
- **Zero Syntax Errors on Full Languages:** Achieves **0% syntax errors** on Python (HumanEval), Go, and SQL, while delivering $2\times\text{--}5\times$ speedups over dynamic trie parsers.
