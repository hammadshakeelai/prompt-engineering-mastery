# Synchromesh & Constrained Semantic Decoding (Poesia et al., ICLR 2022)

## 1. Syntax vs. Semantic Constraints
Standard grammar-constrained decoding (e.g., [[cfsm_jump_forward_decoding|cFSM]], [[llguidance_cfg_earley_trie|llguidance]]) enforces Context-Free Grammars (CFGs, Chomsky Type-2):
- **The Context-Sensitive Flaw:** Programming languages (Python, SQL, Lean 4) require context-sensitive rules (Chomsky Type-1): variable scope, type derivation $\Gamma \vdash e : \tau$, and relational schema definitions.
- Syntactically correct outputs frequently suffer from catastrophic runtime exceptions: referencing undeclared identifiers or joining invalid SQL foreign keys.

## 2. Constrained Semantic Decoding (CSD) State Machine
Synchromesh tracks dynamic program semantics during decoding via a state tuple:
$$\sigma_t = (\mathcal{T}_t, \Gamma_t, \mathcal{S})$$
where $\mathcal{T}_t$ is the partial AST, $\Gamma_t$ is the active typing environment/symbol table, and $\mathcal{S}$ is the external schema.
- **Semantic Masking Criterion:**
  $$\mathcal{V}_{\text{valid}}(x_{<t}) = \{ v \in \mathcal{V} \mid \exists y \text{ s.t. } x_{<t} \circ v \circ y \in \mathcal{L}(\mathcal{G}) \land \text{WellTyped}(x_{<t} \circ v \circ y, \sigma_t) \}$$
- **Partial Expression Typing:** Incomplete token sequences are evaluated using conservative lookahead typing rules, masking tokens that cannot produce any validly typed expression.

## 3. Scope-Trie Pruning & Empirical Gains
- **Dynamic Scope Intersection:** When generating identifiers, CSD dynamically prunes the vocabulary trie against symbols in active scope ($\Gamma_t$), guaranteeing that emitted tokens only produce declared variables or valid table/column names.
- **Benchmark Performance:** On Spider text-to-SQL benchmarks, eliminates 100% of runtime semantic crashes, lifting execution accuracy from $61.2\%$ to $86.7\%$ without retraining base language models.
