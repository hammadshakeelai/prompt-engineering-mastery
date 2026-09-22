# Token Healing: Eliminating Token Boundary Bias (Lundberg et al. 2023)

## 1. The Token Boundary Problem in Greedy Tokenizers
Byte-Pair Encoding (BPE) tokenizers greedily compress character sequences into the longest matching token ID.
- **The Compound Token Trap:** If a prompt ends with `"http:"`, greedy tokenization assigns `["http", ":"]`.
- In pre-training corpora, the compound token `["http://"]` frequently exists as a single token.
- Because the prompt committed to `":"`, the model cannot emit `["http://"]`. It is forced to emit `["//"]`, which has an unnaturally lower prior probability in training distributions.
- Trailing spaces, quotes, and punctuation cause similar artificial fragmentation.

## 2. The Token-Healing Algorithm (Guidance)
Scott Lundberg et al. introduced **Token Healing** to dynamically repair prompt-generation boundaries:
1. **Rollback:** Pop the final prompt token $t_N$, decoding its character string $s = \text{decode}(t_N)$.
2. **Prefix-Constrained Generation:** Truncate prompt context to $t_1, \dots, t_{N-1}$. The model's first sampling step is constrained via a Trie mask to candidate tokens beginning with prefix $s$:
   $$\mathcal{V}_{\text{healed}} = \{ v \in \mathcal{V} \mid \text{decode}(v) \text{ starts with } s \}$$
3. **Healing:** If the model assigns higher probability to a compound token (e.g., `["http://"]`), the boundary is seamlessly merged into a single token.

## 3. Grammar-Constrained Decoding Integration
When combined with **LLGuidance** or **Outlines**:
$$\text{Mask}_1 = \text{Mask}_{\text{healed}}(s) \cap \text{Mask}_{\text{grammar}}(\text{State}_0)$$
Ensures that structured generation (JSON schemas, code ASTs) is immune to prompt boundary artifacts, preserving natural log-likelihoods without syntax validation failures.
