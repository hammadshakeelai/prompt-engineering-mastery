# LLGuidance: Context-Free Grammar Parsing & Subword Trie Masking (2024–2026)

## 1. Limitations of Regular Expressions & DFAs
Flat finite state automata cannot enforce recursive, nested syntax (arbitrary nested JSON objects, SQL queries, programming ASTs) due to the pumping lemma ($a^n b^n$). Enforcing hierarchical syntax requires **Context-Free Grammars (CFGs)** and **Pushdown Automata (PDA)**.

## 2. The Subword Tokenization Disconnect
CFG terminals operate over characters, while LLMs emit multi-character subwords ($V \approx 128\text{k}$). Single tokens routinely straddle production rule boundaries or contain incomplete prefix terminals.

## 3. The LLGuidance Three-Tier Architecture
**LLGuidance** (Microsoft Research, 2024–2025) executes subword CFG masking in $\sim 50\mu\text{s}$ CPU time per token:
1. **Earley Parser:** Maintains dotted production items ($A \to \alpha \cdot B \beta$) to support arbitrary ambiguous and left-recursive grammars.
2. **Brzozowski Regular Expression Derivatives:**
   $$\partial_c R = \{ w \mid c w \in R \}$$
   Lazily tracks regular expression constraints character-by-character without compiling exponential DFAs.
3. **Prefix Tokenizer Trie Pruning:**
   Traverses a pre-indexed byte Trie of the vocabulary $V$. Any branch encountering an empty derivative $\partial_c R = \emptyset$ or invalid Earley transition is pruned immediately.
4. **Deterministic Fast-Forwarding:**
   When remaining grammar transitions specify an unambiguous continuation token, generation skips GPU forward passes entirely and splices the deterministic token directly into the KV cache.
