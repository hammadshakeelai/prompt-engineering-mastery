# LLMLingua-2: Bidirectional Token-Classification Prompt Compression (Wu et al., ACL 2024)

## 1. Limitations of Perplexity-Based Prompt Compression
First-generation prompt compressors (e.g., original LLMLingua, LongLLMLingua) assess token importance via conditional perplexity under small unidirectional language models (e.g., LLaMA-7B, GPT-2):
$$P(x_i \mid x_{<i})$$
- **Unidirectional Blindness:** Tokens early in a sentence receive high surprise scores simply because subsequent disambiguating context is invisible to the causal mask.
- **Inference Latency Tax:** Evaluating token entropy with autoregressive models adds significant latency, often negating the downstream inference savings.

## 2. LLMLingua-2 Formulation & Data Distillation
Wu et al. (*LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression*, Findings of ACL 2024) reconceptualize prompt compression as a **bidirectional token classification problem**:
1. **Bidirectional Contextualization:**
   Replaces causal LMs with a compact bidirectional Transformer encoder (e.g., XLM-RoBERTa, mBERT). Every token $x_i$ attends to full context $(x_{<i}, x_{>i})$ simultaneously, ensuring global structural awareness.
2. **Data Distillation from Frontier Models:**
   - Extracts extractive compression demonstrations from GPT-4 across diverse domains (code, reasoning, summarization).
   - Constrains GPT-4 to produce token-level keep/drop labels $y_i \in \{0, 1\}$ without hallucinating or paraphrasing words.
3. **Chunk-Level Dynamic Filtering:**
   Given classification probabilities $p(y_i = 1 \mid \mathbf{x})$, tokens are preserved based on dynamic rate thresholds $\tau$:
   $$\mathbf{x}_{\text{compressed}} = \{ x_i \mid p(y_i = 1 \mid \mathbf{x}) \ge \tau \}$$

## 3. Empirical Performance
- **$3\times\text{--}6\times$ Speedup:** Processes prompts $3\times\text{--}6\times$ faster than original LLMLingua, delivering $1.6\times\text{--}2.9\times$ net end-to-end latency reductions.
- **$2\times\text{--}5\times$ Compression Ratio:** Compresses prompt tokens by $50\%\text{--}80\%$ while maintaining or improving downstream task performance across GSM8K, BBH, and LongBench.
