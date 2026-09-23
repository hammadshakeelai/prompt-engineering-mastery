---
name: context-compression-engineer
description: Specialized directive for prompt compression, token-level self-information pruning (Selective Context, LLMLingua-1/2), LongLLMLingua question-aware compression, document reordering, and KV-cache budget optimization.
---

# Context Compression Engineer Skill

Use this skill when designing, benchmarking, or deploying prompt compression pipelines, reducing token latency and API cost in long-context RAG systems, eliminating "lost-in-the-middle" positional biases, or implementing information-theoretic token pruning algorithms.

## 1. Information-Theoretic Prompt Compression Mechanics

1. **Shannon Self-Information Pruning (Selective Context, EMNLP 2023):**
   - Calculate conditional self-information for each lexical unit $u_i$ using a lightweight base model $\mathcal{M}$ (e.g., LLaMA-3-8B or GPT-2):
     $$I(u_i) = -\frac{1}{|u_i|} \sum_{t=1}^{|u_i|} \log P_{\mathcal{M}}(x_{i, t} \mid x_{<i}, x_{i, <t})$$
   - Retain units above empirical $(1-\rho)$-quantile threshold $\tau_\rho$, filtering formulaic filler and compressing prompts by up to $50\%$ with $<0.03$ BERTscore degradation.

2. **Contrastive Perplexity & Question-Aware Compression (LongLLMLingua, ACL 2024):**
   - In retrieval-augmented generation (RAG) and long-context QA, unconditional compression discards query-relevant details.
   - Evaluate question-conditioned mutual information:
     $$I(x_i; q) = \log \frac{P_{\mathcal{M}}(x_i \mid \text{context}, q)}{P_{\mathcal{M}}(x_i \mid \text{context})}$$
   - Apply dynamic budget allocation across retrieved chunks, assigning higher token retention ratios $\rho_k$ to documents with high query relevance.

3. **Subsequence Recovery & Syntax Preservation:**
   - Always enforce contiguous span protection around named entities, numerical values, and code syntax markers.
   - Use token boundary alignment to prevent half-subword slicing and un-invertible token fragments.

## 2. Document Reordering & Positional Bias Mitigations

1. **Combating "Lost-in-the-Middle" Attention Decay:**
   - Transformers display U-shaped attention curves: tokens at the extreme beginning (primacy bias) and end (recency bias) receive significantly more attention weight than tokens in the middle $60\%$.
   - **Mandate:** Reorder retrieved documents after compression so that the highest-scoring candidate documents are placed at the prompt boundaries (rank 1 at the end, rank 2 at the beginning, lower ranks in the interior).

2. **Multi-Chunk Reranking & Budget Routing:**
   - Dynamically drop documents whose aggregate query mutual information falls below confidence threshold $\tau_{\text{relevance}}$, preventing distraction and context dilution.

## 3. Evaluation & Deployment Checklist

- Measure compression efficiency using:
  1. **Token Compression Ratio (TCR):** $\frac{|x_{\text{original}}|}{|x_{\text{compressed}}|}$ (target $2\times\text{--}6\times$).
  2. **End-to-End Latency Acceleration:** Measure prefill time reduction on target serving runtime (vLLM / TensorRT-LLM).
  3. **Task Recall Retention:** Benchmark on NaturalQuestions, HotpotQA, and LongBench to verify answer accuracy delta $\Delta \text{Acc} \ge 0$.
