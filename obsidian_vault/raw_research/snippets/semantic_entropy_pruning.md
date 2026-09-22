# Semantic Entropy & Information-Theoretic Prompt Pruning (2024–2026)

## 1. Mathematical Formulation of Semantic Entropy
Kuhn et al. (*Semantic Uncertainty*, Nature 2023) established that raw token perplexity fails to isolate model hallucination because linguistic variance (synonyms, formatting) inflates token entropy without changing factual meaning.
**Semantic Entropy (SE)** groups sampled sequences $\{y^{(1)}, \dots, y^{(M)}\}$ into equivalence classes $[s]$ using bidirectional Natural Language Inference (NLI):
$$\mathcal{H}_{\text{sem}}(x) = -\sum_{[s]} p([s] \mid x) \log p([s] \mid x), \quad p([s] \mid x) = \sum_{y \in [s]} p(y \mid x)$$
- **Low Token Entropy, Low SE:** Factual certainty with deterministic phrasing.
- **High Token Entropy, Low SE:** Factual certainty with diverse stylistic paraphrasing.
- **High Token Entropy, High SE:** Genuine epistemic uncertainty and imminent confabulation.

## 2. Information-Theoretic Prompt Pruning
Prompt compression algorithms use token-level mutual information $I(x_t; Y \mid x_{<t})$ to prune tokens:
- **Causal Compressor Blindness:** Unidirectional models (LLMLingua-1, Selective-Context) cannot evaluate a token's necessity for subsequent text, frequently pruning early key variables.
- **Bidirectional Distillation (LLMLingua-2):** Trains compact bidirectional encoders (XLM-RoBERTa) using GPT-4 token preservation labels, evaluating full past and future context simultaneously.
- **Efficiency:** Delivers 2x–5x compression with 3x–6x lower preprocessing latency, preserving reasoning accuracy on GSM8K and LongBench.
