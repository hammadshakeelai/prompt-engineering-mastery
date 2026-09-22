# RAGAS: Evaluation Framework for RAG (Es et al., 2023)

The **RAGAS** (Retrieval Augmented Generation Assessment) framework evaluates RAG pipelines reference-free by decoupling retrieval from generation.

## Core Metrics
### 1. Retrieval Quality
- **Context Precision:** Evaluates signal-to-noise ratio and ranking. Measures whether relevant chunks are ranked higher using mean average precision.
- **Context Recall:** Quantifies the proportion of reference/ground-truth statements successfully retrieved in the context.

### 2. Generation Quality
- **Faithfulness:** Quantifies factual grounding to eliminate hallucinations. Computes the ratio of atomic claims in generated response directly deducible from context.
- **Answer Relevance:** Synthesizes questions from the generated response and measures mean cosine similarity against original query.

## Comparison to Generic LLM-as-a-Judge
Standard LLM-as-a-judge approaches prompt an LLM for holistic 1-5 scores, introducing verbosity and position biases. RAGAS improves via:
1. **Atomic Decomposition:** Decomposes responses into atomic claims rather than subjective grading.
2. **Deterministic Aggregation:** Programmatic mathematical formulas over verified claims.
3. **Reference-Free Monitoring:** Automated evaluation of live systems without human-annotated references.