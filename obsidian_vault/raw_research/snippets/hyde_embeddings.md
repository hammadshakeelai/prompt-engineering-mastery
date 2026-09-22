# HyDE: Hypothetical Document Embeddings (Gao et al. 2022)

Zero-shot retrieval technique transforming query-to-document search into document-to-document similarity:
- Instructs an LLM to generate a hypothetical answer/document to the query.
- Even if hallucinated, the synthetic document captures the semantic structure, style, and vocabulary of a relevant answer.
- An unsupervised dense encoder embeds the hypothetical document to retrieve actual candidates from the index via nearest-neighbor search, bridging query-document density gaps.