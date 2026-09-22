# Cross-Encoder Reranking & Cohere Rerank

Secondary stage to boost retrieval precision after bi-encoder/BM25 retrieval:
- Bi-encoders embed queries and documents separately, missing fine-grained token interactions.
- Cross-encoders evaluate query-document pairs simultaneously using full cross-attention across all tokens.
- **Production RAG**: Re-scores and filters candidates down to top-k most relevant passages before context injection, suppressing noise, reducing hallucinations, and lowering prompt token costs.