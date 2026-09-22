# Multi-Query Retriever for Dense Search

- **Mechanism**: Prompts an LLM to generate multiple distinct rephrasings and perspectives of the user query.
- **Execution**: Each variation executes independently against the vector database; candidate documents are aggregated via unique union or reciprocal rank fusion.
- **Impact**: Overcomes lexical sensitivity, bridges vocabulary mismatch gaps, and significantly boosts recall in RAG pipelines.