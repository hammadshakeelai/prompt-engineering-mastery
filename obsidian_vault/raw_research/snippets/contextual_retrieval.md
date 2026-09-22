# Contextual Retrieval (Anthropic 2024)

Solves context loss caused by splitting documents into isolated chunks.
Standard chunking strips surrounding document context, causing retrieval failures on pronouns or undefined metrics.
- **Mechanism**: Prompts an LLM (Claude 3 Haiku) with the full document to generate a concise, chunk-specific explanatory context (50-100 tokens), prepended directly to each chunk prior to embedding and indexing.
- **Impact**: When combined with hybrid search (dense + BM25) and reranking, reduces retrieval failure rates by up to 67%. Cost-effective via prompt caching.