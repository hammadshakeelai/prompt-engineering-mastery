# RAG-Fusion & Multi-Query Expansion

Overcomes single-query retrieval limitations through multi-query expansion and rank-based aggregation:
1. **Query Expansion**: LLM generates multiple diverse rephrasings and perspectives of the original prompt.
2. **Parallel Retrieval**: Each variation independently retrieves candidate documents.
3. **Reciprocal Rank Fusion (RRF)**: Merges and re-ranks results:
   $$\text{RRF Score}(d) = \sum_{q} \frac{1}{k + \text{rank}(d, q)}$$
   Prioritizes documents appearing consistently near the top across multiple queries.
4. **Generation**: Top-ranked fused documents are passed into context for grounded synthesis.