# Corrective RAG (CRAG, Yan et al. 2024)

Evaluates and rectifies retrieved documents via a lightweight retrieval evaluator:
1. **Correct**: If documents are relevant, refines them into fine-grained knowledge strips and filters noise.
2. **Incorrect**: If documents are irrelevant, discards them and executes external web search.
3. **Ambiguous**: If confidence is borderline, fuses refined internal documents with external web search results.
The synthesized knowledge is passed to the generator to eliminate hallucinations from faulty retrieval.