# Repository-Level Prompt Engineering: RepoCoder & Aider's Repo Map

Scaling code generation from single functions to multi-file repositories requires structured context management to stay within token budgets without losing architectural awareness.

## 1. RepoCoder: Iterative Retrieval & Generation (Zhang et al., 2023)
- **Problem:** Cross-file dependencies, shared classes, and imports are scattered throughout the repository, exceeding standard context limits.
- **Iterative Loop:** 
  1. *Retrieve:* Similarity-based retriever fetches candidate code snippets across the repo.
  2. *Generate:* LLM produces intermediate completion code.
  3. *Refine:* The newly generated code acts as an expanded query to retrieve more targeted dependencies for the next iteration.
- **Impact:** Eliminates isolated completions by progressively binding unseen types and methods across files.

## 2. Aider's Repo Map: AST & Personalized PageRank
- **Tree-sitter AST Extraction:** Instead of raw text embeddings, uses Tree-sitter parsers to extract semantic declarations (classes, functions, method signatures) across all repository source files.
- **Dependency Graph Construction:** Models symbols and references as a directed call/dependency graph.
- **Personalized PageRank (PPR):** Ranks symbol relevance with respect to currently focused files or user query topics.
- **Token-Budgeted Map:** Renders a compact, hierarchical ASCII map of key definitions and signatures, giving the LLM global codebase visibility for a fraction of the token cost of full files.
