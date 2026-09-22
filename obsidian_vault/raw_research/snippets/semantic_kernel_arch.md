# Semantic Kernel: Connectors & Planners Architecture

- **Connectors**: Modular abstraction layers standardizing access to AI model providers, vector stores (Qdrant, Pinecone), and external APIs for semantic memory and RAG.
- **Planners**: Goal-oriented reasoning engines using LLMs to dynamically synthesize execution plans (sequential, stepwise, or function-calling DAGs) over registered plugins and native functions.