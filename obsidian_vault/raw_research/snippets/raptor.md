# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval (Sarthi et al., 2024)

Citation: arXiv:2401.18059 (ICLR 2024)

## Core Concept
Standard RAG retrieves short isolated chunks, failing on questions requiring holistic synthesis.
RAPTOR builds a multi-level tree by recursively clustering and summarizing chunks bottom-up.

## 1. Hierarchical Clustering
- Chunking & Embedding: Documents divided into leaf chunks (~100 tokens), embedded via SBERT.
- UMAP + GMM Soft Clustering: Reduces dimensionality, then GMM allows overlapping multi-cluster membership.
- Dynamic K: Bayesian Information Criterion (BIC) selects optimal cluster count.

## 2. Recursive Summarization
- LLM synthesizes each cluster's chunks into an abstractive summary node.
- Summaries re-embedded, reclustered, summarized iteratively, forming layers of increasing abstraction.

## 3. Improving Multi-Hop Reasoning
- Tree Traversal: Top-down branch pruning OR collapsed tree search (query all leaf + summary nodes via cosine similarity).
- Global + Local Context: Summary nodes provide macro-level context; leaf nodes provide granular evidence.
- Results: Up to +20% accuracy on QuALITY, significant gains on NarrativeQA and QASPER.