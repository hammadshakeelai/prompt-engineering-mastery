# RAFT and RA-DIT: Retrieval-Augmented Fine-Tuning

## 1. RAFT (Zhang et al. / Gorman et al., 2024)
Trains LLMs for domain-specific "open-book" QA - extracting evidence from context while ignoring noise.
- Training Recipe: Each sample pairs question with oracle documents (containing answer) + distractor documents (that don't). In subset, oracle documents withheld.
- Model generates CoT answers citing verbatim excerpts from relevant docs.
- Builds robustness against retriever failures and hallucinations.

## 2. RA-DIT (Lin et al., 2023 - Meta AI)
Dual optimization of generator (LLM) and dense retriever without full end-to-end backpropagation.
- LM Tuning: Instruction-tunes LLM conditioned on retrieved passages.
- Retriever Tuning: Uses LM output distribution (KL-divergence feedback) to train dual-encoder retriever.

## 3. RAG-Specific vs. Standard Fine-Tuning
- Standard SFT: Forces factual knowledge into model weights (parametric memorization).
- RAG Fine-Tuning: Trains non-parametric extraction - how to search, ground, and cite from dynamic inputs.
- Distractor Tolerance: RAG tuning explicitly conditions on noisy/conflicting documents.
- Retriever Alignment: Co-adapts retriever distributions with LM context-conditioning behavior.