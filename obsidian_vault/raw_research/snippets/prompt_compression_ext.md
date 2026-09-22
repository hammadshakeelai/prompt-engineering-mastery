# Prompt Compression Techniques

## LLMLingua-2 (Pan et al., 2024)
Reframes prompt compression as binary token classification (preserve vs. discard).
Distills compression labels from GPT-4 to train a small bidirectional Transformer encoder (XLM-RoBERTa).
Full bidirectional context for faithful, low-latency, task-agnostic extractive pruning.

## RECOMP Abstractive Compression (Xu et al., ICLR 2024)
Fine-tuned T5-large trained via distillation to synthesize multiple retrieved RAG passages into concise query-focused summary.
Mitigates lost-in-the-middle degradation.
Can output empty string to filter completely irrelevant passages.

## AutoCompressor Soft Tokens (Chevalier et al., 2023)
Extends vocabulary with <Sum> summary tokens that recursively compress text chunks into dense summary vectors.
Cached vectors preserve long-horizon context without storing raw token sequences.

## Gist Tokens (Mu et al., NeurIPS 2023)
Condenses entire natural language instructions into small fixed set of virtual gist tokens.
Modified attention masks during training force model to attend through bottleneck gist tokens.
Enables up to 26x context compression with reusable, cached activation prefixes.