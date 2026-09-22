# Research: Prompt Compression and Context Engineering

## 1. LLMLingua-2: Selective Token Pruning
**LLMLingua-2** is a task-agnostic prompt compression method designed to reduce the size of LLM prompts while maintaining semantic integrity.
* **Token Classification:** Unlike earlier methods that relied on information entropy (perplexity) from causal language models, LLMLingua-2 formulates prompt compression as a token classification problem.
* **Bidirectional Context:** It uses a Transformer-based encoder (e.g., XLM-RoBERTa-large or mBERT) to process the full bidirectional context of a prompt, allowing the model to make more informed decisions about which tokens are essential.
* **Data Distillation:** It employs a data distillation procedure to derive knowledge from a powerful LLM (like GPT-4) to create a high-quality extractive compression dataset.
* **Empirical Results:** It achieves 2x–5x compression ratios, performs 3x–6x faster than entropy-based methods, and enables end-to-end latency reductions of 1.6x–2.9x, saving GPU memory and API costs.

## 2. AutoCompressor: Recursive Summarization
**AutoCompressor** is a model-based soft prompt compression technique for handling long context windows.
* **Soft Tokens (Vectors):** Instead of truncating or manually summarizing text, it processes a document in chunks and compresses information into fixed-length, continuous embeddings (virtual tokens) in the model's latent space.
* **Recursive Process:** For extremely long inputs, it compresses data hierarchically. If the resulting vectors are still too large, it performs further compression on those summary vectors.
* **Comparison to Text Summarization:** While traditional recursive summarization yields human-readable text (but loses fine-grained entropy), AutoCompressor yields highly dense machine-code vectors that can achieve massive compression ratios (4x–480x) when specialized models are used.

## 3. Gist Tokens
**Gist tokens** substitute lengthy system prompts or instructions with a much smaller, learned set of tokens.
* **Mechanism:** Special "gist" tokens are added to the model's vocabulary. The model is trained via knowledge distillation to map a long, natural-language prompt to these specific tokens (the "teacher" uses the full prompt; the "student" produces the same output logits from just the gist tokens).
* **Caching & Reuse:** Once trained, these gist tokens are cached. They can act exactly as the original instructions but require a fraction of the computational load.
* **Empirical Results:** It can achieve up to 26x compression rates (e.g., condensing a 6,000-token prompt into 1,500 gist tokens), significantly lowering the "time to first token" (TTFT) and increasing Queries Per Second (QPS).

## 4. RECOMP: Abstractive Compression
**RECOMP (Retrieve, Compress, Prepend)** is a framework designed to compress retrieved documents before they are fed into Retrieval-Augmented Language Models (RALMs).
* **Abstractive Compressor:** Uses a trained encoder-decoder model (e.g., T5-Large) to synthesize information from multiple retrieved documents into a concise, readable summary. It rewrites content to preserve intent while drastically reducing token count.
* **Extractive Compressor:** Alternatively, RECOMP can select the most relevant sentences directly.
* **Selective Augmentation:** It acts as a noise filter, capable of returning an empty string if retrieved documents contain no relevant information, preventing context dilution and hallucination.

## 5. "Lost in the Middle" Phenomenon (Liu et al., 2023)
Identified by Nelson F. Liu et al., this phenomenon describes how LLMs struggle to retrieve or utilize information located in the **middle** of their input context window.
* **U-Shaped Performance Curve:** Accuracy on multi-document QA and key-value retrieval is highest when the target information is at the beginning (primacy effect) or the end (recency effect) of the prompt, with a massive degradation in the middle.
* **Model Robustness:** This degradation was observed even in models explicitly designed and marketed for ultra-long context windows.
* **Practical Implications for RAG:** In Retrieval-Augmented Generation, if the most relevant document chunk is placed in the center of the retrieved context, the LLM is statistically likely to overlook it.
* **Root Causes:** This positional bias is thought to stem from autoregressive attention dynamics and pre-training objectives, where models adapt to prioritize recent tokens and structural context at the boundaries.
