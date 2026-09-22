---
name: kv-cache-optimizer
description: Specialized directive for prompt caching, token economy, RadixAttention tree-prefix caching, FlashAttention-3 chunked prefill, context compression (LLMLingua-2), SnapKV/PyramidKV eviction, and Cross-Model KV transfer.
---

# KV Cache Optimizer Skill

Use this skill when designing high-throughput LLM architectures, multi-turn agent memory pipelines, or latency-critical prompt engineering workflows that interact with GPU KV-cache budgets.

## 1. Prefix Caching & Attention Topologies

1. **RadixAttention & Tree-Prefix Sharing:**
   - Structure complex system prompts and few-shot exemplars with deterministic, static prefixes placed at the beginning of the prompt.
   - Avoid interleaving dynamic timestamps or session IDs at the start of prompt templates; push all volatile parameters to the terminal user turn to maximize Radix tree cache hits (up to 95%+ prefill savings).

2. **Chunked Prefill & Decode Piggybacking:**
   - Interleave long prompt prefill computation in discrete chunks (e.g., 512–2048 tokens) across concurrent decode steps (Sarathi-Serve / vLLM) to mitigate TTFT (Time-To-First-Token) latency spikes and eliminate GPU compute underutilization.

3. **Multi-Head Latent Attention (MLA) Architecture:**
   - Compress Key-Value projections into low-rank latent spaces: $c_t^{KV} = W_{DKV} h_t$, drastically reducing KV cache memory footprint to a fraction of standard MHA/GQA without losing reasoning fidelity.

## 2. Context & Prompt Compression Directives

1. **Bidirectional Prompt Compression (LLMLingua-2):**
   - Use bidirectional encoder-based token filtering rather than unidirectional causal language models to calculate importance, preventing the premature eviction of early foundational entities.
   - Target 50%–70% token reductions for lengthy retrieved context chunks (RAG) while preserving exact entity relationships and numerical values.

2. **Dynamic KV Eviction (SnapKV & PyramidKV):**
   - Apply layer-adaptive KV retention budgets: maintain larger observation windows in lower layers where broad attention pooling occurs, and compact, focused window budgets in higher layers where feature abstraction dominates.
   - Protect critical "attention sink" tokens ($[0:4]$ initial tokens) to preserve stable softmax normalization.

## 3. Cross-Model KV Transfer Directives

1. **Heterogeneous Draft-Target Caching:**
   - Utilize linear or shallow non-linear alignment matrices $W_{\text{trans}}$ to project KV activations from small, high-throughput draft models directly into target model key-value spaces, avoiding duplicate prefill overhead in speculative decoding pipelines.
