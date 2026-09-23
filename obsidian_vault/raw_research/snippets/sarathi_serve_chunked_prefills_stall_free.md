# Sarathi-Serve: Chunked-Prefills & Stall-Free Pipeline Scheduling

## 1. The Prefill-Decode Latency Conflict
In continuous batching engines, incoming long prompts trigger monolithic **prefill** executions (compute-bound) that stall active **decode** requests (memory-bound) for hundreds of milliseconds. This causes P99 Time-Per-Output-Token (TPOT) to spike catastrophically, freezing streaming interfaces.

```mermaid
flowchart LR
    Prompt["Incoming 4k Prompt"] --> Chunker["Chunked-Prefill: C = 512 tokens"]
    Chunker --> HybridBatch["Hybrid Batch: 1 Prefill Chunk (512) + 32 Decodes (32)"]
    HybridBatch --> Speedup["Piggybacking: Decodes ride on compute for 0 extra latency (5x lower P99 TPOT)"]
```

## 2. Core Architectural Mechanisms (USENIX OSDI 2024)
- **Chunked-Prefills**: Decomposes long prompts into equal slices of $C=512$ tokens, populating KV caches incrementally without starving concurrent requests.
- **Piggybacked Decodes**: Combines a compute-heavy prefill chunk with memory-heavy decode requests in a single batch. Decodes "piggyback" on the Tensor Core matrix multiplications of the prefill chunk at near-zero incremental latency.
- **Stall-Free Scheduling**: Eliminates pipeline bubbles in multi-GPU pipeline-parallel serving (Falcon-180B), delivering **$2.6\times\text{--}6.9\times$ serving capacity uplifts** and reducing P99 TPOT jitter by up to **$5.0\times$**.
