---
name: agentic-memory-architect
description: Specialized directive for autonomous agent memory topologies, episodic vs semantic vs procedural memory, hierarchical KV offloading (Mooncake, MemOS), cognitive reflection trees (Generative Agents), virtual memory paging (MemGPT), and neural long-term memory (Titans).
---

# Agentic Memory Architect Skill

Use this skill when designing long-horizon autonomous agents, multi-turn stateful conversational systems, persistent retrieval architectures, or hierarchical memory engines.

## 1. Cognitive Memory Taxonomies

1. **Episodic Memory (Stream of Experience):**
   - Maintain time-indexed, chronological observation records containing raw sensory observations, action invocations, and environment feedback.
   - Calculate retrieval scores via tripartite weighting:
     $$\text{Score}(m) = \alpha_{\text{recency}} \cdot \text{Recency}(m) + \alpha_{\text{importance}} \cdot \text{Importance}(m) + \alpha_{\text{relevance}} \cdot \text{Relevance}(m, q)$$
   - Decay recency exponentially: $\text{Recency}(m) = \exp(-\lambda \Delta t)$.

2. **Semantic & Reflection Memory (Generative Agents):**
   - Synthesize periodic abstract reflections when cumulative observation importance crosses threshold $\tau_{\text{reflect}}$.
   - Organize insights in a directed acyclic reflection graph (DAG) where nodes represent high-level beliefs and edges ground justifications to lower-level episodic leaves.

3. **Procedural Memory (Tool & Skill Synthesis):**
   - Store verified code snippets, API workflows, and validated action execution scripts in an execution cache (Voyager / Code-as-Policies pattern).
   - Key procedural entries by semantic task embeddings and preconditions; index deterministic tool calls to bypass expensive LLM planning steps.

## 2. Operating System & Virtual Memory Abstractions (MemGPT / MemOS)

1. **Multi-Tier Context Hierarchy:**
   - **Main Context (GPU Working Context):** Limited by model context window ($L_{\text{window}}$); houses the active system prompt, working memory scratchpad, and immediate turn buffer.
   - **External Context (Disk / Vector Database):** Houses archival storage (unbounded vector search) and recall storage (searchable history log).
   - **Memory Paging Protocol:** Implement explicit tool calls (`core_memory_append`, `archival_memory_insert`, `archival_memory_search`) enabling the LLM agent to page historical context in and out of working memory without external heuristic truncation.

2. **Sleep & Memory Consolidation:**
   - Implement offline asynchronous memory consolidation cycles: during idle periods, prune redundant episodic tokens, resolve conflicting factual beliefs, and compress conversational dialogues into structured knowledge triples.

## 3. Disaggregated & Neural KV Cache Offloading (Mooncake / Titans)

1. **Hierarchical KV Cache Tiering:**
   - Offload inactive agent conversation KV states across a three-tier hierarchy: GPU HBM $\leftrightarrow$ Host CPU DRAM (CXL) $\leftrightarrow$ Distributed NVMe SSDs.
   - Re-inject hot agent prefixes via zero-copy RDMA over Converged Ethernet (RoCEv2), eliminating redundant prefill computation across agent turns.

2. **Neural Long-Term Memory (Titans / DeepSeek MTP):**
   - Complement standard KV caches with neural memory modules trained via online gradient descent:
     $$M_t = (1 - \alpha_t) M_{t-1} + \eta_t \nabla \mathcal{L}(h_t)$$
     storing unbounded sequential dependencies in compressed recurrent matrix states.
