# Agent Sleep-Replay & Episodic-to-Semantic Memory Consolidation (2024–2026)

## 1. The Finite Context Bottleneck in Autonomous Agents
Autonomous agents executing prolonged workflows accumulate massive trajectory logs $(s_t, a_t, r_t, o_t)$. Storing infinite raw episodic logs leads to context bloating, semantic interference, and catastrophic retrieval drift.

Modern architectures draw on **Complementary Learning Systems (CLS)** theory to structure dual-memory agent systems:
- **Hippocampal Episodic Buffer:** A fast, high-fidelity short-term memory store holding recent task observations, intermediate tool outputs, and conversational context.
- **Neocortical Semantic Store:** A slow, compact, generalized knowledge store (concept graphs, procedural heuristics, verified system rules).

## 2. The Offline "Sleep" & Replay Pipeline
During scheduled idle or "sleep" cycles, agents decouple from external execution to consolidate episodic experiences into permanent semantic knowledge:
1. **Surprise-Driven Salience Filtering:**
   Episodes are scored by informational surprisal $\mathcal{S}(e) = -\log p(o_t \mid s_t, a_t)$ or prediction error. Highly predictable, routine actions are evicted, while surprising successes or failures are flagged for replay.
2. **Generative Replay ("Active Dreaming"):**
   The agent replays flagged trajectory segments through synthetic counterfactual simulations, testing whether extracted lessons hold under alternate conditions.
3. **Graph Distillation & Rule Induction:**
   Verified patterns are distilled into structured triple updates $(E_{\text{sub}}, R, E_{\text{obj}})$ within an episodic knowledge graph or appended to reusable skill libraries.

## 3. Bidirectional Consolidation Dynamics
- **Episodic $\to$ Semantic:** Continuous distillation of specific, dated experiences into timeless heuristics and behavioral guardrails.
- **Semantic $\to$ Episodic:** Consolidated semantic knowledge acts as a high-level heuristic scaffold, constraining future search spaces and guiding multi-agent task planning.
