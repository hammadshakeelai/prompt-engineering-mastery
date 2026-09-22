# Event-Driven Actor Model & Swarm Handoff Protocols (2024–2026)

## 1. Limitations of Conversational Multi-Agent Loops
First-generation multi-agent systems (e.g., AutoGen 0.2, CrewAI) rely on synchronous turn-taking in shared chat rooms. This introduces severe systemic bottlenecks:
- **Thread Blocking:** Execution is strictly serial; parallel worker execution is stalled by slow upstream token generation.
- **Context Pollution:** Transcripts grow as $\mathcal{O}(N \cdot T)$, flooding downstream agents with irrelevant scratchpads and degrading retrieval attention.
- **Router Hallucination:** Central orchestrators repeatedly make brittle global routing decisions.

## 2. Asynchronous Actor Model & Event Buses (AutoGen 0.4 / AG2)
Modern systems decouple agents into independent **actors** with private state and asynchronous mailboxes:
- **Topic Pub-Sub:** Actors publish typed events (`ArtifactProposed`, `TestFailed`) to an event bus. Specialized agents subscribe to topic filters, reacting asynchronously.
- **Distributed Concurrency:** Workload distribution across processes, containers, or remote microservices without shared memory locks.

## 3. Dynamic Handoff Protocols (OpenAI Swarm)
Decentralized swarms replace central routers with first-class **handoff tool functions**:
- An active agent delegates execution by invoking `transfer_to_<agent>()`.
- **Context Hygiene:** The receiving agent does not inherit the sender's noisy trajectory. It receives only its private system prompt, the user objective, and a structured transfer payload: $\text{Context}_{\text{recv}} = \mathcal{O}(1)$.

## 4. StateGraph Checkpointing (LangGraph)
For production-grade determinism:
- Agents operate as nodes in cyclic directed graphs $(V, E)$ updating state via reducer functions.
- State transitions are recorded in Write-Ahead Logs (WAL), enabling human-in-the-loop interruption, fault recovery, and time-travel replay without re-executing completed LLM invocations.
