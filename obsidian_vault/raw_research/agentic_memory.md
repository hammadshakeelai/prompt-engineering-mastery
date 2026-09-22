# Agentic Memory Systems: Research Findings

## 1. MemGPT's Virtual Context Management
MemGPT applies principles from traditional operating systems (like virtual memory and paging) to manage an LLM's context window. It treats the limited context window as "RAM" and external storage (like vector databases) as "disk space." The LLM acts as the OS, autonomously deciding when to page data in and out using function calls (e.g., search, read, write). This provides the illusion of an unbounded context, allowing agents to maintain persistence over long interactions.

## 2. Zep's Temporal Knowledge Graphs
Zep utilizes a temporal knowledge graph architecture (via its Graphiti engine) to provide long-term, context-aware memory. It tracks the "validity window" of facts (when a fact became true and when it ceased to be true). This bi-temporal modeling (tracking both real-world validity time and ingestion time) prevents agents from acting on stale or contradictory information, significantly improving accuracy in complex temporal reasoning compared to standard RAG setups.

## 3. Vector Store-Based Episodic Memory
Episodic memory captures the "who, what, when, and where" of an agent's history (specific events, decisions, tool outcomes). Vector stores enable this by converting logs into embeddings. When a new situation arises, the agent performs a similarity search to recall how it handled similar tasks in the past. This contextual retrieval reduces token usage and latency by pulling only relevant past experiences rather than processing raw logs.

## 4. Semantic Memory with Entity Extraction
Semantic memory stores generalized facts, domain concepts, and user preferences independent of specific past events. A core part of this is entity extraction, where an LLM analyzes interactions to distill key information (e.g., "User prefers Python"). These extracted entities are typically stored in knowledge graphs or vector databases. Systems like Mem0, LangMem, and Letta manage these facts, updating them and resolving conflicts so the agent maintains accurate, stateful context across sessions.

## 5. Procedural Memory via Tool Definitions
Procedural memory represents the "how-to" layer—the encoded instructions, workflows, and tool-use patterns. It acts as the agent's "muscle memory," allowing it to execute tasks efficiently. Successful tool-use sequences are captured as parameterized workflows. Instead of re-planning from scratch, the agent retrieves these templates and fills in the variables. This reduces cognitive load and latency while improving reliability as the agent refines its skills based on feedback.

## 6. Reflexion: Verbal Reinforcement Learning with Episodic Memory
Reflexion is a framework for verbal reinforcement learning where agents learn through trial and error without model weight updates. Instead of numerical rewards, the agent receives or generates natural language feedback (self-reflection) on its past trajectories, identifying logic errors or successes. This reflective feedback is stored in an episodic memory buffer. During subsequent attempts, the agent retrieves these insights to avoid repeating mistakes, enabling a form of self-evolution.
