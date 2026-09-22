# Multi-Agent Orchestration Frameworks: AutoGen, CrewAI, and LangGraph

## 1. AutoGen
**Conversation Patterns:**
AutoGen builds its multi-agent systems primarily around conversational patterns. Every interaction (task delegation, tool execution, decision-making) is treated as a message exchange. It supports several topologies:
*   **Two-Way Chat:** A simple direct exchange between two agents or an agent and a human.
*   **Group Chat:** Multiple agents in a shared thread, typically using a "manager" agent or dynamic speaker selection logic to determine who speaks next.
*   **Sequential Chat:** A pipeline of two-agent chats where the output of one chat is passed to the next.
*   **Nested Chat:** Encapsulated workflows where an agent initiates an "inner" conversation with others to solve a sub-task and brings the result back to the main chat.

**Agent Memory:**
Standard AutoGen agents are stateless across sessions. Memory is managed via:
*   **Built-in Memory:** Components like `ListMemory` append contextual facts to the context window.
*   **Persistent Memory:** Frequently integrated with vector databases (e.g., Chroma, Pinecone) or external memory frameworks (e.g., MemGPT).
*   **Tool-based Memory:** Agents are given explicit tools to read/write persistent state.

**Tool Routing:**
Assistant agents can be registered with specific functions. When an LLM decides a tool is needed, it generates a structured message that a `UserProxyAgent` or a dedicated executor agent runs. Routing is typically handled by orchestrator/manager agents inspecting messages, condition-based token triggers (e.g., `[ESCALATE]`), or dynamic selection in group chats.

---

## 2. CrewAI
**Role-Based Crew Hierarchies:**
CrewAI focuses on a role-based design where agents act as "new hires." Each agent is defined by a Role, Goal, and Backstory, operating within strict boundaries. The framework orchestrates these agents in specific ways:
*   **Sequential Process:** Tasks run sequentially, passing outputs as inputs to the next.
*   **Hierarchical Process:** A manager agent dynamically delegates tasks to the crew, validates results, and ensures a structured chain of command.
*   **Flows:** For complex workflows, event-driven orchestration manages multiple crews, state, and conditional routing.

**Agent Memory:**
CrewAI implements a sophisticated multi-layered memory system:
*   **Short-Term Memory:** Uses vector databases for recent interactions.
*   **Long-Term Memory:** Uses relational databases (e.g., SQLite) to persist insights and knowledge across sessions.
*   **Entity Memory:** Tracks specific subjects (people, places, concepts) across tasks.

**Tool Routing:**
Agents dynamically decide which tools to use based on the task and tool descriptions. Tools are explicitly bound to agents or tasks. Because of its role-based nature, routing is implicitly managed by the manager agent (in hierarchical mode) or the rigid sequence (in sequential mode), ensuring only specialized agents execute their designated tools.

---

## 3. LangGraph
**Stateful Graph Execution Model:**
LangGraph models multi-agent workflows as stateful graphs. Unlike DAGs, it supports cycles, which are critical for iterative reasoning, self-correction, and planning.
*   **Nodes:** Functions representing agent actions, tool calls, or logic.
*   **Edges:** Transitions between nodes, which can be conditional (routing based on state).
*   **State:** A shared, typed data structure (e.g., `TypedDict`) passed through the graph, serving as the central source of truth.

**Agent Memory:**
LangGraph utilizes built-in **checkpointers** to save the state at each step (node execution). This state persistence provides:
*   Short-term memory (within the execution cycle).
*   Long-term memory across sessions.
*   "Time-travel" and debugging capabilities, as well as easy pausing/resuming for Human-in-the-Loop workflows.

**Tool Routing:**
Tool execution is handled via specific graph nodes. An agent node decides to call a tool, the graph conditionally routes execution to a tool-executing node based on that decision, and the result is fed back into the state. In multi-agent scenarios, a Supervisor agent often acts as the routing node, inspecting the state and conditionally passing execution control to specialist sub-agent nodes.
