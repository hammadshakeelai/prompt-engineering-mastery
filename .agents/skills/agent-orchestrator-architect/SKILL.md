---
name: agent-orchestrator-architect
description: Specialized directive for designing, deploying, and auditing multi-agent orchestration architectures, dual-loop ledger systems (Magentic-One), cyclic graph state machines (LangGraph / Pregel), durable execution checkpointing, and dynamic subagent dispatch.
---

# Agent Orchestrator Architect Skill

Use this skill when designing enterprise-grade multi-agent swarms, orchestrating long-horizon workflows across heterogeneous tools (web, code, files), implementing dual-loop ledger patterns (Task Ledger & Progress Ledger), creating cyclic state graphs with durable checkpoints, or mitigating single-agent failure cascades.

## 1. Core Architectural Pillars

1. **Dual-Loop Separation (Outer Planning vs. Inner Execution):**
   - **Outer Loop (Strategic Orchestrator):** Manages high-level goal decomposition, monitors milestone completion, diagnoses structural failures, and triggers dynamic re-planning. Never directly calls low-level environment tools.
   - **Inner Loop (Specialized Subagents):** Specialized workers (`WebSurfer`, `FileSurfer`, `CoderTerminal`) operating within tightly constrained action spaces. Subagents filter raw environmental noise (HTML, terminal dumps) into concise semantic facts before returning control to the orchestrator.

2. **Dual Ledger Pattern (Magentic-One):**
   - **Task Ledger:** Tracks the dynamic plan $\langle g_{\text{root}}, \{s_1, \dots, s_K\}, i_{\text{active}} \rangle$ with step statuses (`Pending`, `In-Progress`, `Completed`, `Blocked`). Prevents goal drift.
   - **Progress Ledger:** Epistemic state tracking $\langle \mathcal{F}_{\text{verified}}, \mathcal{D}_{\text{dead\_ends}}, \mathcal{U}_{\text{uncertainties}} \rangle$. Explicitly registers failure points to forbid downstream agents from entering infinite retry loops on dead ends.

3. **Cyclic Graph State Machines & Durable Execution (LangGraph):**
   - Model long-running workflows as cyclic directed graphs $\mathcal{G} = \langle \mathcal{V}, \mathcal{E}, \mathcal{S} \rangle$ where nodes are pure functions or agent turns and edges are conditional state transitions.
   - Enforce **Durable Checkpointing**: Serialize state after every node execution to enable instantaneous disaster recovery, transactional rollback (time-travel debugging), and Human-in-the-Loop (HitL) action approvals.

## 2. Standard Orchestration State Schema

When implementing agent graphs, enforce the following structured state contract:

```python
from typing import TypedDict, List, Dict, Any, Optional

class AgentOrchestrationState(TypedDict):
    # Core User Goal
    task_goal: str
    
    # Task Ledger (Plan)
    plan_steps: List[Dict[str, Any]]  # [{"id": 1, "desc": "...", "status": "completed"}]
    current_step_id: int
    
    # Progress Ledger (Epistemic Facts & Dead Ends)
    verified_facts: List[str]
    dead_ends: List[str]
    active_uncertainties: List[str]
    
    # Context & Artifact Storage
    shared_artifacts: Dict[str, Any]
    subagent_scratchpad: Optional[str]
    
    # System Execution Metrics
    iteration_count: int
    error_retries: int
```

## 3. Subagent Routing & Context Sanitization Rules

1. **Strict Context Isolation:**
   - Raw DOM trees, binary file dumps, and multi-page compiler tracebacks must never be emitted into the shared Orchestrator context. Subagents must summarize observations into a maximum of 3–5 bullet points.
2. **Action Space Partitioning:**
   - Cap each subagent's toolset at $\le 5$ focused tools to maximize parameter extraction accuracy and eliminate schema confusion.
3. **Dead-End Quarantine:**
   - When a subagent encounters an unrecoverable failure (e.g., authentication required, 404 resource missing), it must immediately log the exact failure signature into `dead_ends` and yield control back to the Orchestrator for alternative routing.

## 4. Benchmarking & Verification Directives

- Validate orchestration workflows against multi-step benchmarks (**GAIA**, **AssistantBench**, **SWE-bench**) assessing autonomous recovery rates, plan revision quality, and token efficiency.
- Measure dead-end avoidance: ensure agents never re-execute an identical failed tool invocation within the same session.
