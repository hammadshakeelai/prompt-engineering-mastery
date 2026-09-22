# CrewAI: Hierarchical Process & Delegation Mechanics

- **Hierarchical Process (`Process.hierarchical`)**: Organizes agents into a corporate-style chain of command governed by a manager agent or `manager_llm`. The manager dynamically assigns tasks to specialists based on roles and skills, validating or requesting revisions on deliverables.
- **Top-Down Delegation**: Manager routes subtasks, tracks intermediate progress, and synthesizes final outputs.
- **Peer Delegation**: When `allow_delegation=True`, specialist agents invoke built-in tools (`Delegate work to co-worker`, `Ask question to co-worker`) to hand off subtasks or query colleagues for specialized knowledge.