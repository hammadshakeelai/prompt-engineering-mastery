# 01. ReAct & Autonomous Agent Control Loops

Autonomous agents transition LLMs from passive text predictors into interactive decision-making systems capable of observing an environment, reasoning over state transitions, invoking tools, and updating their internal world model.

---

## 1. ReAct: Synergizing Reasoning and Acting (Yao et al., 2022)

The ReAct framework unifies **Reasoning Traces** (Chain-of-Thought) and **Task-Specific Actions** into an interleaved execution loop:

$$\text{Trajectory: } \tau = (o_0, t_0, a_0, o_1, t_1, a_1, \dots, o_T, t_T, a_T)$$

Where:
- $o_t \in \mathcal{O}$ is the observation received from the external environment / tool execution.
- $t_t \in \mathcal{T}$ is the internal thought or reasoning step ("scratchpad").
- $a_t \in \mathcal{A}$ is the action dispatched to an external API or tool.

```
       +---------------------------------------------+
       | Environment / Tools / System State           |
       +---------------------------------------------+
             |                                ^
Observation  |                                | Action
             v                                |
       +-----------+                     +-----------+
       | Perception|                     | Execution |
       +-----------+                     +-----------+
             |                                ^
             v                                |
       +---------------------------------------------+
       | LLM Working Memory & Reasoning Trace (CoT)  |
       | - "Thought: What do I know so far?"         |
       | - "Thought: What is the next missing piece?"|
       +---------------------------------------------+
```

### Why ReAct Outperforms Act-Only and Reason-Only:
1. **Vs. Action-Only (e.g. naive WebGPT / Toolformer)**: Without explicit intermediate thoughts, the agent cannot synthesize multi-hop evidence, plan multi-step searches, or recover from unexpected tool errors.
2. **Vs. Reason-Only (e.g. static CoT)**: Reasoning without external grounding suffers from knowledge staleness, hallucinated API contracts, and inability to inspect real-time system state.

---

## 2. Production ReAct Prompt Template

```markdown
<system_identity>
You are an autonomous Cloud Infrastructure Diagnostic Agent.
Solve the incident by alternating between reasoning steps and tool executions.
</system_identity>

<tools_available>
1. get_pod_logs(pod_name: str, namespace: str, tail_lines: int) -> str
2. describe_service(service_name: str, namespace: str) -> dict
3. query_prometheus(promql: str, duration: str) -> dict
4. restart_deployment(deployment_name: str, namespace: str) -> bool
</tools_available>

<instruction_protocol>
For each turn, you MUST strictly follow this format:

Thought: [Analyze the current state, hypotheses, and what information is missing]
Action: tool_name(param1=val1, param2=val2)
Observation: [The execution result will be injected here by the orchestrator]

When you have collected sufficient evidence to resolve the incident, output:
Thought: [Final synthesis and root cause analysis]
Final Answer: [Actionable remediation summary and root cause]
</instruction_protocol>

<stopping_criteria>
- Never invent tool outputs. Always wait for the Observation block.
- Maximum iterations: 6.
- If a tool fails twice with the same parameters, change your diagnostic hypothesis.
</stopping_criteria>
```

---

## 3. Beyond ReAct: Plan-and-Solve & Reflexion

```
ReAct:          [Think] -> [Act] -> [Observe] -> [Think] -> [Act] -> [Observe] ...
Plan-and-Solve: [Decompose Plan: Steps 1..N] -> [Execute Step 1] -> [Execute Step 2] ...
Reflexion:      [Run Full Task] -> [Evaluate Failure] -> [Self-Reflect / Episodic Memory] -> [Retry]
```

### A. Plan-and-Solve (Wang et al., 2023)
For long-horizon tasks, ReAct can fall into myopic local loops. Plan-and-Solve enforces a two-tier cognitive architecture:
1. **Planner Agent**: Generates a DAG of sub-tasks.
2. **Executor Agent**: Operates step-by-step, taking sub-tasks and tool interfaces.
3. **Re-planner**: Evaluates progress after each step and dynamically adjusts remaining DAG branches.

### B. Reflexion: Verbal Reinforcement Learning (Shinn et al., 2023)
Reflexion endows agents with **episodic self-reflective memory** without parameter fine-tuning:
1. Agent attempts task trajectory $\tau$.
2. Environment evaluator computes scalar reward or error trace.
3. If task failed, a **Self-Reflection prompt** generates verbal feedback:
   `"I failed because I assumed the API token had admin scope. Next time, I must verify permissions via /auth/verify before executing write commands."`
4. The verbal reflection is appended to `<episodic_memory>` in the prompt for subsequent trials.

```markdown
<self_reflection_prompt>
You previously attempted the database migration task and failed with:
`Error: Deadlock detected while waiting for lock on relation 'orders'`.

Analyze your previous trajectory:
1. What incorrect assumption did you make?
2. What diagnostic step did you omit?
3. State a precise operational rule to prevent this error on the next attempt.
</self_reflection_prompt>
```

---

## 4. Language Agent Tree Search (LATS) (Zhou et al., 2023)

LATS combines the environmental grounding of ReAct, the memory of Reflexion, and the deliberative search of Monte Carlo Tree Search (MCTS):
- Explores multiple trajectory trees in parallel.
- Evaluates intermediate states using an LLM heuristic evaluator.
- Backpropagates success/failure reflections to guide tree expansion.
