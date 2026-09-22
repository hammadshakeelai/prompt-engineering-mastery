# Prompt Chaining and Sequential Pipeline Design

Key References: Wu et al. PromptChainer (CHI 2022); Anthropic Building Effective Agents (2024)

## Decomposition
Complex tasks split into modular sequential subtasks: ingest -> outline -> draft -> critique -> verify.
Each prompt focuses on isolated sub-problem, avoiding context dilution.

## Handoff Patterns
- Direct Output-to-Input: Output of step N directly inputs to step N+1.
- State Accumulation: Shared state dict accumulates progressive outputs.
- Typed Schema Handoff: JSON/Pydantic schemas enforce interface contracts between nodes.

## Validation Gates
- Deterministic: Regex, type validation, Pydantic schema compliance.
- Semantic: LLM-as-judge evaluators for correctness and tone.
- Routing: Failed gates trigger retry loops or fallback paths.

## Frameworks
- LangChain: LCEL (prompt | model | parser), LangGraph for cyclic stateful pipelines.
- LlamaIndex: QueryPipeline (declarative DAG), Workflow with typed @step event handlers.

## Failure Modes
- Cascading hallucinations: mitigated by early-abort circuit breakers.
- Malformed outputs: recovered via OutputFixingParser with error feedback.