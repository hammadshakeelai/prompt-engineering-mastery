# ReAct: Reason + Act Trajectory Formatting

- **Cycle**: Strict interleaving of **Thought -> Action -> Observation**:
  - `Thought`: Formulates subgoals, tracks state, and evaluates progress.
  - `Action`: Executable tool call syntax (e.g., `Search[query]`, `Lookup[term]`).
  - `Observation`: External execution output grounding next thought.
- **Termination**: Concludes when action `Finish[answer]` returns final output.