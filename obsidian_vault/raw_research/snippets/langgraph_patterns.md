# LangGraph State Machine Patterns

LangGraph models workflows as cyclical state machines where nodes read and update shared, typed state:
- **Reducers**: Specify how state updates merge. Custom reducer functions (e.g., `operator.add` via `Annotated`) append or modify specific state keys.
- **Checkpointers**: Provide persistence by saving state snapshots at each superstep. Enables session continuity, fault tolerance, and time travel (rewinding to prior states).
- **Human-in-the-Loop**: Leverages checkpointers to pause execution via breakpoints or dynamic `interrupt()` calls for operator review, editing, or approval before resuming.