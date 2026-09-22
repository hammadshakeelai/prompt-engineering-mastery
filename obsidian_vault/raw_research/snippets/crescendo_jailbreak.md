# Crescendo Multi-Turn Alignment Bypass

- **Mechanism**: Initiates a seemingly benign conversation on an adjacent topic, gradually escalating the request across successive turns.
- **Exploitation**: Leverages the model's objective to remain helpful, coherent, and consistent with earlier dialogue context.
- **Bypass**: Keeps each individual turn within safety thresholds until cumulative context causes the model to override alignment guardrails.