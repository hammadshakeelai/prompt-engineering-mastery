# Test-Time Compute Scaling: Sequential vs. Parallel

- **Sequential Scaling (Longer CoT)**: Depth scaling where models dynamically plan, backtrack, and self-correct across extended reasoning traces; excels at multi-step tasks with tight dependency tracking.
- **Parallel Scaling (Best-of-N)**: Breadth scaling generating multiple independent candidate solutions evaluated via PRMs/ORMs or consensus voting; low wall-clock latency, high coverage.
- **Pareto Frontier**: Optimal test-time compute dynamically combines both sequential depth and parallel breadth.