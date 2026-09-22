# Plan-and-Solve Prompting (Wang et al. 2023)

- **Two-Phase Cognitive Scaffolding**: Replaces generic Zero-Shot-CoT with explicit instruction: *"Let's first understand the problem and devise a plan... then carry out the plan..."*
- **Plan Generation**: Decomposes problems into an ordered sequence of sub-tasks without few-shot exemplars.
- **Plan Execution**: Sequentially carries out sub-tasks, preventing missing-step and calculation errors. PS+ adds variable extraction directives.