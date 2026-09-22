# Promptbreeder: Self-Referential Prompt Evolution (Fernando et al., 2023)

- **Co-Evolution**: Evolves task-prompts coupled with meta-level mutation-prompts simultaneously. Mutation-prompts undergo "hypermutation," refining the optimizer itself.
- **Evolutionary Strategies**: Population-based genetic algorithm driven by binary tournament selection, mutation operators, and estimation-of-distribution algorithms.
- **Fitness Landscapes**: Traverses rugged, non-differentiable discrete prompt spaces by dynamically adapting mutation operators to landscape geometry.