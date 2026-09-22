# TextGrad: Automatic Differentiation via Text (Yuksekgonul et al., 2024)

- **Computation Graph Optimization**: Represents multi-component LLM workflows as PyTorch-like computation graphs where nodes represent variables (prompts, code, intermediate reasoning) and edges represent LLM calls. Prompts are tunable parameters.
- **LLM Backpropagation**: LLMs act as critics evaluating outputs against loss functions, generating structured qualitative critiques termed *textual gradients*. These gradients are propagated backward in reverse topological order.
- **Automatic Differentiation**: Aggregates textual gradients at target prompt variables; an LLM optimizer updates prompt text iteratively without weight access.