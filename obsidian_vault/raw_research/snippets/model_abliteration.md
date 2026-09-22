# Model Abliteration: Refusal Erasure via Orthogonal Projection (Arditi et al., 2024)

- **Mechanism**: Contrasts residual stream activations between harmful and harmless prompts to isolate a low-dimensional "refusal vector" via difference-in-means.
- **Weight Modification**: Modifies weight matrices (attention output and MLP down-projections) by orthogonally projecting out the refusal direction:
  $$W' = W - \hat{r}\hat{r}^T W$$
- **Impact**: Permanently prevents the model from writing or propagating refusal representations during forward passes without gradient descent or fine-tuning, preserving general reasoning performance.