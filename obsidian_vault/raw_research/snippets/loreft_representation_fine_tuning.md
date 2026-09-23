# LoReFT: Low-Rank Linear Subspace Representation Fine-Tuning (Wu et al., NeurIPS 2024)

## 1. Weight vs. Representation Interventions
Conventional PEFT (LoRA) modifies model weights globally across all sequence positions. Representation Fine-Tuning (ReFT) freezes all base model weights and applies localized causal interventions directly to hidden representation manifolds.

## 2. Low-Rank Subspace Intervention Formulation
LoReFT (Wu et al., Stanford / NeurIPS 2024 / arXiv:2404.03592) parameterizes hidden state intervention on $h \in \mathbb{R}^d$:
$$R_\Phi(h) = (I - R^\top R)h + R^\top (W h + b)$$
where $R \in \mathbb{R}^{r \times d}$ is an orthonormal projection matrix ($R R^\top = I_r$, $r \ll d$), and $W \in \mathbb{R}^{r \times d}, b \in \mathbb{R}^r$ are learned linear transformation parameters.
- **Orthogonal Subspace Invariance:** The projector $(I - R^\top R)$ leaves all representations orthogonal to the target subspace completely uncorrupted.
- **Subspace Steering:** The low-rank component $R^\top(Wh+b)$ steers task-specific causal circuits.

## 3. Parameter Efficiency & Performance
LoReFT tunes representations at sparse coordinate tuples $(l, p, \Phi)$, requiring **$10\times\text{--}50\times$ fewer parameters than LoRA** ($\sim 0.001\%\text{--}0.05\%$ of model weights) while matching or exceeding full LoRA performance on commonsense and instruction benchmarks.
