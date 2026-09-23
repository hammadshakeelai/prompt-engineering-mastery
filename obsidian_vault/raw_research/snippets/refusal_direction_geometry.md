# Refusal Direction Geometry & Representation Erasure (Arditi et al., ICML 2024)

## 1. Linear Collapse of Post-Hoc Alignment
Safety alignment methodologies such as RLHF, DPO, and KTO train language models to refuse harmful requests. Mechanistic investigations by Andy Arditi et al. (*Refusal in Language Models Is Mediated by a Single Direction*, ICML 2024 / arXiv:2406.11717) demonstrate that despite complex multi-layer architectures, refusal behavior collapses into a **single one-dimensional linear subspace** within the model's residual stream.

## 2. Mathematical Formulation & Causal Intervention

### Difference-in-Means Extraction
Given contrastive datasets of harmful requests $\mathcal{D}_{\text{harm}}$ and harmless requests $\mathcal{D}_{\text{harmless}}$, the empirical refusal vector at layer $l$ for final prompt token activations $h_l(x) \in \mathbb{R}^d$ is:
$$\mathbf{r}_l = \frac{1}{|\mathcal{D}_{\text{harm}}|} \sum_{x \in \mathcal{D}_{\text{harm}}} h_l(x) - \frac{1}{|\mathcal{D}_{\text{harmless}}|} \sum_{x \in \mathcal{D}_{\text{harmless}}} h_l(x)$$
Normalized to unit length:
$$\hat{\mathbf{r}}_l = \frac{\mathbf{r}_l}{\|\mathbf{r}_l\|_2}$$

### Dual Causal Validation
1. **Activation Addition (Inducing Refusal):** Adding $\hat{\mathbf{r}}_l$ with scalar coefficient $c > 0$:
   $$h_l' = h_l + c \cdot \hat{\mathbf{r}}_l$$
   induces benign requests to be rejected with standard refusal phrasing (`"I cannot fulfill this request..."`).
2. **Orthogonal Null-Space Projection (Erasing Refusal):** Projecting activations onto the orthogonal complement:
   $$h_l' = \left( I - \hat{\mathbf{r}}_l \hat{\mathbf{r}}_l^\top \right) h_l$$
   completely bypasses refusal across 13 open-weight model families (LLaMA-2/3, Mistral, Qwen, Yi up to 72B parameters) without retraining.

### Zero-Overhead Weight Ablation
Because $(I - \hat{\mathbf{r}}_l \hat{\mathbf{r}}_l^\top)$ is linear, it can be folded directly into transformer projection weights offline:
$$W_{\text{down}}^{(l)\prime} = \left( I - \hat{\mathbf{r}}_l \hat{\mathbf{r}}_l^\top \right) W_{\text{down}}^{(l)}, \quad W_O^{(l)\prime} = \left( I - \hat{\mathbf{r}}_l \hat{\mathbf{r}}_l^\top \right) W_O^{(l)}$$
yielding an ablated model that physically cannot write activation mass along the refusal axis.

## 3. Defense Frontier: Representation Circuit Breakers
The 1D linear collapse proves that post-hoc preference optimization constructs a superficial linear mask. Robust alignment requires [[representation_circuit_breakers|Representation Circuit Breakers]] (Zou et al., NeurIPS 2024), which disrupt high-dimensional internal representation trajectories when safety boundaries are crossed.
