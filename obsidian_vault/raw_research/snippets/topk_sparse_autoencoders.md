# TopK Sparse Autoencoders & Exact Sparsity (Gao et al., OpenAI 2024 / ICLR 2025)

## 1. Failure Modes of $L_1$-Penalized SAEs
In traditional dictionary learning on transformer residual streams $x \in \mathbb{R}^d$:
$$\mathcal{L}_{L_1}(x) = \|x - \hat{x}\|_2^2 + \lambda \sum_{i=1}^M |f_i(x)|$$
- **Shrinkage Bias:** The constant subgradient $\nabla |f_i| = \text{sign}(f_i)$ systematically pulls true positive activations towards zero. To minimize $\lambda |f_i|$, the encoder suppresses feature magnitude, causing severe reconstruction error.
- **Dead Latent Waste:** Over-penalizing features with high $\lambda$ permanently deactivates large fractions ($>60\%$) of dictionary neurons.
- **Uncontrolled Sparsity ($L_0$ Variance):** A fixed $\lambda$ results in unstable $L_0 = \|f(x)\|_0$ across diverse input distributions.

## 2. TopK Activation Function & Pure MSE Loss
Gao et al. replace continuous $L_1$ shrinkage with a discrete top-$k$ projection operator:
$$f_i(x) = \text{TopK}(z, k)_i = \begin{cases} z_i & \text{if } z_i \in \text{top-}k(z) \text{ and } z_i > 0 \\ 0 & \text{otherwise} \end{cases}$$
where $z = W_{\text{enc}}(x - b_{\text{dec}}) + b_{\text{enc}} \in \mathbb{R}^M$.
- **Zero Shrinkage:** Sparsity is hard-bounded to $\|f(x)\|_0 \le k$ by construction. Training uses pure Mean Squared Error with no auxiliary regularizer:
  $$\mathcal{L}_{\text{TopK}}(x) = \|x - \hat{x}\|_2^2 = \|x - (W_{\text{dec}} f(x) + b_{\text{dec}})\|_2^2$$
- **Straight-Through Gradient Flow:** Selected top-$k$ coordinates receive unscaled identity gradients ($\frac{\partial f_i}{\partial z_i} = 1$), while unselected neurons receive zero gradient.

## 3. Scale & Downstream Steerability
- **Scaling Frontier:** Enabled training a **16-million latent** SAE on GPT-4 activations across 40 billion tokens without dead latent collapse.
- **Steering Precision:** Linear steering $x + \alpha W_{\text{dec}}[:, j]$ produces predictable, uncorrupted behavioral shifts because decoder columns preserve unattenuated physical magnitudes.
