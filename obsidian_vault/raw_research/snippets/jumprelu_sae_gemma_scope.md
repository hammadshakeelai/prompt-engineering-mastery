# JumpReLU SAEs & Heaviside-Gated Feature Steering (Gemma Scope, Google DeepMind 2024)

## 1. Shrinkage Bias in $L_1$-Regularized SAEs
Standard Sparse Autoencoders use $L_1$ penalties $\|f\|_1$ to induce sparsity. The constant gradient penalty $\frac{\partial \mathcal{L}}{\partial f_i} \propto \lambda \operatorname{sign}(f_i)$ systematically penalizes large activations, causing shrinkage bias that distorts reconstructed feature magnitudes and corrupts activation steering interventions.

## 2. JumpReLU Mathematical Formulation
Lieberum et al. (*Gemma Scope*, Google DeepMind 2024 / arXiv:2408.05147) formulate the JumpReLU activation function with learned threshold vector $\theta \in \mathbb{R}_+^m$:
$$\operatorname{JumpReLU}_\theta(z) = z \odot H(z - \theta) = \begin{cases} z_i & \text{if } z_i > \theta_i \\ 0 & \text{if } z_i \le \theta_i \end{cases}$$
where $H$ is the Heaviside step function.
- **Unbiased Magnitudes:** Active features ($z_i > \theta_i$) preserve their exact linear activation scale $f_i = z_i$ without shrinkage penalty.
- **Direct $L_0$ Objective:** Enables exact optimization of the active feature count $\mathcal{L}_{L_0} = \lambda \sum_{i=1}^m H(z_i - \theta_i)$.

## 3. Straight-Through Estimators (STE)
Subgradient updates through the discontinuous step function use a bandwidth-controlled rectangular window:
$$\frac{\partial H(z - \theta)}{\partial z} \approx \frac{1}{\epsilon} \operatorname{rect}\left(\frac{z - \theta}{\epsilon}\right)$$
allowing gradients to update threshold $\theta_i$ when activations hover near the boundary.

## 4. Gemma Scope Empirical Suite
Provides $>400$ JumpReLU SAEs across all layers of Gemma-2-2B and Gemma-2-9B (up to $163\text{k}$ latents per layer), establishing the Pareto frontier in reconstruction fidelity vs. $L_0$ sparsity for monosemantic mechanistic steering.
