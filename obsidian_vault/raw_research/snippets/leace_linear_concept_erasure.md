# LEACE: Perfect Linear Concept Erasure (Belrose et al., NeurIPS 2023)

## 1. Limitations of Iterative Null-Space Projection
Iterative methods (INLP, RLACE) for removing sensitive concepts $Z \in \mathbb{R}^k$ from activation space $X \in \mathbb{R}^d$ suffer from incomplete concept erasure (residual leakage), uncontrolled representation distortion, and hyperparameter instability.

## 2. Closed-Form LEACE Formulation
Belrose et al. (*LEACE*, EleutherAI / NeurIPS 2023 / arXiv:2306.03819) derive the unique, closed-form affine transformation $P(x) = A x + b$ that provably enforces:
$$\operatorname{Cov}(P(X), Z) = 0$$
while minimizing expected Frobenius distance $\mathbb{E}[\|P(X) - X\|_2^2]$:
$$A = I - \Sigma_{XZ} \left( \Sigma_{XZ}^\top \Sigma_{XX}^{-1} \Sigma_{XZ} \right)^{-1} \Sigma_{XZ}^\top \Sigma_{XX}^{-1}, \quad b = (I - A)\mu_X$$
Equivalently:
$$P(x) = x - \Sigma_{XZ} \Sigma_{ZZ}^{-1} (z(x) - \mu_Z)$$
where $z(x)$ is the optimal linear least-squares prediction of $Z$ from $x$.

## 3. Optimality Theorem & Concept Scrubbing
- **Provable Linear Impossibility:** Guarantees that any linear probe achieves $R^2 = 0$ in predicting $Z$ from $P(X)$.
- **Layerwise Scrubbing:** Can be inserted into the residual stream across transformer layers ($h^{(l)} \leftarrow P^{(l)}(h^{(l)})$) to eliminate concepts (demographic attributes, part-of-speech syntax) with zero parameter gradient training and negligible language modeling perplexity degradation.
