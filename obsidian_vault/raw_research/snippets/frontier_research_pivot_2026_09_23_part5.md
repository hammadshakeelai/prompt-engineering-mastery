# Frontier Research Pivot Directives (Phase V - 2026-09-23)

## 1. Executive Summary & Frontier Direction
This research pivot expands beyond single-node inference and token-by-token heuristics, establishing three novel theoretical frontiers uniting test-time diffusion reasoning, geometric model surgery, and manifold epistemic calibration.

---

## 2. Pillar 1: Monte Carlo Tree Diffusion (MCTD) for Non-Autoregressive Reasoning
- **Problem Formulation:** Standard Test-Time Compute (TTC) scaling utilizes Process Reward Model (PRM) guided search over discrete token branches (MCTS). However, discrete token trees suffer from step-horizon myopic errors where early token commitments catastrophically constrain downstream exploration.
- **Proposed Frontier Formulation:** Formulate reasoning as continuous diffusion trajectories over latent thought embeddings $z_t \in \mathbb{R}^d$. Rather than sampling discrete tokens $y_k \sim P(y \mid y_{<k})$, Monte Carlo Tree Diffusion (MCTD) conducts continuous-time Langevin updates conditioned on PRM energy gradients:
  $$dz_t = \left[ f(z_t, t) - g(t)^2 \nabla_z \log p_t(z_t) + \alpha g(t)^2 \nabla_z \mathcal{R}_{\text{PRM}}(z_t) \right] dt + g(t) dw$$
- **Actionable Directive:** Develop a hybrid discrete-continuous verifier harness where discrete nodes are expanded by discrete score-based diffusion language models (SEDD) and pruned via latent energy gradient descent.

---

## 3. Pillar 2: Subspace Projective Model Abliteration: Multi-Dimensional Refusal Null-Space Surgery
- **Problem Formulation:** Existing model abliteration (Arditi et al., 2024) computes a single difference-of-means vector $v_{\text{refusal}} = \mu_{\text{harmful}} - \mu_{\text{harmless}}$ and projects it out of weight matrices via $W \leftarrow W (I - \hat{v} \hat{v}^\top)$. However, real safety refusal circuits span a high-dimensional subspace $\mathcal{S}_{\text{refusal}} \subset \mathbb{R}^d$ ($k \ge 8$ dimensions). Projecting a single 1D vector causes refusal leakage on complex multi-turn jailbreaks or damages unrelated mathematical capabilities.
- **Proposed Frontier Formulation:** Perform singular value decomposition on the activation difference covariance matrix:
  $$\Sigma_{\text{diff}} = \mathbb{E}\left[ (x_{\text{refuse}} - x_{\text{comply}}) (x_{\text{refuse}} - x_{\text{comply}})^\top \right] = U \Lambda U^\top$$
  Extract the top-$k$ eigenvectors $U_k \in \mathbb{R}^{d \times k}$ and construct the projection operator $P_k = U_k U_k^\top$. The surgical weight transformation is:
  $$\tilde{W} = W (I - P_k)$$
- **Actionable Directive:** Implement and benchmark multi-rank projective abliteration against Many-Shot and Crescendo attacks on frontier open-weight models.

---

## 4. Pillar 3: Manifold Epistemic Calibration via Tangent Bundle Curvature
- **Problem Formulation:** Estimating LLM epistemic uncertainty currently relies on Semantic Entropy (Kuhn et al., Nature 2023), which requires generating $N = 10\text{--}20$ complete stochastic completions, clustering their meanings via an auxiliary NLI model, and computing cluster entropy. This introduces massive $10\times\text{--}20\times$ inference overhead.
- **Proposed Frontier Formulation:** Evaluate local manifold geometry in a single forward pass. Measure the local Riemannian sectional curvature $\kappa(x)$ and local activation density $\rho(x)$ at intermediate transformer layers using nearest-neighbor tangent space projections. Low manifold density and extreme extrinsic curvature directly correlate with hallucination propensity:
  $$\mathcal{U}_{\text{epistemic}}(x) = \frac{\text{Tr}\left( \text{Cov}(J_f(x)) \right)}{\rho_{\text{kNN}}(x)}$$
- **Actionable Directive:** Construct a zero-sampling hallucination detector evaluating hidden states directly during token emission.
