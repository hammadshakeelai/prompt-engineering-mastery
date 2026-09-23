# Representation Circuit Breakers: Manifold Disruption (Zou et al., NeurIPS 2024)

## 1. Vulnerability of Linear Refusal Directions
Standard safety alignment (RLHF, DPO) trains language models to output refusal sequences, which [[refusal_direction_geometry|Arditi et al. (ICML 2024)]] proved collapses into a fragile 1D direction $\hat{\mathbf{r}}$ that can be removed via orthogonal projection $(I - \hat{\mathbf{r}}\hat{\mathbf{r}}^\top)$ or bypassed by adversarial token sequences.
Andy Zou et al. (*Improving Alignment and Robustness with Circuit Breakers*, NeurIPS 2024 / arXiv:2406.04313) propose **Representation Circuit Breakers (RCB)** to shatter harmful latent manifolds across intermediate layers rather than adding brittle refusal text.

## 2. Mathematical Formulation of Circuit Breaking Losses
Circuit Breakers train parameters $\theta$ (typically via LoRA on intermediate layers $l \in L_{\text{target}}$, e.g., layers 10 and 20) via a dual-objective loss over harmful ($\mathcal{D}_s$) and benign ($\mathcal{D}_r$) datasets:

### Disruption Loss on Harmful Inputs ($\mathcal{D}_s$)
Drives representations of harmful requests away from the unaligned representation manifold $h_l^{\text{orig}}(x)$:
$$\mathcal{L}_{\text{cb}}(\theta) = \mathbb{E}_{x \in \mathcal{D}_s, l \in L_{\text{target}}} \left[ \text{ReLU}\left( \cos\left( h_l^\theta(x), h_l^{\text{orig}}(x) \right) \right) \right]$$
The $\text{ReLU}$ gate stops optimization once cosine similarity drops to or below zero, preventing inversion artifacts. Alternatively, representations are steered toward a benign anchor $h_l^{\text{target}}(x)$:
$$\mathcal{L}_{\text{rr}}(\theta) = \mathbb{E}_{x \in \mathcal{D}_s, l \in L_{\text{target}}} \left[ 1 - \cos\left( h_l^\theta(x), h_l^{\text{target}}(x) \right) \right]$$

### Utility Retention Loss on Benign Inputs ($\mathcal{D}_r$)
Preserves general reasoning, knowledge, and benchmark fidelity by strictly penalizing $L_2$ representation deviation:
$$\mathcal{L}_{\text{retain}}(\theta) = \mathbb{E}_{x \in \mathcal{D}_r, l \in L_{\text{target}}} \left[ \| h_l^\theta(x) - h_l^{\text{orig}}(x) \|_2^2 \right]$$

### Combined Optimization
$$\mathcal{L}_{\text{RCB}}(\theta) = \mathcal{L}_{\text{cb}}(\theta) + \lambda \mathcal{L}_{\text{retain}}(\theta)$$

## 3. Defense Against Zero-Day Jailbreaks
Because Representation Circuit Breakers destroy the intermediate latent concept manifolds required to execute harmful capabilities, the model is inherently robust to input-level prompt jailbreaks (GCG, AutoDAN, PAIR) and immune to 1D linear direction ablation.
