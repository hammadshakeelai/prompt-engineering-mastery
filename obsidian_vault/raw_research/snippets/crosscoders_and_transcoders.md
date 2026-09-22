# Crosscoders, Transcoders & Superposition Geometry (Anthropic 2024–2025)

## 1. Mathematical Geometry of Superposition
Neural networks represent $M$ sparse concepts in $d$ residual dimensions ($M \gg d$).
- **Johnson-Lindenstrauss & Interference:** Exponentially many unit directions $\{v_i\}_{i=1}^M$ exist in $\mathbb{R}^d$ with pairwise inner product $|\langle v_i, v_j \rangle| \le \epsilon$.
- **Interference Thresholding:** When feature activation probability $p \ll 1$, non-linear thresholding acts as an optimal denoiser:
  $$\hat{f}_i = \text{JumpReLU}_{\theta_i}(v_i^T x) = v_i^T x \cdot \mathbb{I}(v_i^T x > \theta_i)$$
- **Polysemanticity:** Arises because canonical basis neurons align with linear combinations of multiple non-orthogonal feature vectors.

## 2. Crosscoder Architecture & Cross-Layer Superposition
Standard per-layer [[sparse_autoencoders|SAEs]] redundantly rediscover persistent residual features across consecutive layers. Anthropic (2024) introduced **Crosscoders** to read and write across multiple layers simultaneously:
- **Multi-Layer Encoder:**
  $$f = \text{ReLU}\left( \sum_{l \in L} W_{\text{enc}}^{(l)} (x^{(l)} - b_{\text{dec}}^{(l)}) + b_{\text{enc}} \right)$$
- **Decoder-Norm Weighted L1 Loss:**
  $$\mathcal{L} = \sum_{l \in L} \|x^{(l)} - \hat{x}^{(l)}\|_2^2 + \lambda \sum_i f_i \left( \sum_{l \in L} \|W_{\text{dec}, i}^{(l)}\|_2 \right)$$
- **Model Diffing:** By training a Crosscoder across $M_{\text{base}}$ and $M_{\text{chat}}$, shared features align decoder vectors while distinct features cleanly isolate post-training safety boundaries, refusal latents, and sycophancy.

## 3. Cross-Layer Transcoders (CLTs)
- **MLP Replacement:** Reconstructs the non-linear MLP output $\text{MLP}(x)$ directly from layer inputs $x$ or preceding residual streams via sparse latents.
- **Attribution Graphs:** Replaces opaque MLP transformations with discrete, sparse feature connections, enabling transparent end-to-end attribution graphs across transformer depth.
