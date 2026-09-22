# Multimodal Sparse Autoencoders & Joint Feature Steering (JSAE) (2024–2026)

## 1. Cross-Modal Feature Entanglement in VLMs
In Vision-Language Models (VLMs, e.g., LLaVA, Qwen2-VL, Gemma-2-Vision), visual patch embeddings $x_v$ and language token embeddings $x_l$ merge into a shared residual stream $h \in \mathbb{R}^d$.
Standard unimodal [[topk_sparse_autoencoders|TopK SAEs]] trained solely on text fail to steer multimodal representations:
- **Projector Non-Linearity:** Modality projectors warp visual tokens, dispersing monosemantic visual concepts across polysemantic superposition directions.
- **Cross-Modal Steering Gap:** Intervening on pure text SAE directions produces semantic dissonance or language hallucinations when conflicting visual tokens persist in the context window.

## 2. Joint Sparse Autoencoders (JSAE) Formulation
Joint Sparse Autoencoders resolve entanglement by training a unified dictionary with explicit cross-modal alignment regularization:
1. **Bimodal Encoder Projections with Shared Decoder:**
   Visual activations $h_v$ and text activations $h_l$ are encoded via modality-specific weights into latent representations $f_v, f_l \in \mathbb{R}^M$, but reconstructed through a shared dictionary $W_d$:
   $$f_v = \operatorname{TopK}\left(W_e^{(v)}(h_v - b) + b_e^{(v)}\right), \quad \hat{h}_v = W_d f_v + b$$
2. **Alignment Regularization:**
   During pretraining on aligned image-caption pairs $(x, y)$, a latent consistency penalty binds visual features to linguistic counterparts:
   $$\mathcal{L}_{\text{JSAE}} = \frac{1}{2}\left(\|h_v - \hat{h}_v\|_2^2 + \|h_l - \hat{h}_l\|_2^2\right) + \lambda_{\text{align}} \sum_{j=1}^M \mathcal{D}_{\text{cos}}\left(f_{v, j}(x), f_{l, j}(y)\right)$$
3. **Select-and-Project (S&P) Activation Steering:**
   Causal control directions $\mathbf{v}_j = W_{d, j}/\|W_{d, j}\|_2$ are projected directly into visual token embeddings:
   $$h_v' = h_v + \alpha \mathbf{v}_j$$
   Clamping $\alpha$ steers fine-grained visual attributes (e.g., color, orientation, object presence) upstream before language generation without weight updates.

## 3. Empirical Results
- **Hallucination Mitigation:** Suppressing over-activated object latents cuts multimodal hallucinations by **42.6%** in medical report generation (e.g., SAE-V).
- **VLA Robot Control:** Applied to Vision-Language-Action models, S&P steering on motion latents successfully controls robotic manipulator trajectories zero-shot via activation intervention.
