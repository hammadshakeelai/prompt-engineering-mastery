# Rectified Flow Conditioning & MMDiT Prompt Dynamics (SD3, FLUX 2024–2025)

## 1. Mathematical Foundation: Rectified Flow vs. Classical Diffusion
Classical score-based diffusion models (DDPM, SDEs) model generative trajectories along curved stochastic paths, requiring hundreds of sampling steps or complex ODE solvers (DPM-Solver).
**Rectified Flow (Liu et al., 2023 / Esser et al., 2024)** replaces stochastic diffusion with deterministic straight-line velocity interpolation between standard Gaussian noise $z_0 \sim \mathcal{N}(0, I)$ and target data $z_1 \sim p_{\text{data}}$:
$$z_t = t z_1 + (1 - t) z_0, \quad \frac{d z_t}{d t} = v_\theta(z_t, t, c)$$
Where $v_\theta$ is a neural network predicting the constant velocity vector conditioned on prompt embedding $c$.
Straight trajectories dramatically minimize discretization truncation errors, enabling high-fidelity synthesis in 4–20 steps (or 1–4 steps via distillation).

## 2. MMDiT Architecture: Bidirectional Cross-Modal Routing
In Stable Diffusion 3 and FLUX, text conditioning shifts from passive cross-attention keys/values to **Multimodal Diffusion Transformers (MMDiT)**:
- **Independent Modulation Streams:** Text tokens and image spatial patch tokens maintain separate parameter weights and layer norms.
- **Bidirectional Self-Attention:** Text and image tokens are concatenated into a unified sequence in self-attention layers, allowing image features to contextualize text tokens and vice-versa.
- **Dual Encoder Conditioning:** Blends dense semantic sentence representations from T5-XXL (complex grammar, logic, spelling) with token-aligned visual concepts from CLIP ViT-L.

## 3. Paradigm Shift in Prompt Engineering
1. **Natural Language Syntax Over Tag Soup:** Because MMDiT utilizes T5-XXL bidirectional parsing, comma-separated quality keywords (`"masterpiece, 8k, trending on artstation"`) produce attention saturation and feature interference. Descriptive prose with explicit subject-predicate-object structure yields higher semantic adherence.
2. **Spatial & Relational Grounding:** Bidirectional attention accurately binds compositional attributes (e.g., *"a green cube on top of a red pyramid behind a blue sphere"*) that consistently failed under uncoupled cross-attention backbones.
