# Strategic Research Pivot 07: Cross-Layer Attention, Circuit Breakers & Diffusion Topologies

## 1. Executive Summary & Pivot Directive
Having formalized Mamba-2 State Space Duality (SSD), LK loss direct acceptance optimization, and Copy Suppression attention head circuits, this pivot establishes three frontier research horizons for systems efficiency, mechanistic safety alignment, and non-autoregressive reasoning:

---

## 2. Three Novel Research Horizons

### Horizon 1: Cross-Layer Attention (CLA) & Inter-Layer KV Sharing (Brandon et al., NeurIPS 2024)
- **Problem Statement:** Grouped-Query Attention (GQA) and Multi-Query Attention (MQA) reduce the number of KV heads along the head dimension, but still allocate separate KV representations across all $L$ layers ($\mathcal{O}(L \cdot T)$).
- **Frontier Architecture:** *Cross-Layer Attention (CLA)* (Brandon et al., MIT / NeurIPS 2024). Shares Key-Value projections across adjacent transformer layers (e.g., pairs $2l$ and $2l+1$), delivering an additional **$2\times$ memory reduction** beyond MQA with negligible perplexity difference.
- **Focus Area:** Combining CLA with 2-bit quantization (KIVI) and matrix absorption (MLA) to achieve sub-gigabyte KV footprints for 70B+ models.

### Horizon 2: Representation Disruption & Mechanistic Circuit Breakers (Zou et al., NeurIPS 2024)
- **Problem Statement:** Post-hoc refusal training (RLHF, DPO) trains models to emit superficial rejection tokens (`"I cannot..."`), leaving dangerous latent capabilities fully intact in middle-layer residual streams and vulnerable to activation steering, token smuggling, or LoRA unlearning.
- **Frontier Architecture:** *Circuit Breakers* (Zou et al., Gray Swan AI / UC Berkeley, NeurIPS 2024). Intervenes directly in the internal residual stream geometry, training representation disruption loss that scrambles internal concepts whenever harmful activations are primed, rendering latent representations unexecutable.
- **Focus Area:** Mechanistic evaluation of representation scramble metrics and resistance against white-box weight editing attacks.

### Horizon 3: Non-Autoregressive Trajectory Denoising & Diffusion LLMs (Plaid & SEDD)
- **Problem Statement:** Autoregressive generation forces models to plan tokens in a strict left-to-right unidirectional sequence, preventing global plan backtracking and iterative refinement.
- **Frontier Architecture:** Discrete and continuous text diffusion models (SEDD, Plaid, Lookahead-then-Verify Diffusion). Models formulate reasoning as iterative denoising of all trajectory tokens simultaneously.
- **Focus Area:** Grammar-constrained denoising loops and multi-step plan verification.

---

## 3. Directives for Immediate Research Execution
1. Append Section 36 to `latent_mechanics_dossier.md` on **Cross-Layer Attention (CLA) & Inter-Layer KV Cache Sharing (Brandon et al., NeurIPS 2024)**.
2. Author atomic notes for **Cross-Layer Attention (CLA)** and **Circuit Breakers Representation Disruption**.
3. Rebuild `000_Master_Brain_Index.md` and push persistently to GitHub `origin/main`.
