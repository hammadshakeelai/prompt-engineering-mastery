# Causal Tracing & Null-Space Model Editing (MEMIT to AlphaEdit)

**Key Citations:**
- ROME (Meng et al., NeurIPS 2022) / MEMIT (Meng et al., ICLR 2023)
- AlphaEdit (Fang et al., ICLR 2025 Outstanding Paper Award)

## 1. Causal Mediation Analysis of Transformer Memories
Causal tracing reveals that factual associations $(s, r, o)$ are stored as key-value pairs inside intermediate MLP down-projection weights:
- **Corrupt-and-Restore:** Adding Gaussian noise to subject embeddings disrupts factual recall. Layer-by-layer activation restoration isolates the decisive causal contribution to late-subject MLP layers.
- **Closed-Form Weight Rewriting:** Modifies weights via closed-form linear algebra: $\Delta W = (v^* - W k) (C^{-1} k)^T$, inserting factual edits without gradient descent.

## 2. The Catastrophic Forgetting Breakdown in MEMIT
When performing sequential edits ($N > 1,000$), unconstrained weight perturbations $\Delta W$ accumulate in shared parameter dimensions, corrupting unrelated facts and degrading overall model perplexity.

## 3. AlphaEdit: Null-Space Preservation (ICLR 2025)
AlphaEdit constrains weight updates to the orthogonal complement (null space) of preserved knowledge activations $K_0$:
$$\Delta W \cdot K_0 = 0 \implies (W + \Delta W) K_0 = W K_0$$
Formulated as:
$$\Delta W = \Delta W_{\text{raw}} \left( I - K_0 (K_0^T K_0)^{-1} K_0^T \right)$$
- **Performance:** Outperforms MEMIT by **+36.7%** on sequential editing benchmarks.
- **Invariance:** Provably guarantees zero mathematical alteration to preserved knowledge activations.
