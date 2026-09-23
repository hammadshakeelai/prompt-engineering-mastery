# EAGLE-2: Recurrent Feature Drafting & Dynamic Draft Trees (Li et al., EMNLP 2024)

## 1. The Inefficiency of Static Speculative Trees
Standard tree-based speculative decoding engines (e.g., SpecInfer, Medusa, EAGLE-1) employ static draft trees with predefined depth and branching factors. However, token predictability varies by context: static trees over-branch in deterministic contexts and over-speculate deep chains in high-entropy contexts.
EAGLE-2 (Li et al., EMNLP 2024 / arXiv:2406.16858) introduces context-aware dynamic draft trees driven by calibrated draft model confidence.

## 2. Feature-Level Drafting & Dynamic Priority Queue Expansion
1. **Recurrent Feature Prediction:**
   Instead of an independent model, EAGLE-2 uses a single lightweight transformer decoder layer $\mathcal{M}_{\text{draft}}$ that autoregressively predicts top-layer target hidden states $\hat{f}_{t+m} \in \mathbb{R}^d$:
   $$\hat{f}_{t+m} = \mathcal{M}_{\text{draft}}\left(\hat{f}_{t+m-1}, e(x_{t+m-1})\right)$$
   Draft probabilities are computed through the frozen target unembedding matrix $W_U$.

2. **Calibrated Confidence as Acceptance Proxy:**
   Draft confidence $s(v) = \max_v P_d(v \mid \hat{f})$ reliably approximates target acceptance probability $\alpha(v)$.

3. **Dynamic Tree Expansion:**
   Maintains a priority queue where each tree path $p = (v_1, \dots, v_m)$ is ranked by cumulative confidence:
   $$S(p) = \prod_{i=1}^m s(v_i)$$
   - **Low-Entropy Sequences:** Expands deep linear chains ($7\text{--}10$ tokens).
   - **High-Entropy Sequences:** Expands shallow, diverse candidate branches.

4. **Tree Attention Verification:**
   Verifies all tree nodes in a single forward pass of the target model using a 2D causal ancestor mask:
   $$M_{i,j} = \mathbb{I}(j \in \text{Ancestors}(i))$$

## 3. Empirical Results
- Achieves **$3.05\times\text{--}4.26\times$ wall-clock speedup** over non-speculative autoregression across LLaMA-2/3, Mistral, and Mixtral.
- Delivers a **$20\%\text{--}40\%$ speedup improvement** over EAGLE-1 with mathematically lossless distribution preservation.
