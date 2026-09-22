# Induction Circuit Formation Dynamics & Phase Change (Olsson et al., Anthropic 2022)

## 1. Compositional Mechanics of Induction Heads
Induction heads are specialized attention heads that mechanically implement the algorithmic in-context completion rule: $[A][B] \dots [A] \to [B]$. In a standard transformer stack, this behavior emerges from a minimal **two-head composition circuit** across layers $l_1 < l_2$:

1. **Previous-Token Head ($H_1 \in L_{l_1}$):**
   Attends to position $i$ from position $i+1$, projecting the token representation $x_i$ into the residual stream at $i+1$ via its $OV$ circuit:
   $$\Delta x_{i+1} = x_i W_V^{(1)} W_O^{(1)}$$
2. **Induction Head ($H_2 \in L_{l_2}$):**
   - **$QK$ Subcircuit:** Formulates Query $q_t = x_t W_Q^{(2)}$ at step $t$ and reads Key $k_{i+1} \approx \Delta x_{i+1} W_K^{(2)}$ at prior step $i+1$. The bilinear matching score is:
     $$A_{t, i+1} \propto x_t \left( W_Q^{(2)} W_K^{(2)\top} (W_V^{(1)} W_O^{(1)})^\top \right) x_i^\top$$
     When $x_t = x_i$, this composite product produces high attention alignment, pointing attention precisely to position $i+1$.
   - **$OV$ Subcircuit:** Copies token $x_{i+1}$ directly into the output logits:
     $$\Delta z_t = x_{i+1} W_V^{(2)} W_O^{(2)} W_U$$
     The product matrix $W_U^\top W_O^{(2)} W_V^{(2)} W_U$ exhibits dominant positive real eigenvalues ($\lambda > 0$), acting as a direct copying transformation. (Contrast with [[copy_suppression_attention_heads|Copy Suppression]], which features negative eigenvalues).

## 2. Autocatalytic Phase Change Dynamics
Pretraining loss does not decline smoothly during induction head formation; instead, models undergo an abrupt **macroscopic phase change** ($\sim 2.5\cdot 10^9$ to $10^{10}$ tokens):
- **Feedback Coupling:** Because neither head can complete the induction task without the other, gradients are weak until stochastic updates align $H_1$ to write previous-token info. Once initiated, $H_2$ gains immediate reward for attending to $H_1$'s keys, triggering an exponential feedback loop that rapidly locks the circuit in place.
- **The Training Loss Bump:** A temporary inflection in validation loss coincides with this transition, signaling parameter reorganization from rote n-gram memorization to algorithmic subcircuits.
- **In-Context Generalization:** Across model scales, the sharp formation of induction heads directly correlates with the emergence of few-shot learning, cross-lingual translation, and generalized [[implicit_gradient_descent_icl|in-context learning]].
