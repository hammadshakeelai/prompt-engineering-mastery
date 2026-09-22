# Copy Suppression in Attention Heads (McDougall et al., 2023)

## 1. The Over-Copying Dilemma & Negative Heads
In-context learning depends heavily on [[transformer_circuits_and_induction_heads|Induction Heads]] copying tokens from context. However, uninhibited induction leads to repetition loops and overconfidence.
Language models contain **Copy Suppression Heads** (e.g., L10H7 in GPT-2 Small, with counterparts across modern LLMs) that actively downvote candidate tokens that appear earlier in the context.

## 2. Two-Stage Circuit Mechanics
1. **$QK$ Query-Key Routing:**
   The query matches tokens currently predicted by earlier layers. When candidate token $t$ is primed, the head attends directly to prior occurrences of $t$ in the prompt history:
   $$A_{\text{dest}, \text{src}} \propto \exp\left( \frac{x_{\text{dest}}^\top W_Q^\top W_K x_{\text{src}}}{\sqrt{d}} \right)$$
2. **Negative $OV$ Unembedding Projection:**
   Projecting the attended state through value and output weights into vocabulary unembedding space $W_U$:
   $$M = W_U^\top W_O W_V W_U \in \mathbb{R}^{|V| \times |V|}$$
   The diagonal entries are strictly negative: $\text{diag}(M)_{i, i} \ll 0$.
   Thus, attending to token $t$ directly subtracts from its logit:
   $$\Delta z_t = (W_U[:, t])^\top W_O W_V x_{\text{src}} < 0$$

## 3. Calibration & The "Self-Repair" Mechanism
- **Calibration Balancing:** Acts as a Bayesian damper, preventing premature convergence on copied context tokens.
- **The Self-Repair Illusion:** Ablating upstream copying heads does not cause downstream performance collapse. This is not active network compensation, but passive circuit equilibrium: without upstream copy activations, copy suppression heads stop firing, releasing their negative dampening automatically.
