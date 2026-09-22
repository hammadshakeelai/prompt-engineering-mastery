# Circuit Breakers & Representation Rerouting (Zou et al. NeurIPS 2024)

## 1. Vulnerability of Token-Level Refusals
Superficial safety alignment (e.g., standard DPO/SFT refusals like "I cannot fulfill this request") only aligns output logits. The underlying internal features corresponding to dangerous capabilities remain fully intact in early and intermediate layers. Consequently, adversarial jailbreaks (GCG, PAIR, token smuggling) easily bypass logit filters by disabling the superficial refusal tokens.

## 2. Representation Rerouting (RR) Mechanics
**Circuit Breakers** alter the model's internal latent representations directly rather than training behavioral token-level refusals:
- **Latent Orthogonalization:** When processing harmful inputs $x_{\text{harm}}$, the intermediate residual activations $r^{(l)}(x_{\text{harm}})$ are forced to become orthogonal to the original capability activations $r_{\text{orig}}^{(l)}(x_{\text{harm}})$:
  $$\mathcal{L}_{\text{reroute}} = \cos\left(r^{(l)}(x_{\text{harm}}), r_{\text{orig}}^{(l)}(x_{\text{harm}})\right) = \frac{\langle r^{(l)}, r_{\text{orig}}^{(l)} \rangle}{\|r^{(l)}\|_2 \|r_{\text{orig}}^{(l)}\|_2}$$
  Minimizing cosine similarity collapses harmful representations into incoherent noise states.
- **Utility Retention Loss:** On benign data $x_{\text{benign}}$, model activations are preserved to maintain standard performance:
  $$\mathcal{L}_{\text{retain}} = \|r^{(l)}(x_{\text{benign}}) - r_{\text{orig}}^{(l)}(x_{\text{benign}})\|_2^2 + \mathcal{L}_{\text{LM}}(x_{\text{benign}})$$
- **Total Training Loss:**
  $$\mathcal{L} = \mathcal{L}_{\text{reroute}} + \lambda \mathcal{L}_{\text{retain}}$$

## 3. Empirical Robustness
- **Jailbreak Invariance:** Successfully neutralized >90% of state-of-the-art jailbreaks across LLaMA-3-8B and Mistral-7B, where standard RLHF and DPO failed under multi-turn adversarial prompt optimization.
- **LoRA Efficiency:** Can be applied via parameter-efficient LoRA adapters trained on specific critical layers ($L/2$ through $3L/4$) in under 30 minutes on a single GPU.
