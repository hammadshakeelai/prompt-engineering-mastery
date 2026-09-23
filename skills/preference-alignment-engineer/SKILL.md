---
name: preference-alignment-engineer
description: Specialized directive for post-training preference alignment, direct policy optimization (DPO, IPO, KTO, ORPO, CPO, SimPO), reference-free objectives, length bias mitigation, target reward margins, and Bradley-Terry calibration.
---

# Preference Alignment Engineer Skill

Use this skill when designing, training, auditing, or optimizing post-training preference alignment pipelines for large language models, mitigating verbosity/length hacking, implementing reference-free preference loss functions, or establishing rigorous Bradley-Terry preference boundaries.

## 1. Offline Preference Optimization Taxonomy & Loss Objectives

1. **Direct Preference Optimization (DPO, Rafailov et al., NeurIPS 2023):**
   - Eliminates explicit reward model training by parameterizing reward implicitly via the closed-form optimal policy under Bradley-Terry preferences:
     $$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$
   - Loss objective:
     $$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$
   - *Limitation:* Prone to length exploitation (verbosity bias) and requires maintaining $\pi_{\text{ref}}$ permanently in GPU VRAM.

2. **Simple Preference Optimization (SimPO, Meng et al., NeurIPS 2024):**
   - Reference-free alignment objective eliminating $\pi_{\text{ref}}$ and controlling length bias via explicit sequence length normalization and target margins:
     $$\mathcal{L}_{\text{SimPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma \right) \right]$$
   - **Target Margin $\gamma > 0$:** Enforces that winning completions $y_w$ exceed losing completions $y_l$ by at least $\gamma$ in normalized log probability, preventing gradient vanishing on already-separated pairs.
   - Halves training memory overhead by dispensing with the static reference checkpoint.

3. **Contrastive Preference Optimization (CPO, Xu et al., ICML 2024):**
   - Bounded preference formulation paired with supervised fine-tuning regularization to prevent degeneration in high-precision domains (e.g., machine translation, formal syntax):
     $$\mathcal{L}_{\text{CPO}}(\theta) = \mathcal{L}_{\text{DPO\_bound}}(\theta) + \alpha \mathcal{L}_{\text{SFT}}(y_w \mid x)$$
   - Prevents catastrophic forgotten knowledge and hallucinated omissions during preference fine-tuning.

4. **Kahneman-Tversky Optimization (KTO, Ethayarajh et al., ICML 2024):**
   - Operates directly on unpaired binary feedback (thumbs up / thumbs down) using Prospect Theory value functions, avoiding the requirement of strict paired preferences:
     $$v(z) = \begin{cases} 1 - \sigma(\lambda_D (z - z_0)) & \text{if desirable} \\ 1 - \sigma(\lambda_U (z_0 - z)) & \text{if undesirable} \end{cases}$$
   - Penalizes losses more steeply than equivalent gains ($\lambda_U > \lambda_D$).

## 2. Length Bias & Reward Hacking Mitigation Directives

1. **Length Normalization Protocol:**
   - In raw DPO, total sequence log probability grows negatively with token length: $\log \pi(y \mid x) = \sum_{t=1}^{|y|} \log \pi(y_t \mid y_{<t}, x)$.
   - Longer responses frequently acquire spurious preference credit if per-token penalties are mildly smaller than in concise responses.
   - **Mandate:** Apply average per-token log-likelihood normalization $\frac{1}{|y|} \log \pi(y \mid x)$ or length-controlled margin adjustment $\gamma(y_w, y_l) = \gamma_0 + \delta (|y_w| - |y_l|)$.

2. **Conservative Policy Optimization (IPO):**
   - If policy $\pi_\theta$ overfits to preference dataset noise, enforce Identity Preference Optimization:
     $$\mathcal{L}_{\text{IPO}}(\theta) = \mathbb{E}_{(x, y_w, y_l)} \left[ \left( \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} - \frac{\tau}{2} \right)^2 \right]$$
   - Regulates log-ratio gaps directly without sigmoid saturation, avoiding policy degeneration towards deterministic degenerate strings.

## 3. Evaluation & Calibration Checklist

- Always benchmark aligned checkpoints on **Arena-Hard-Auto** and **AlpacaEval 2.0 (Length-Controlled LC Win Rate)** to distinguish true reasoning/instruction gains from cosmetic verbosity expansion.
- Symmetrize all LLM judge evaluations via bidirectional pair swapping ($A$ vs $B$ and $B$ vs $A$) to neutralize position bias.
- Verify log-likelihood margins across token validation splits to detect mode collapse before final model export.
