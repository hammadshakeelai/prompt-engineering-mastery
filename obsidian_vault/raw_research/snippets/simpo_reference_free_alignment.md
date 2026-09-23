# SimPO & CPO: Reference-Free Preference Optimization (Meng et al., NeurIPS 2024 / Xu et al., ICML 2024)

## 1. Vulnerabilities of Standard DPO
Direct Preference Optimization (DPO, Rafailov et al., NeurIPS 2023) reparameterizes the reward under Bradley-Terry preferences:
$$r_{\text{DPO}}(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$
Two persistent failure modes arise:
1. **Verbosity Exploitation (Length Hacking):** Unnormalized sequence log-probabilities $\sum_{t=1}^{|y|} \log \pi(y_t \mid y_{<t}, x)$ inherently favor longer generations, leading to excessive token bloat rather than qualitative reasoning.
2. **Reference Model Memory Footprint:** Maintaining $\pi_{\text{ref}}$ frozen in GPU VRAM consumes $50\%$ of accelerator memory throughout post-training.

## 2. SimPO Formulation & Target Reward Margin
Yu Meng, Mengzhou Xia, and Danqi Chen (*SimPO*, Princeton / NeurIPS 2024 / arXiv:2405.14734) eliminate $\pi_{\text{ref}}$ and align the implicit reward with average per-token log-likelihood:
$$r_{\text{SimPO}}(x, y) = \frac{\beta}{|y|} \log \pi_\theta(y \mid x) = \frac{\beta}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid y_{<t}, x)$$

SimPO introduces a target reward margin $\gamma > 0$ into the loss objective:
$$\mathcal{L}_{\text{SimPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma \right) \right]$$

- **Target Margin $\gamma$:** Unlike DPO where gradients vanish as soon as margin exceeds zero, $\gamma$ forces winning completions $y_w$ to beat losing completions $y_l$ by at least $\gamma$, preventing premature convergence.
- **Length Normalization:** Explicitly divides by $|y|$, preventing length hacking and producing concise, dense reasoning chains.

## 3. Contrastive Preference Optimization (CPO)
In high-precision generation (machine translation, code syntax), Haoran Xu et al. (*CPO*, ICML 2024 / arXiv:2401.08417) integrate supervised fine-tuning regularization:
$$\mathcal{L}_{\text{CPO}}(\theta) = \mathcal{L}_{\text{DPO\_bound}}(\theta) + \alpha \mathcal{L}_{\text{SFT}}(y_w \mid x)$$
preventing distribution collapse and hallucinated omissions on gold-standard outputs.

## 4. Empirical Performance
- **AlpacaEval 2.0:** $+6.4\%$ length-controlled win rate boost over DPO while generating $18\%$ shorter responses.
- **VRAM Savings:** Halves post-training GPU memory requirements by completely eliminating reference model checkpoints.
