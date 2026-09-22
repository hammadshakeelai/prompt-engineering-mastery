# Self-Play Fine-Tuning (SPIN, Chen et al. ICML 2024)

## 1. Paradigm Shift: Alignment via Self-Play
Traditional preference alignment (RLHF, DPO) requires external supervision—either human preference pairs or stronger teacher models (e.g., UltraFeedback). **Self-Play Fine-Tuning (SPIN)** converts a weak LLM into a strong LLM exclusively using an existing SFT dataset without collecting new human annotations or hiring stronger judges.

## 2. Two-Player Game Formulation
SPIN frames alignment as a two-player zero-sum game between the target model $\pi_\theta$ (the main player) and its previous checkpoint $\pi_{\theta_t}$ (the opponent):
- **Opponent Generation:** At iteration $t$, the opponent model generates synthetic responses $\tilde{y} \sim \pi_{\theta_t}(\cdot \mid x)$ for prompt $x$ from the supervised dataset $\mathcal{D}_{\text{SFT}}$.
- **Main Player Discriminator Objective:** The target model updates parameters $\theta$ to distinguish human ground-truth responses $y$ from self-generated completions $\tilde{y}$:
  $$\mathcal{L}_{\text{SPIN}}(\pi_\theta; \pi_{\theta_t}) = -\mathbb{E}_{(x, y) \sim \mathcal{D}, \tilde{y} \sim \pi_{\theta_t}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\theta_t}(y \mid x)} - \beta \log \frac{\pi_\theta(\tilde{y} \mid x)}{\pi_{\theta_t}(\tilde{y} \mid x)} \right) \right]$$

## 3. Convergence & Theoretical Guarantees
- **Nash Equilibrium:** The game converges to the unique Nash equilibrium when $\pi_\theta(y \mid x) = p_{\text{data}}(y \mid x)$, meaning the model's output distribution becomes indistinguishable from the ground-truth distribution.
- **Iterative Performance Gains:** On Open LLM Leaderboard (GSM8K, HumanEval, ARC, MMLU), applying 3 iterations of SPIN onto Zephyr-7B-SFT improved average benchmark performance from 58.14 to 63.16, outperforming models fine-tuned with 50k external DPO preference pairs.
