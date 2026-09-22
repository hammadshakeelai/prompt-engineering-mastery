# DPO Mathematical Formulation vs. PPO

- **DPO Formulation**: Eliminates explicit reward models by substituting the optimal policy closed-form relation into the Bradley-Terry objective:
  $$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$
- **Contrast with PPO**: Directly optimizes preferences via binary cross-entropy on offline pairs without actor-critic rollouts, value networks, or online RL training instabilities.