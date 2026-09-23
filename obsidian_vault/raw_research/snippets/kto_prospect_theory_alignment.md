# Kahneman-Tversky Optimization (KTO): Prospect Utility (Ethayarajh et al., ICML 2024)

## 1. Unpaired Production Telemetry vs. Counterfactual Pairing
Direct Preference Optimization (DPO) requires paired counterfactuals $(x, y_w, y_l)$. Real-world telemetry consists of unpaired binary signals (upvotes/downvotes). Forcing unpaired data into synthetic pairs creates intransitive loops and noisy labels.

## 2. Prospect Theory Value Formulation
KTO (Ethayarajh et al., Stanford / Contextual AI / ICML 2024 / arXiv:2402.01306) formalizes alignment using Kahneman-Tversky Prospect Theory:
$$\mathcal{L}_{\text{KTO}} = \lambda_D \mathbb{E}_{\mathcal{D}_D} \left[ \sigma\left(-\beta \left(\log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - z_0\right)\right) \right] + \lambda_U \mathbb{E}_{\mathcal{D}_U} \left[ \sigma\left(\beta \left(\log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - z_0\right)\right) \right]$$
- **Dynamic Reference Point $z_0$:** $z_0 = \mathbb{E}[\text{KL}(\pi_\theta \parallel \pi_{\text{ref}})]$. Rewards completions only when they outperform the policy's average divergence.
- **Loss Aversion ($\lambda_U > \lambda_D$):** Penalizes undesirable completions more aggressively than rewarding desirable completions ($\lambda_U \in [1.0, 1.33]$).

## 3. Empirical Supremacy
Matches or outperforms DPO on 1B–30B models (Llama-3, Mistral) on AlpacaEval 2 without requiring paired preference data.
