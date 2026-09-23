# Self-Play Preference Optimization (SPPO) & Nash Equilibrium Alignment (Wu et al., ICML 2024)

## 1. Intransitivity & Condorcet Cycles in Preferences
Standard alignment objectives (RLHF, DPO, SimPO) assume a scalar reward function exists under the Bradley-Terry model. However, empirical human preferences frequently exhibit **intransitivity and Condorcet cycles** ($A \succ B \succ C \succ A$). Forcing cyclic graphs into a scalar reward induces reward hacking and policy oscillation.

## 2. Two-Player Zero-Sum Game Formulation
Yue Wu et al. (*SPPO*, UCLA / ICML 2024 / arXiv:2405.00675) frame alignment as finding the **Nash Equilibrium policy $\pi^*$** of a symmetric constant-sum game:
$$\max_{\pi_1} \min_{\pi_2} \mathbb{E}_{x, y_1 \sim \pi_1, y_2 \sim \pi_2} \left[ \mathcal{P}(y_1 \succ y_2 \mid x) - \frac{1}{2} \right]$$
At the Nash equilibrium, no opponent policy can achieve a win rate above $50\%$, guaranteeing immunity to cyclical exploitation.

## 3. Multiplicative Weights Update (MWU) Protocol
1. Policy samples $K$ self-play candidate completions $\{y^{(1)}, \dots, y^{(K)}\} \sim \pi_k(\cdot \mid x)$.
2. Oracle judge computes pairwise advantage matrix $A_{i, j} = \mathcal{P}(y^{(i)} \succ y^{(j)} \mid x) - \frac{1}{2}$.
3. Target mixture distribution is updated via mirror descent / MWU:
   $$p_{k+1}(y^{(i)} \mid x) \propto p_k(y^{(i)} \mid x) \exp\left( \eta \bar{A}_i \right)$$
   where $\bar{A}_i = \frac{1}{K} \sum_j A_{i, j}$.
4. Policy parameters $\theta_{k+1}$ minimize cross-entropy loss against target weights:
   $$\mathcal{L}_{\text{SPPO}}(\theta) = -\mathbb{E} \left[ \sum_{i=1}^K p_{k+1}(y^{(i)} \mid x) \log \pi_\theta(y^{(i)} \mid x) \right]$$

## 4. Benchmark Performance
Guarantees $\mathcal{O}(1/\sqrt{T})$ convergence to Nash equilibrium. Delivers **$28.53\%$ win-rate on AlpacaEval 2.0** on Mistral-7B without external model distillation.
